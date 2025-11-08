import hashlib
import logging
import os
import shutil
import tempfile
import traceback
from typing import Any, Dict, Generator, Optional, Tuple

import gradio as gr
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 从上级目录导入
from core.agent_builder import build_agent
from core.llm_services import embeddings

# 配置日志
logger = logging.getLogger(__name__)


def _default_session_state() -> Dict[str, Any]:
    return {
        "rag_retriever": None,
        "agent_executor": None,
        "last_index_cache_dir": None,
        "last_file_key": None,
        "agent_built_for_file_key": None,
    }


def _get_default_base_cache_dir() -> str:
    env_dir = os.getenv("RAG_FAISS_DIR")
    if env_dir and env_dir.strip():
        return os.path.expanduser(env_dir.strip())
    return os.path.join(os.path.expanduser("~"), ".rag_faiss_cache")


def _get_base_cache_dir(persist_dir_text: Optional[str]) -> str:
    base = (persist_dir_text or "").strip()
    if base:
        return os.path.expanduser(base)
    return _get_default_base_cache_dir()


def _file_cache_key(file_path: str) -> str:
    try:
        stat = os.stat(file_path)
        raw = f"{os.path.abspath(file_path)}::{stat.st_size}::{int(stat.st_mtime)}"
    except Exception:
        raw = os.path.abspath(file_path)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _embeddings_fingerprint(obj: Any) -> str:
    parts = [type(obj).__name__]
    for attr in [
        "model",
        "model_name",
        "deployment",
        "deployment_name",
        "base_url",
        "endpoint",
        "encoding",
        "dimension",
    ]:
        val = getattr(obj, attr, None)
        if val is not None:
            parts.append(f"{attr}={val}")
    try:
        parts.append(repr(obj))
    except Exception:
        pass
    data = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha1(data).hexdigest()


def _faiss_cache_dir_for(file_key: str, emb_fp: str, base_dir: str) -> str:
    base = base_dir
    os.makedirs(base, exist_ok=True)
    # 目录名包含 embeddings 指纹，避免 embeddings 变化导致错配
    return os.path.join(base, f"{file_key}__{emb_fp[:12]}")
def _ocr_pdf_to_temp(input_pdf_path: str) -> Optional[str]:
    """Try to OCR a PDF to a temporary searchable PDF. Returns path or None.
    Uses ocrmypdf Python API if available, else tries the `ocrmypdf` CLI.
    """
    try:
        # Prefer Python API if installed
        import ocrmypdf  # type: ignore
        tmp_dir = tempfile.mkdtemp(prefix="ocrpdf_")
        output_pdf_path = os.path.join(tmp_dir, "ocr_output.pdf")
        try:
            ocrmypdf.ocr(
                input_file=input_pdf_path,
                output_file=output_pdf_path,
                force_ocr=True,
                deskew=True,
                optimize=1,
                progress_bar=False,
            )
            if os.path.isfile(output_pdf_path) and os.path.getsize(output_pdf_path) > 0:
                return output_pdf_path
        except Exception:
            traceback.print_exc()
    except Exception:
        pass

    # Fallback to CLI if available
    try:
        import shutil as _sh
        if _sh.which("ocrmypdf"):
            tmp_dir = tempfile.mkdtemp(prefix="ocrpdf_")
            output_pdf_path = os.path.join(tmp_dir, "ocr_output.pdf")
            import subprocess
            cmd = [
                "ocrmypdf",
                "--force-ocr",
                "--deskew",
                "--optimize", "1",
                input_pdf_path,
                output_pdf_path,
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if os.path.isfile(output_pdf_path) and os.path.getsize(output_pdf_path) > 0:
                    return output_pdf_path
            except Exception:
                traceback.print_exc()
    except Exception:
        pass

    return None

def _load_pdf_with_fallbacks(file_path: str):
    """Load PDF documents using multiple strategies to handle scanned or tricky PDFs.
    Returns a list[Document].
    """
    documents = []
    # 1) Try PyPDFLoader
    try:
        try:
            from langchain_community.document_loaders import PyPDFLoader
        except Exception:
            # fallback older path (some versions expose under .pdf)
            from langchain_community.document_loaders.pdf import PyPDFLoader  # type: ignore
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        if documents:
            return documents
    except Exception:
        traceback.print_exc()

    # 2) Try PDFPlumberLoader if available
    try:
        from langchain_community.document_loaders import PDFPlumberLoader
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        if documents:
            return documents
    except Exception:
        pass

    # 3) Try PyMuPDFLoader (fitz) if available
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        if documents:
            return documents
    except Exception:
        pass

    # 4) Lightweight fallback using pypdf directly
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        texts = []
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt.strip():
                texts.append(txt)
        if texts:
            # Minimal Document structure to be compatible with splitters
            from langchain_core.documents import Document
            return [Document(page_content=t) for t in texts]
    except Exception:
        pass

    # 5) Try OCR to make it searchable, then reload
    try:
        ocr_path = _ocr_pdf_to_temp(file_path)
        if ocr_path:
            # Re-run fast loaders on OCR'd PDF
            try:
                from langchain_community.document_loaders import PDFPlumberLoader
                loader = PDFPlumberLoader(ocr_path)
                documents = loader.load()
                if documents:
                    return documents
            except Exception:
                pass
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(ocr_path)
                documents = loader.load()
                if documents:
                    return documents
            except Exception:
                pass
            # Fallback again to pypdf
            try:
                from pypdf import PdfReader
                reader = PdfReader(ocr_path)
                texts = []
                for page in reader.pages:
                    try:
                        txt = page.extract_text() or ""
                    except Exception:
                        txt = ""
                    if txt.strip():
                        texts.append(txt)
                if texts:
                    from langchain_core.documents import Document
                    return [Document(page_content=t) for t in texts]
            except Exception:
                pass
    except Exception:
        traceback.print_exc()

    return []



def _try_load_faiss_from_cache(cache_dir: str) -> Optional[FAISS]:
    try:
        if os.path.isdir(cache_dir):
            return FAISS.load_local(cache_dir, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        traceback.print_exc()
    return None


def _save_faiss_to_cache(cache_dir: str, vs: FAISS) -> None:
    try:
        os.makedirs(cache_dir, exist_ok=True)
        vs.save_local(cache_dir)
    except Exception:
        traceback.print_exc()


def _safe_clear_directory(dir_path: str) -> None:
    try:
        if not os.path.isdir(dir_path):
            return
        # 尝试整体删除；失败则逐项删除
        try:
            shutil.rmtree(dir_path)
        except Exception:
            for name in os.listdir(dir_path):
                p = os.path.join(dir_path, name)
                try:
                    if os.path.isdir(p):
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        os.remove(p)
                except Exception:
                    traceback.print_exc()
        # 重新创建空目录
        os.makedirs(dir_path, exist_ok=True)
    except Exception:
        traceback.print_exc()


def process_uploaded_file(
    file_obj,
    state: Dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    persist_dir_text: str,
    progress: gr.Progress = None,
) -> Generator[Tuple[str, Any, Dict[str, Any]], None, None]:
    """
    处理上传的文件，构建向量索引
    
    Args:
        file_obj: 上传的文件对象（支持多选）
        state: 会话状态
        chunk_size: 文本切分大小
        chunk_overlap: 文本切分重叠
        top_k: 检索条数
        persist_dir_text: 持久化目录
        
    Yields:
        (状态消息, UI更新, 状态字典) 元组
    """
    if not file_obj:
        logger.warning("未提供文件对象")
        if progress:
            progress(0.0, desc="错误：请先上传文件")
        status_html = "<div style='padding: 15px; border-radius: 8px; background-color: #fff4e6; border: 2px solid #FF9800; text-align: center; font-size: 14px;'>⚠️ <b>提示：</b>请先上传文件（支持多选 TXT/PDF/DOCX 格式）</div>"
        result_html = "<div style='padding: 20px; border-radius: 10px; background-color: #ffe6e6; border: 3px solid #ff4444; text-align: center; font-size: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>❌ <b style='font-size: 18px; color: #d32f2f;'>上传失败</b><br><br><span style='font-size: 14px;'>请先选择文件后再点击处理按钮</span></div>"
        yield status_html, result_html, gr.update(interactive=True), state
        return

    try:
        logger.info(f"开始处理上传的文件，chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, top_k={top_k}")
        # 统一为列表处理
        files = file_obj if isinstance(file_obj, list) else [file_obj]

        base_cache_dir = _get_base_cache_dir(persist_dir_text)
        emb_fp = _embeddings_fingerprint(embeddings)

        # 多文件使用会话级合并目录；单文件仍使用文件特定目录
        if len(files) > 1:
            cache_dir = os.path.join(base_cache_dir, f"combined__{emb_fp[:12]}")
            file_key = "combined"
        else:
            try:
                single_path = files[0].name
            except Exception:
                single_path = str(files[0])
            file_key = _file_cache_key(single_path)
            cache_dir = _faiss_cache_dir_for(file_key, emb_fp, base_cache_dir)

        vs = _try_load_faiss_from_cache(cache_dir)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        total_chunks = 0
        result_html = ""  # 初始化结果提示为空
        
        if vs:
            logger.info(f"从缓存加载向量索引: {cache_dir}")
            if progress:
                progress(0.1, desc="已从缓存加载向量索引")
            status_html = "<div style='padding: 15px; border-radius: 8px; background-color: #e6f3ff; border: 2px solid #4CAF50; text-align: center; font-size: 14px;'>📦 <b>步骤 1/4：</b>已从缓存加载向量索引</div>"
            yield status_html, result_html, gr.update(interactive=False), state
        else:
            logger.info(f"开始加载 {len(files)} 个文件")
            if progress:
                progress(0.05, desc="正在加载文档...")
            file_names = ", ".join([os.path.basename(f.name if hasattr(f, 'name') else str(f)) for f in files[:3]])
            if len(files) > 3:
                file_names += f" 等 {len(files)} 个文件"
            status_html = f"<div style='padding: 15px; border-radius: 8px; background-color: #fff4e6; border: 2px solid #FF9800; text-align: center; font-size: 14px;'>📄 <b>步骤 1/4：</b>正在加载文档...<br><small>文件：{file_names}</small></div>"
            yield status_html, result_html, gr.update(interactive=False), state

            total_files = len(files)
            for file_idx, f in enumerate(files):
                try:
                    file_path = f.name
                except Exception:
                    file_path = str(f)
                file_ext = os.path.splitext(file_path)[1].lower()
                file_name = os.path.basename(file_path)

                # 更新进度：加载文档阶段 (10% - 30%)
                if progress:
                    progress(0.1 + (file_idx / total_files) * 0.2, desc=f"正在加载文档 ({file_idx + 1}/{total_files}): {file_name}")

                # 文件大小提示（尽力而为）
                try:
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    if size_mb > 50:
                        warning_html = f"<div style='padding: 10px; border-radius: 5px; background-color: #fff3cd; border-left: 4px solid #ffc107;'>⚠️ <b>提示：</b>{file_name} 文件较大（约{size_mb:.1f}MB），处理可能较慢</div>"
                        yield warning_html, gr.update(interactive=False), state
                except Exception:
                    pass

                # 加载
                if file_ext == ".pdf":
                    documents = _load_pdf_with_fallbacks(file_path)
                    if not documents:
                        continue
                elif file_ext == ".txt":
                    loader = TextLoader(file_path, encoding="utf-8")
                    documents = loader.load()
                elif file_ext == ".docx":
                    loader = Docx2txtLoader(file_path)
                    documents = loader.load()
                else:
                    warning_html = f"<div style='padding: 10px; border-radius: 5px; background-color: #fff3cd; border-left: 4px solid #ffc107;'>⚠️ <b>跳过：</b>不支持的文件格式 {file_ext}<br><small>文件：{file_name}</small></div>"
                    yield warning_html, gr.update(interactive=False), state
                    continue

                # 更新进度：切分文档阶段 (30% - 50%)
                if progress:
                    progress(0.3 + (file_idx / total_files) * 0.2, desc=f"正在切分文档 ({file_idx + 1}/{total_files}): {file_name}")

                # 切分
                status_html = f"<div style='padding: 15px; border-radius: 8px; background-color: #fff4e6; border: 2px solid #FF9800; text-align: center; font-size: 14px;'>✂️ <b>步骤 2/4：</b>正在切分文档...<br><small>文件：{file_name} ({file_idx + 1}/{total_files})</small></div>"
                yield status_html, result_html, gr.update(interactive=False), state
                chunks = splitter.split_documents(documents)
                if not chunks:
                    continue
                total_chunks += len(chunks)

                # 更新进度：创建向量索引阶段 (50% - 80%)
                if progress:
                    progress(0.5 + (file_idx / total_files) * 0.3, desc=f"正在生成向量索引 ({file_idx + 1}/{total_files}): {file_name}")

                # 建索引/追加
                if vs is None:
                    logger.info(f"创建向量索引，包含 {len(chunks)} 个片段")
                    status_html = f"<div style='padding: 15px; border-radius: 8px; background-color: #fff4e6; border: 2px solid #FF9800; text-align: center; font-size: 14px;'>🔍 <b>步骤 3/4：</b>正在创建向量索引...<br><small>生成嵌入向量中，请稍候（包含 {len(chunks)} 个文本片段）</small></div>"
                    yield status_html, result_html, gr.update(interactive=False), state
                    vs = FAISS.from_documents(chunks, embeddings)
                else:
                    logger.info(f"追加 {len(chunks)} 个片段到现有索引")
                    status_html = f"<div style='padding: 15px; border-radius: 8px; background-color: #fff4e6; border: 2px solid #FF9800; text-align: center; font-size: 14px;'>➕ <b>步骤 3/4：</b>追加到向量索引...<br><small>文件：{file_name}（{len(chunks)} 个片段）</small></div>"
                    yield status_html, result_html, gr.update(interactive=False), state
                    vs.add_documents(chunks)

            if vs is None:
                if progress:
                    progress(1.0, desc="处理失败：未能提取文本内容")
                status_html = "<div style='padding: 15px; border-radius: 8px; background-color: #ffe6e6; border: 2px solid #ff4444; text-align: center; font-size: 14px;'>❌ <b>处理失败：</b>未能从所选文件中提取任何文本内容</div>"
                result_html = "<div style='padding: 20px; border-radius: 10px; background-color: #ffe6e6; border: 3px solid #ff4444; text-align: center; font-size: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>❌ <b style='font-size: 18px; color: #d32f2f;'>上传失败</b><br><br><span style='font-size: 14px;'>未能从文件中提取文本内容</span><br><small style='color: #666; margin-top: 10px; display: block;'>💡 提示：若为扫描版PDF，请先进行OCR处理（例如使用OCRmyPDF）后再尝试</small></div>"
                yield status_html, result_html, gr.update(interactive=True), state
                return

            _save_faiss_to_cache(cache_dir, vs)
            logger.info(f"向量索引已保存到: {cache_dir}")

        # 更新进度：构建 Agent 阶段 (80% - 95%)
        if progress:
            progress(0.85, desc="正在构建智能 Agent...")

        # 创建检索器和 Agent
        try:
            logger.info(f"创建检索器，top_k={top_k}")
            status_html = "<div style='padding: 15px; border-radius: 8px; background-color: #fff4e6; border: 2px solid #FF9800; text-align: center; font-size: 14px;'>🤖 <b>步骤 4/4：</b>正在构建智能 Agent...</div>"
            yield status_html, result_html, gr.update(interactive=False), state
            
            state["rag_retriever"] = vs.as_retriever(search_kwargs={"k": int(top_k)})
            logger.info("正在构建 Agent...")
            state["agent_executor"] = build_agent(state["rag_retriever"])
            state["last_index_cache_dir"] = cache_dir
            state["last_file_key"] = file_key
            state["agent_built_for_file_key"] = file_key
            logger.info("文件处理完成，Agent 已更新")
        except Exception as e:
            logger.error(f"构建 Agent 失败: {e}", exc_info=True)
            if progress:
                progress(1.0, desc="处理失败")
            status_html = f"<div style='padding: 15px; border-radius: 8px; background-color: #ffe6e6; border: 2px solid #ff4444; text-align: center; font-size: 14px;'>⚠️ <b>部分成功：</b>文件处理成功，但 Agent 构建失败</div>"
            result_html = f"<div style='padding: 20px; border-radius: 10px; background-color: #fff3cd; border: 3px solid #ffc107; text-align: center; font-size: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>⚠️ <b style='font-size: 18px; color: #f57c00;'>部分成功</b><br><br><span style='font-size: 14px;'>文件处理成功，但 Agent 构建失败</span><br><small style='color: #666; margin-top: 10px; display: block;'>错误：{str(e)}<br>💡 提示：请检查网络连接或稍后重试</small></div>"
            yield status_html, result_html, gr.update(interactive=True), state
            return

        # 更新进度：完成 (100%)
        if progress:
            progress(1.0, desc="✅ 处理完成！上传成功")

        if len(files) > 1:
            status_html = f"<div style='padding: 15px; border-radius: 8px; background-color: #e8f5e9; border: 2px solid #4CAF50; text-align: center; font-size: 14px;'>✅ <b>处理完成！</b>已处理并合并 {len(files)} 个文件</div>"
            result_html = f"<div style='padding: 25px; border-radius: 10px; background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); border: 3px solid #4CAF50; text-align: center; font-size: 16px; box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);'>✅ <b style='font-size: 22px; color: #2e7d32;'>上传成功！</b><br><br><div style='font-size: 15px; margin: 15px 0;'>已处理并合并 <b style='color: #1b5e20;'>{len(files)}</b> 个文件<br>共生成 <b style='color: #1b5e20;'>{total_chunks}</b> 个文本片段</div><div style='margin-top: 15px; padding-top: 15px; border-top: 2px solid #4CAF50;'><span style='font-size: 18px;'>🎉</span> <b style='color: #2e7d32;'>现在可以开始提问了！</b></div></div>"
            logger.info(f"已处理并合并 {len(files)} 个文件（新增 {total_chunks} 个片段）")
            yield status_html, result_html, gr.update(interactive=True), state
        else:
            status_html = f"<div style='padding: 15px; border-radius: 8px; background-color: #e8f5e9; border: 2px solid #4CAF50; text-align: center; font-size: 14px;'>✅ <b>处理完成！</b>文件 '{os.path.basename(single_path)}' 已处理</div>"
            result_html = f"<div style='padding: 25px; border-radius: 10px; background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); border: 3px solid #4CAF50; text-align: center; font-size: 16px; box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);'>✅ <b style='font-size: 22px; color: #2e7d32;'>上传成功！</b><br><br><div style='font-size: 15px; margin: 15px 0;'>文件 <b style='color: #1b5e20;'>'{os.path.basename(single_path)}'</b> 处理完成<br>共生成 <b style='color: #1b5e20;'>{total_chunks}</b> 个文本片段</div><div style='margin-top: 15px; padding-top: 15px; border-top: 2px solid #4CAF50;'><span style='font-size: 18px;'>🎉</span> <b style='color: #2e7d32;'>现在可以开始提问了！</b></div></div>"
            logger.info(f"文件 '{os.path.basename(single_path)}' 处理成功！")
            yield status_html, result_html, gr.update(interactive=True), state

    except Exception as e:
        if progress:
            progress(1.0, desc="处理失败")
        status_html = f"<div style='padding: 15px; border-radius: 8px; background-color: #ffe6e6; border: 2px solid #ff4444; text-align: center; font-size: 14px;'>❌ <b>处理失败：</b>{str(e)[:50]}...</div>"
        result_html = f"<div style='padding: 25px; border-radius: 10px; background: linear-gradient(135deg, #ffe6e6 0%, #ffcdd2 100%); border: 3px solid #ff4444; text-align: center; font-size: 16px; box-shadow: 0 4px 12px rgba(244, 67, 54, 0.3);'>❌ <b style='font-size: 22px; color: #d32f2f;'>上传失败</b><br><br><div style='font-size: 15px; margin: 15px 0; color: #c62828;'>{str(e)}</div><div style='margin-top: 15px; padding-top: 15px; border-top: 2px solid #ff4444;'><small style='color: #666;'>💡 提示：请检查文件格式是否正确，或查看日志获取更多信息</small></div></div>"
        logger.error(f"处理失败: {e}", exc_info=True)
        yield status_html, result_html, gr.update(interactive=True), state


def chat_with_agent(question: str, history: list, state: Dict[str, Any]) -> str:
    """
    与 Agent 进行对话
    
    Args:
        question: 用户问题
        history: 对话历史
        state: 会话状态
        
    Returns:
        Agent 的回答
    """
    if not question or not question.strip():
        logger.warning("收到空问题")
        return "请输入有效的问题。"
    
    logger.info(f"收到问题: {question[:100]}...")
    
    agent = state.get("agent_executor")
    retriever = state.get("rag_retriever")
    last_file_key = state.get("last_file_key")
    built_for_key = state.get("agent_built_for_file_key")

    needs_rebuild = False
    if agent is None:
        logger.info("Agent 未初始化，需要构建")
        needs_rebuild = True
    elif last_file_key and last_file_key != built_for_key:
        # 文件变化，需重建以启用最新RAG
        logger.info("文件已更新，需要重建 Agent")
        needs_rebuild = True

    if needs_rebuild:
        try:
            logger.info("正在重建 Agent...")
            agent = build_agent(retriever)
            state["agent_executor"] = agent
            state["agent_built_for_file_key"] = last_file_key
            logger.info("Agent 重建成功")
        except Exception as e:
            logger.error(f"Agent初始化失败: {e}", exc_info=True)
            return f"Agent初始化失败: {e}。请尝试重新上传文档。"
    
    try:
        logger.debug("调用 Agent 处理问题...")
        response = agent.invoke({"input": question})
        output = response.get("output", "抱歉，我没有得到有效的回答。")
        logger.info("Agent 回答生成成功")
        return output
    except Exception as e:
        logger.error(f"Agent 处理问题失败: {e}", exc_info=True)
        return f"发生错误: {e}。请检查网络连接或稍后重试。"


def process_more_files(
    files,
    state: Dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    persist_dir_text: str,
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    """
    追加上传多个文件，将内容合并到当前向量索引中。
    
    Args:
        files: 要追加的文件列表
        state: 会话状态
        chunk_size: 文本切分大小
        chunk_overlap: 文本切分重叠
        top_k: 检索条数
        persist_dir_text: 持久化目录
        
    Yields:
        (状态消息, 状态字典) 元组
    """
    if not files:
        logger.warning("未提供要追加的文件")
        yield "请先选择要追加的文件。", state
        return

    try:
        logger.info(f"开始追加 {len(files) if isinstance(files, list) else 1} 个文件到现有索引")
        base_cache_dir = _get_base_cache_dir(persist_dir_text)
        emb_fp = _embeddings_fingerprint(embeddings)

        # 优先使用现有索引目录
        cache_dir = state.get("last_index_cache_dir")
        if not cache_dir or not os.path.isdir(cache_dir):
            # 使用一个稳定的“会话合并”目录
            cache_dir = os.path.join(base_cache_dir, f"combined__{emb_fp[:12]}")

        # 尝试加载已有索引
        vs = _try_load_faiss_from_cache(cache_dir)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        total_chunks = 0
        for f in files:
            try:
                file_path = f.name
            except Exception:
                file_path = str(f)
            file_ext = os.path.splitext(file_path)[1].lower()
            status_html = f"<div style='padding: 10px; border-radius: 5px; background-color: #fff4e6; border-left: 4px solid #FF9800;'>📄 <b>正在处理：</b>{os.path.basename(file_path)}...</div>"
            yield status_html, state

            # 加载文档
            if file_ext == ".pdf":
                documents = _load_pdf_with_fallbacks(file_path)
                if not documents:
                    continue
            elif file_ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
                documents = loader.load()
            elif file_ext == ".docx":
                loader = Docx2txtLoader(file_path)
                documents = loader.load()
            else:
                continue

            # 切分
            chunks = splitter.split_documents(documents)
            if not chunks:
                continue
            total_chunks += len(chunks)

            # 创建或追加到索引
            if vs is None:
                vs = FAISS.from_documents(chunks, embeddings)
            else:
                vs.add_documents(chunks)

        if vs is None:
            error_html = "<div style='padding: 10px; border-radius: 5px; background-color: #ffe6e6; border-left: 4px solid #ff4444;'>❌ <b>追加失败：</b>未能从所选文件中提取可用文本内容<br><small>💡 提示：请检查文件格式或内容是否正确</small></div>"
            yield error_html, state
            return

        # 保存并更新会话
        _save_faiss_to_cache(cache_dir, vs)
        logger.info(f"向量索引已保存到: {cache_dir}")
        
        try:
            status_html = "<div style='padding: 10px; border-radius: 5px; background-color: #fff4e6; border-left: 4px solid #FF9800;'>🤖 <b>最后一步：</b>正在更新智能 Agent...</div>"
            yield status_html, state
            
            state["rag_retriever"] = vs.as_retriever(search_kwargs={"k": int(top_k)})
            state["agent_executor"] = build_agent(state["rag_retriever"])
            state["last_index_cache_dir"] = cache_dir
            state["last_file_key"] = "combined"
            state["agent_built_for_file_key"] = "combined"
            logger.info("文件追加完成，Agent 已更新")
        except Exception as e:
            logger.error(f"构建 Agent 失败: {e}", exc_info=True)
            error_html = f"<div style='padding: 10px; border-radius: 5px; background-color: #ffe6e6; border-left: 4px solid #ff4444;'>⚠️ <b>部分成功：</b>文件追加成功，但 Agent 构建失败<br><small>错误：{str(e)}<br>💡 提示：请检查网络连接或稍后重试</small></div>"
            yield error_html, state
            return

        success_msg = f"✅ <b>追加成功！</b><br>已追加 <b>{total_chunks}</b> 个文本片段到现有索引<br><small>🎉 新文档已生效，可以开始提问了！</small>"
        success_html = f"<div style='padding: 15px; border-radius: 5px; background-color: #e8f5e9; border-left: 4px solid #4CAF50;'>{success_msg}</div>"
        logger.info(f"已追加完成（新增 {total_chunks} 个片段）")
        yield success_html, state

    except Exception as e:
        error_msg = f"❌ <b>追加失败：</b>{str(e)}<br><small>💡 提示：请检查文件格式是否正确，或查看日志获取更多信息</small>"
        error_html = f"<div style='padding: 15px; border-radius: 5px; background-color: #ffe6e6; border-left: 4px solid #ff4444;'>{error_msg}</div>"
        logger.error(f"追加失败: {e}", exc_info=True)
        yield error_html, state


def clear_cache(persist_dir_text: str, state: Dict[str, Any]):
    """
    清理缓存并重置 Agent
    
    Args:
        persist_dir_text: 持久化目录
        state: 会话状态
        
    Returns:
        (状态消息, 状态字典) 元组
    """
    try:
        base_cache_dir = _get_base_cache_dir(persist_dir_text)
        logger.info(f"正在清理缓存目录: {base_cache_dir}")
        _safe_clear_directory(base_cache_dir)
        # 重置会话中的 RAG 状态
        state["rag_retriever"] = None
        state["agent_executor"] = build_agent(None)
        state["last_index_cache_dir"] = None
        state["last_file_key"] = None
        logger.info("缓存已清理，Agent 已重置")
        success_html = "<div style='padding: 15px; border-radius: 5px; background-color: #e8f5e9; border-left: 4px solid #4CAF50;'>✅ <b>清理成功！</b><br>缓存已清理，Agent 已重置为无RAG模式<br><small>💡 提示：如需使用文档检索功能，请重新上传文档</small></div>"
        return success_html, state
    except Exception as e:
        error_msg = f"❌ <b>清理失败：</b>{str(e)}<br><small>💡 提示：请检查目录权限或稍后重试</small>"
        error_html = f"<div style='padding: 15px; border-radius: 5px; background-color: #ffe6e6; border-left: 4px solid #ff4444;'>{error_msg}</div>"
        logger.error(f"清理缓存失败: {e}", exc_info=True)
        return error_html, state


def build_ui():
    with gr.Blocks(theme=gr.themes.Soft(), title="电信行业智能对话系统") as demo:
        session_state = gr.State(_default_session_state())

        gr.Markdown("# 电信行业智能对话系统由LangChain + 通义千问 + Neo4j + RAG驱动")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ▲ 上传业务文档")

                file_uploader = gr.File(
                    label="选择或拖拽文件 (支持多选：TXT/PDF/DOCX)",
                    file_types=[".txt", ".pdf", ".docx"],
                    file_count="multiple",
                )

                with gr.Row():
                    chunk_size = gr.Slider(
                        minimum=200,
                        maximum=2000,
                        value=500,
                        step=50,
                        label="切分片段大小 (chunk_size)",
                    )
                    chunk_overlap = gr.Slider(
                        minimum=0,
                        maximum=400,
                        value=50,
                        step=10,
                        label="切分重叠 (chunk_overlap)",
                    )

                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="检索条数 (top_k)",
                )

                with gr.Accordion("高级设置", open=False):
                    persist_dir = gr.Textbox(
                        label="索引持久化目录 (留空使用 RAG_FAISS_DIR 或 ~/.rag_faiss_cache)",
                        value=_get_default_base_cache_dir(),
                    )
                    clear_cache_btn = gr.Button("清理缓存", variant="secondary")

                process_button = gr.Button("处理上传的文件", variant="primary", size="lg")
                
                # 状态显示区域 - 更醒目
                status_display = gr.Markdown(
                    value="<div style='padding: 15px; border-radius: 8px; background-color: #f5f5f5; border: 2px solid #e0e0e0; text-align: center; font-size: 14px;'>📋 <b>状态：</b>等待上传文件...</div>",
                    label="📊 文件处理状态",
                    visible=True
                )
                
                # 成功/失败提示框 - 独立显示，更加醒目
                result_display = gr.Markdown(
                    value="",
                    visible=True,
                    label="📢 处理结果",
                    elem_classes=["result-display"]
                )

        with gr.Column(scale=2):
            gr.Markdown("### 对话窗口")
            with gr.Row():
                more_files = gr.File(
                    label="追加上传文档 (支持多选)", file_types=[".txt", ".pdf", ".docx"], file_count="multiple"
                )
                more_files_btn = gr.Button("上传更多文档（追加到当前索引）", variant="secondary")
            more_status = gr.Markdown()

        gr.ChatInterface(
            fn=chat_with_agent,
            chatbot=gr.Chatbot(height=500, type="messages"),
            textbox=gr.Textbox(
                placeholder="输入您的问题，例如：'王伟的套餐是什么？'", container=False, scale=7
            ),
            title=None,
            submit_btn="发送",
            additional_inputs=[session_state],
        )

        process_button.click(
            fn=process_uploaded_file,
            inputs=[file_uploader, session_state, chunk_size, chunk_overlap, top_k, persist_dir],
            outputs=[status_display, result_display, process_button, session_state],
        )

        clear_cache_btn.click(
            fn=clear_cache,
            inputs=[persist_dir, session_state],
            outputs=[status_display, session_state],
        )

        more_files_btn.click(
            fn=process_more_files,
            inputs=[more_files, session_state, chunk_size, chunk_overlap, top_k, persist_dir],
            outputs=[more_status, session_state],
        )

        demo.queue()

    return demo