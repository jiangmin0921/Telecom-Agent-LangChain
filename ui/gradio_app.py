import hashlib
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
) -> Generator[Tuple[str, Any, Dict[str, Any]], None, None]:
    if not file_obj:
        yield "请先上传文件（支持多选）。", gr.update(interactive=True), state
        return

    try:
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
        if vs:
            yield "1/3: 已从缓存加载向量索引。", gr.update(interactive=False), state
        else:
            yield "1/3: 正在加载文档...", gr.update(interactive=False), state

            for f in files:
                try:
                    file_path = f.name
                except Exception:
                    file_path = str(f)
                file_ext = os.path.splitext(file_path)[1].lower()

                # 文件大小提示（尽力而为）
                try:
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    if size_mb > 50:
                        yield f"提示：{os.path.basename(file_path)} 文件较大（约{size_mb:.1f}MB），处理可能较慢。", gr.update(interactive=False), state
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
                    yield f"跳过不支持的文件格式: {file_ext}", gr.update(interactive=False), state
                    continue

                # 切分
                yield f"2/3: 正在切分 {os.path.basename(file_path)}...", gr.update(interactive=False), state
                chunks = splitter.split_documents(documents)
                if not chunks:
                    continue
                total_chunks += len(chunks)

                # 建索引/追加
                if vs is None:
                    yield "3/3: 正在创建向量索引...", gr.update(interactive=False), state
                    vs = FAISS.from_documents(chunks, embeddings)
                else:
                    yield f"3/3: 追加到向量索引（{os.path.basename(file_path)}）...", gr.update(interactive=False), state
                    vs.add_documents(chunks)

            if vs is None:
                yield (
                    "未能从所选文件中提取任何文本内容。若为扫描版PDF，请先进行OCR处理（例如使用OCRmyPDF）后再尝试。",
                    gr.update(interactive=True),
                    state,
                )
                return

            _save_faiss_to_cache(cache_dir, vs)

        state["rag_retriever"] = vs.as_retriever(search_kwargs={"k": int(top_k)})
        state["agent_executor"] = build_agent(state["rag_retriever"])
        state["last_index_cache_dir"] = cache_dir
        state["last_file_key"] = file_key
        state["agent_built_for_file_key"] = file_key

        if len(files) > 1:
            yield f"已处理并合并 {len(files)} 个文件（新增 {total_chunks} 个片段）。", gr.update(interactive=True), state
        else:
            yield f"文件 '{os.path.basename(single_path)}' 处理成功！", gr.update(interactive=True), state

    except Exception as e:
        traceback.print_exc()
        yield f"处理失败: {e}", gr.update(interactive=True), state


def chat_with_agent(question: str, history: list, state: Dict[str, Any]) -> str:
    agent = state.get("agent_executor")
    retriever = state.get("rag_retriever")
    last_file_key = state.get("last_file_key")
    built_for_key = state.get("agent_built_for_file_key")

    needs_rebuild = False
    if agent is None:
        needs_rebuild = True
    elif last_file_key and last_file_key != built_for_key:
        # 文件变化，需重建以启用最新RAG
        needs_rebuild = True

    if needs_rebuild:
        try:
            agent = build_agent(retriever)
            state["agent_executor"] = agent
            state["agent_built_for_file_key"] = last_file_key
        except Exception as e:
            traceback.print_exc()
            return f"Agent初始化失败: {e}"
    try:
        response = agent.invoke({"input": question})
        return response.get("output", "抱歉，我没有得到有效的回答。")
    except Exception as e:
        traceback.print_exc()
        return f"发生错误: {e}"


def process_more_files(
    files,
    state: Dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    persist_dir_text: str,
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    """追加上传多个文件，将内容合并到当前向量索引中。"""
    if not files:
        yield "请先选择要追加的文件。", state
        return

    try:
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
            yield f"正在处理: {os.path.basename(file_path)}...", state

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
            yield "未能从所选文件中提取可用文本内容。", state
            return

        # 保存并更新会话
        _save_faiss_to_cache(cache_dir, vs)
        state["rag_retriever"] = vs.as_retriever(search_kwargs={"k": int(top_k)})
        state["agent_executor"] = build_agent(state["rag_retriever"])
        state["last_index_cache_dir"] = cache_dir
        state["last_file_key"] = "combined"
        state["agent_built_for_file_key"] = "combined"

        yield f"已追加完成（新增 {total_chunks} 个片段）。", state

    except Exception as e:
        traceback.print_exc()
        yield f"追加失败: {e}", state


def clear_cache(persist_dir_text: str, state: Dict[str, Any]):
    try:
        base_cache_dir = _get_base_cache_dir(persist_dir_text)
        _safe_clear_directory(base_cache_dir)
        # 重置会话中的 RAG 状态
        state["rag_retriever"] = None
        state["agent_executor"] = build_agent(None)
        state["last_index_cache_dir"] = None
        state["last_file_key"] = None
        return "缓存已清理，Agent 已重置为无RAG模式。", state
    except Exception as e:
        traceback.print_exc()
        return f"清理缓存失败: {e}", state


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

                process_button = gr.Button("处理上传的文件", variant="primary")
                status_display = gr.Textbox(
                    label="文件处理状态", interactive=False, value="等待上传文件..."
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
            outputs=[status_display, process_button, session_state],
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