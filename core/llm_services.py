# core/llm_services.py

import os
import logging
from typing import List, Optional
from dotenv import load_dotenv
from langchain_community.llms import Tongyi
import dashscope
from dashscope import TextEmbedding
from tenacity import retry, stop_after_attempt, wait_exponential

# 配置日志
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not DASHSCOPE_API_KEY:
    raise ValueError("DASHSCOPE_API_KEY 未在 .env 文件中设置")

# 设置 DashScope API Key
dashscope.api_key = DASHSCOPE_API_KEY

# DashScope Embedding API 批量限制
BATCH_SIZE = 25


class TongyiEmbeddings:
    """
    自定义的 TongyiEmbeddings 实现，支持批量处理和重试机制。
    兼容 LangChain 的 Embeddings 接口，并实现 __call__ 方法以支持 FAISS 查询。
    """

    def __init__(self, model: str = "text-embedding-v1"):
        """
        初始化 TongyiEmbeddings
        
        Args:
            model: DashScope 嵌入模型名称，默认为 "text-embedding-v1"
        """
        self.model = model
        logger.info(f"初始化 TongyiEmbeddings，模型: {model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成嵌入向量（内部方法，带重试机制）
        
        Args:
            texts: 文本列表，单批最多 25 条
            
        Returns:
            嵌入向量列表
            
        Raises:
            Exception: API 调用失败时抛出异常
        """
        if not texts:
            return []

        # 根据 README，使用 input=[text, ...] 格式
        response = TextEmbedding.call(
            model=self.model,
            input=texts
        )

        if response.status_code == 200:
            embeddings_raw = response.output.get("embeddings", [])
            logger.debug(f"成功获取 {len(embeddings_raw)} 个嵌入向量")
            
            # 调试：记录第一个元素的类型和结构（仅在前几次调用时）
            if embeddings_raw and logger.isEnabledFor(logging.DEBUG):
                first_item = embeddings_raw[0]
                logger.debug(f"第一个 embedding 项的类型: {type(first_item)}, 结构: {str(first_item)[:200]}")
            
            # DashScope API 返回的格式可能是字典列表，每个字典包含 'embedding' 字段
            # 需要提取实际的向量数组
            embeddings = []
            for idx, item in enumerate(embeddings_raw):
                if isinstance(item, dict):
                    # 如果是字典，提取 'embedding' 字段
                    embedding = item.get("embedding") or item.get("vector") or item.get("text_embedding")
                    if embedding:
                        embeddings.append(embedding)
                    else:
                        # 如果字典中没有常见的 embedding 字段，尝试其他方式
                        logger.warning(f"第 {idx} 个 embedding 未找到标准字段，字典键: {list(item.keys())}")
                        # 尝试获取字典中第一个列表类型的值
                        for value in item.values():
                            if isinstance(value, list) and len(value) > 0:
                                if isinstance(value[0], (int, float)):
                                    embeddings.append(value)
                                    break
                        else:
                            logger.error(f"无法从字典中提取 embedding: {item}")
                elif isinstance(item, list):
                    # 如果已经是列表，直接使用
                    if item and isinstance(item[0], (int, float)):
                        embeddings.append(item)
                    else:
                        logger.warning(f"第 {idx} 个 embedding 列表格式异常: {type(item[0]) if item else 'empty'}")
                else:
                    logger.error(f"第 {idx} 个 embedding 未知格式: {type(item)}")
            
            if not embeddings:
                error_msg = f"未能从 API 响应中提取有效的嵌入向量。原始数据示例: {str(embeddings_raw[:1]) if embeddings_raw else 'empty'}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            if len(embeddings) != len(texts):
                logger.warning(f"提取的嵌入向量数量 ({len(embeddings)}) 与输入文本数量 ({len(texts)}) 不匹配")
            
            logger.debug(f"提取了 {len(embeddings)} 个有效的嵌入向量")
            return embeddings
        else:
            error_msg = f"Embedding API error: {response.code} - {response.message}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def embed_query(self, text: str) -> List[float]:
        """
        生成单个文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        if not text or not text.strip():
            logger.warning("收到空文本，返回空向量")
            return []
        
        results = self._embed_batch([text])
        return results[0] if results else []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成多个文本的嵌入向量，自动分批处理（单批最多 25 条）
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        if not texts:
            return []

        all_embeddings = []
        total = len(texts)
        
        # 分批处理，每批最多 BATCH_SIZE 条
        for i in range(0, total, BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
            
            logger.info(f"处理第 {batch_num}/{total_batches} 批，包含 {len(batch)} 条文本")
            
            try:
                batch_embeddings = self._embed_batch(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"批量嵌入失败（第 {batch_num} 批）: {e}")
                # 如果批量失败，尝试逐条处理作为降级策略
                logger.warning(f"尝试逐条处理第 {batch_num} 批...")
                for text in batch:
                    try:
                        embedding = self.embed_query(text)
                        all_embeddings.append(embedding)
                    except Exception as single_error:
                        logger.error(f"单条嵌入失败: {single_error}")
                        # 添加空向量以保持索引一致性
                        all_embeddings.append([])

        logger.info(f"完成 {total} 个文本的嵌入向量生成")
        return all_embeddings

    def __call__(self, text: str) -> List[float]:
        """
        使对象可调用，兼容 FAISS 在查询阶段对 embedding 的可调用要求
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        return self.embed_query(text)


# 初始化并提供全局的 LLM 和 Embedding 模型实例
llm = Tongyi(model_name="qwen-plus", temperature=0)
embeddings = TongyiEmbeddings(model="text-embedding-v1")