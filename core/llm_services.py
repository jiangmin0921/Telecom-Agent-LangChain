# core/llm_services.py

import os
from dotenv import load_dotenv
from langchain_community.llms import Tongyi
import dashscope
from dashscope import TextEmbedding

# 加载环境变量
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not DASHSCOPE_API_KEY:
    raise ValueError("DASHSCOPE_API_KEY 未在 .env 文件中设置")

# 设置 DashScope API Key
dashscope.api_key = DASHSCOPE_API_KEY


# 自定义的 TongyiEmbeddings 替代实现
class TongyiEmbeddings:
    def __init__(self, model="text-embedding-v1"):
        self.model = model

    def __call__(self, text):
        """兼容 LangChain 在查询阶段将 embedding 当作可调用对象的用法"""
        return self.embed_query(text)

    def embed_query(self, text):
        """生成单个文本的嵌入向量"""
        # 确保输入是字符串
        if not isinstance(text, str):
            if hasattr(text, 'page_content'):  # 处理 Document 对象
                text = str(text.page_content)
            else:
                text = str(text)
        
        response = TextEmbedding.call(
            model=self.model,
            input=[text]
        )
        # DashScope 返回结构参考：
        # { "output": { "embeddings": [ { "embedding": [...], "text_index": 0 } ] } }
        if getattr(response, "status_code", None) == 200 and response.output and "embeddings" in response.output:
            return response.output["embeddings"][0]["embedding"]
        # 兜底错误信息
        code = getattr(response, "code", getattr(response, "status_code", "Unknown"))
        message = getattr(response, "message", "Unknown error")
        raise Exception(f"Embedding API error: {code} - {message}")

    def embed_documents(self, texts):
        """生成多个文本的嵌入向量"""
        if not isinstance(texts, list):
            raise TypeError("embed_documents expects a list of strings")
        if len(texts) == 0:
            return []

        # 确保所有元素都是字符串，如果不是则转换为字符串
        text_list = []
        for text in texts:
            if isinstance(text, str):
                text_list.append(text)
            elif hasattr(text, 'page_content'):  # 处理 Document 对象
                text_list.append(str(text.page_content))
            else:
                text_list.append(str(text))

        # DashScope API 限制：每批最多 25 个文本
        batch_size = 25
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i + batch_size]
            
            response = TextEmbedding.call(
                model=self.model,
                input=batch
            )
            
            if getattr(response, "status_code", None) == 200 and response.output and "embeddings" in response.output:
                # 保持与输入顺序一致，根据 text_index 排序
                embeddings_items = response.output["embeddings"]
                embeddings_items.sort(key=lambda item: item.get("text_index", 0))
                batch_embeddings = [item["embedding"] for item in embeddings_items]
                all_embeddings.extend(batch_embeddings)
            else:
                code = getattr(response, "code", getattr(response, "status_code", "Unknown"))
                message = getattr(response, "message", "Unknown error")
                raise Exception(f"Embedding API error: {code} - {message}")
        
        return all_embeddings


# 初始化并提供全局的 LLM 和 Embedding 模型实例
llm = Tongyi(model_name="qwen-plus", temperature=0)
embeddings = TongyiEmbeddings(model="text-embedding-v1")