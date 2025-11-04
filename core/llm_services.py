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

    def embed_query(self, text):
        """生成单个文本的嵌入向量"""
        response = TextEmbedding.call(
            model=self.model,
            input={"text": text}
        )
        if response.status_code == 200:
            return response.output["embeddings"][0]
        else:
            raise Exception(f"Embedding API error: {response.code} - {response.message}")

    def embed_documents(self, texts):
        """生成多个文本的嵌入向量"""
        results = []
        for text in texts:
            results.append(self.embed_query(text))
        return results


# 初始化并提供全局的 LLM 和 Embedding 模型实例
llm = Tongyi(model_name="qwen-plus", temperature=0)
embeddings = TongyiEmbeddings(model="text-embedding-v1")