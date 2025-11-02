# core/llm_services.py

import os
from dotenv import load_dotenv
from langchain_community.llms import Tongyi
from langchain_community.embeddings import TongyiEmbeddings

# 加载环境变量
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not DASHSCOPE_API_KEY:
    raise ValueError("DASHSCOPE_API_KEY 未在 .env 文件中设置")

# 初始化并提供全局的 LLM 和 Embedding 模型实例
# 这样整个应用中所有模块都可以从这里导入，确保使用同一个模型实例
llm = Tongyi(model_name="qwen-plus", temperature=0)
embeddings = TongyiEmbeddings(model="text-embedding-v1")
