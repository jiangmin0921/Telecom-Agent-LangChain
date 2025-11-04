# core/tool_builder.py

import os

from dotenv import load_dotenv
from langchain_core.tools.simple import Tool
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_neo4j import Neo4jGraph

# 从同级目录导入模型
from .llm_services import llm

# 加载数据库环境变量
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    raise ValueError("Neo4j 数据库连接信息未在 .env 文件中完全设置")


# --- 工具构建函数 ---

def get_graph_tool():
    """构建知识图谱查询工具"""
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    graph.refresh_schema()

    chain = GraphCypherQAChain.from_llm(
        cypher_llm=llm, qa_llm=llm, graph=graph, verbose=True, validate_cypher=True
    )
    return Tool(
        name="GraphDBQuery",
        func=chain.run,
        description="""
        非常适用于回答关于电信客户、套餐、使用情况等具体结构化数据的问题。
        输入应该是一个完整的问题，例如'张三办理了什么套餐？'或'哪个客户的10月份流量使用最多？'。
        """,
    )


def get_general_chat_tool():
    """构建通用对话工具"""
    return Tool(
        name="GeneralChat",
        func=llm.invoke,
        description="适用于处理闲聊、问候或不涉及具体客户数据和文档内容的通用性问题，例如'你好'或'你能做什么？'",
    )


def get_rag_tool(retriever):
    """根据传入的 retriever 构建文档问答(RAG)工具"""
    if not retriever:
        return None

    rag_qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True,
    )
    return Tool(
        name="DocumentQuery",
        func=rag_qa_chain.run,
        description="""
        当你需要根据上传的文档内容回答问题时使用此工具。
        它最适合回答关于文档中包含的特定信息、概念、定义或流程的问题。
        例如，如果文档是关于5G技术的白皮书，你可以问'文档中提到的5G核心技术有哪些？'。
        """,
    )

