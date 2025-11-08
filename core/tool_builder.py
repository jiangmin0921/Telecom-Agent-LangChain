# core/tool_builder.py

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from langchain_core.tools import Tool
from langchain_core.retrievers import BaseRetriever
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from .llm_services import llm

# 配置日志
logger = logging.getLogger(__name__)

# 加载数据库环境变量
load_dotenv()


def get_graph_tool() -> Tool:
    """
    构建知识图谱查询工具，并在启动时检测 Neo4j 是否可连接。
    如果连接失败，直接抛错提示，不影响其他工具的加载。
    
    Returns:
        Tool: 知识图谱查询工具
        
    Raises:
        ConnectionError: Neo4j 连接信息不完整或连接失败时抛出
    """
    # 读取数据库环境变量（在函数内部读取，避免导入时失败）
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
        error_msg = "Neo4j 数据库连接信息未在 .env 文件中完全设置"
        logger.error(error_msg)
        raise ConnectionError(error_msg)

    try:
        logger.info(f"正在连接 Neo4j 数据库: {NEO4J_URI}")
        # 初始化 Neo4jGraph 对象
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )

        # 自检连接（如果失败会抛出异常）
        if hasattr(graph, "_driver") and graph._driver:
            graph._driver.verify_connectivity()
            logger.info("Neo4j 数据库连接验证成功")
        else:
            raise ConnectionError("无法初始化 Neo4j 驱动对象")

    except Exception as e:
        error_msg = (
            f"Neo4j 数据库连接失败，请检查 URI/用户名/密码 是否正确，以及数据库是否已启动。\n"
            f"详细错误: {e}"
        )
        logger.error(error_msg)
        raise ConnectionError(error_msg)

    # 刷新图数据库 Schema
    try:
        graph.refresh_schema()
        logger.info("Neo4j 图数据库 Schema 已刷新")
    except Exception as e:
        logger.warning(f"刷新 Neo4j Schema 失败: {e}，继续使用现有 Schema")

    # 构建 Cypher QA 链
    try:
        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True,
        )
        logger.info("Cypher QA 链构建成功")
    except Exception as e:
        logger.error(f"Cypher QA 链构建失败: {e}")
        raise

    # 返回 Tool 对象
    return Tool(
        name="GraphDBQuery",
        func=chain.run,
        description=(
            "非常适用于回答关于电信客户、套餐、使用情况等具体结构化数据的问题。"
            "输入应该是一个完整的问题，例如'张三办理了什么套餐？'或'哪个客户的10月份流量使用最多？'。"
        )
    )


def get_general_chat_tool() -> Tool:
    """
    构建通用聊天工具
    
    Returns:
        Tool: 通用聊天工具
    """
    logger.info("创建通用聊天工具")
    return Tool(
        name="GeneralChat",
        func=llm.invoke,
        description=(
            "适用于处理闲聊、问候或不涉及具体客户数据和文档内容的通用性问题，例如'你好'或'你能做什么？'。"
        )
    )


def get_rag_tool(retriever: Optional[BaseRetriever]) -> Optional[Tool]:
    """
    根据传入的 retriever 构建文档问答(RAG)工具
    
    Args:
        retriever: 向量检索器对象，如果为 None 则返回 None
        
    Returns:
        Tool: 文档问答工具，如果 retriever 为 None 则返回 None
    """
    if not retriever:
        logger.warning("未提供 retriever，无法创建 RAG 工具")
        return None

    try:
        logger.info("正在创建 RAG 问答链...")
        rag_qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            verbose=True,
        )
        logger.info("RAG 问答链创建成功")
    except Exception as e:
        logger.error(f"RAG 问答链创建失败: {e}")
        raise

    return Tool(
        name="DocumentQuery",
        func=rag_qa_chain.run,
        description=(
            "当你需要根据上传的文档内容回答问题时使用此工具。"
            "它最适合回答关于文档中包含的特定信息、概念、定义或流程的问题。"
            "例如，如果文档是关于5G技术的白皮书，你可以问'文档中提到的5G核心技术有哪些？'。"
        )
    )
