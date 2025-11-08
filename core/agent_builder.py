# core/agent_builder.py

import logging
from typing import Optional, Any
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents.initialize import initialize_agent
from langchain_classic.memory.buffer_window import ConversationBufferWindowMemory
from langchain_core.retrievers import BaseRetriever

# 从同级目录导入
from .llm_services import llm
from .tool_builder import get_graph_tool, get_general_chat_tool, get_rag_tool

# 配置日志
logger = logging.getLogger(__name__)


def build_agent(rag_retriever: Optional[BaseRetriever] = None) -> Any:
    """
    根据是否存在 RAG retriever 来构建 Agent。
    
    Args:
        rag_retriever: 从FAISS创建的 retriever 对象，可以为 None。
                      如果提供，将启用文档检索增强功能。
    
    Returns:
        一个 Agent Executor 实例，包含配置好的工具和记忆。
        
    Raises:
        Exception: Agent 初始化失败时抛出异常
    """
    tools = []
    
    # 图谱工具是可选的，失败时不阻断整体Agent
    try:
        graph_tool = get_graph_tool()
        tools.append(graph_tool)
        logger.info("知识图谱工具已成功加载")
    except Exception as e:
        logger.warning(f"图谱工具加载失败，已跳过。原因: {e}")

    # 通用聊天工具（必需）
    try:
        general_chat_tool = get_general_chat_tool()
        tools.append(general_chat_tool)
        logger.info("通用聊天工具已成功加载")
    except Exception as e:
        logger.error(f"通用聊天工具加载失败: {e}")
        raise

    # RAG 工具（可选，取决于是否提供 retriever）
    if rag_retriever:
        try:
            rag_tool = get_rag_tool(rag_retriever)
            if rag_tool:
                tools.append(rag_tool)
                logger.info("RAG工具已成功创建并添加到Agent中")
            else:
                logger.warning("RAG工具创建失败，返回 None")
        except Exception as e:
            logger.error(f"RAG工具创建失败: {e}")
            # RAG 工具失败不影响 Agent 构建，继续使用其他工具
    else:
        logger.info("未提供RAG retriever，Agent仅包含图谱查询和通用对话工具")

    if not tools:
        raise ValueError("无法构建Agent：没有可用的工具")

    # 配置对话记忆（保留最近3轮对话）
    memory = ConversationBufferWindowMemory(
        k=3, 
        memory_key="chat_history", 
        return_messages=True
    )

    try:
        agent_executor = initialize_agent(
            tools,
            llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True,
            max_iterations=5,
        )
        logger.info(f"Agent已成功初始化/更新，包含 {len(tools)} 个工具")
        return agent_executor
    except Exception as e:
        logger.error(f"Agent初始化失败: {e}")
        raise
