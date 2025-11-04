# core/agent_builder.py

from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents.initialize import initialize_agent
from langchain_classic.memory.buffer_window import ConversationBufferWindowMemory

# 从同级目录导入
from .llm_services import llm
from .tool_builder import get_graph_tool, get_general_chat_tool, get_rag_tool


def build_agent(rag_retriever=None):
    """
    根据是否存在 RAG retriever 来构建 Agent。
    :param rag_retriever: 从FAISS创建的 retriever 对象，可以为 None。
    :return: 一个 Agent Executor 实例。
    """
    tools = [get_graph_tool(), get_general_chat_tool()]

    if rag_retriever:
        rag_tool = get_rag_tool(rag_retriever)
        if rag_tool:
            tools.append(rag_tool)
            print("RAG工具已成功创建并添加到Agent中。")
    else:
        print("未提供RAG retriever，Agent仅包含图谱查询和通用对话工具。")

    memory = ConversationBufferWindowMemory(
        k=3, memory_key="chat_history", return_messages=True
    )

    agent_executor = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        max_iterations=5,
    )
    print("Agent已初始化/更新。")
    return agent_executor
