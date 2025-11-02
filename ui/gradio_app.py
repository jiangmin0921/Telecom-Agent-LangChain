# ui/gradio_app.py

import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 从上级目录导入
from core.agent_builder import build_agent
from core.llm_services import embeddings

# 使用一个字典来管理应用的状态，主要是agent和retriever
app_state = {
    "rag_retriever": None,
    "agent_executor": None,
}

# 初始化一个没有RAG能力的agent
app_state["agent_executor"] = build_agent(None)


def process_uploaded_file(file_obj):
    """处理上传的文件，创建RAG检索器并更新Agent"""
    if not file_obj:
        yield "请先上传一个文件。", gr.update(interactive=False)

    try:
        file_path = file_obj.name
        file_ext = os.path.splitext(file_path)[1].lower()
        yield "开始处理文件...", gr.update(interactive=False)

        if file_ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            yield f"不支持的文件格式: {file_ext}", gr.update(interactive=True)
            return

        yield "1/4: 正在加载文档...", gr.update(interactive=False)
        documents = loader.load()

        yield "2/4: 正在切分文本...", gr.update(interactive=False)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        if not chunks:
            yield "未能从文档中提取任何文本内容。", gr.update(interactive=True)
            return

        yield "3/4: 正在创建向量索引...", gr.update(interactive=False)
        vector_store = FAISS.from_documents(chunks, embeddings)
        app_state["rag_retriever"] = vector_store.as_retriever(search_kwargs={"k": 3})

        yield "4/4: 正在更新智能体...", gr.update(interactive=False)
        app_state["agent_executor"] = build_agent(app_state["rag_retriever"])

        yield f"文件 '{os.path.basename(file_path)}' 处理成功！", gr.update(interactive=True)

    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        yield f"处理失败: {e}", gr.update(interactive=True)


def chat_with_agent(question, history):
    """与Agent进行对话的主函数"""
    agent = app_state.get("agent_executor")
    if not agent:
        return "Agent尚未初始化。请重启应用。"

    try:
        response = agent.invoke({"input": question})
        return response.get('output', '抱歉，我没有得到有效的回答。')
    except Exception as e:
        print(f"与Agent对话时发生错误: {e}")
        return f"发生错误: {e}"


def build_ui():
    """构建Gradio界面"""
    with gr.Blocks(theme=gr.themes.Soft(), title="电信行业智能对话系统") as demo:
        gr.Markdown("# 电信行业智能对话系统由LangChain + 通义千问 + Neo4j + RAG驱动")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ▲ 上传业务文档")
                file_uploader = gr.File(label="选择或拖拽文件 (支持TXT/PDF)", file_types=['.txt', '.pdf'])
                process_button = gr.Button("处理上传的文件", variant="primary")
                status_display = gr.Textbox(label="文件处理状态", interactive=False, value="等待上传文件...")


        with gr.Column(scale=2):
            gr.Markdown("### 对话窗口")
        gr.ChatInterface(
            fn=chat_with_agent,
            chatbot=gr.Chatbot(height=500),
            textbox=gr.Textbox(placeholder="输入您的问题，例如：'王伟的套餐是什么？'", container=False, scale=7),
            title=None,
            submit_btn="发送",
            retry_btn=None,
            undo_btn=None,
            clear_btn="清除对话历史",
        )

        process_button.click(
            fn=process_uploaded_file,
            inputs=[file_uploader],
            outputs=[status_display, process_button]
        )
    return demo
