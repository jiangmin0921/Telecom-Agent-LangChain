1、项目结构
TelecomAgentProject/
├── .env
├── .gitignore
├── requirements.txt
|
├── data_utils/               # 存放数据处理相关的脚本
│   ├── __init__.py
│   ├── generate_data.py
│   └── import_to_neo4j.py
|
├── core/                     # 存放智能体的核心逻辑
│   ├── __init__.py
│   ├── agent_builder.py      # 负责构建和初始化Agent
│   ├── llm_services.py       # 统一管理LLM和Embedding模型
│   └── tool_builder.py       # 负责构建所有工具(Tool)
|
├── ui/                       # 存放用户界面相关代码
│   ├── __init__.py
│   └── gradio_app.py         # Gradio界面和交互逻辑
|
└── app.py                    # 项目的唯一入口点，负责启动应用
