Telecom-Agent-LangChain（电信行业智能对话系统）

简介
- 基于 LangChain + 通义千问（DashScope）+ FAISS（RAG）+ 可选 Neo4j 的行业智能问答与检索增强系统。
- 支持上传 TXT/PDF/DOCX，自动切分文本并构建向量索引，随后在对话中进行知识检索与回答。
- 支持批量上传与增量追加，无需重启即可让新文档即时生效。

核心特性
- RAG 检索增强：基于 FAISS 的向量检索，支持自定义检索条数（Top-K）。
- 文档处理：支持 TXT / PDF / DOCX，含 PDF 多策略加载与 OCR 兜底（如安装了 OCRmyPDF）。
- 批量上传与追加：
  - 左侧上传区支持多选文件，一次性构建合并索引。
  - 右侧“上传更多文档（追加到当前索引）”支持多选，增量追加到当前索引，不打断对话。
- 缓存与持久化：向量索引可持久化到磁盘，重复加载时自动命中缓存，避免重复向量化。
- 兼容 DashScope 限制：自动批处理（单批 ≤ 25 条）并按需重试。

架构概览
- UI 层（Gradio）：`ui/gradio_app.py`
  - 文件上传、参数设置（chunk_size / chunk_overlap / top_k / 持久化目录）、会话对话。
- 核心层（Core）：`core/`
  - `llm_services.py`：封装通义千问 LLM 与 DashScope Embedding（含批处理与异常处理）。
  - `agent_builder.py`：根据检索器构建 Agent / Chain。
  - `tool_builder.py`（如使用）：统一注册与构建工具。
- 向量索引（FAISS）：基于 `langchain_community.vectorstores.FAISS` 构建/加载/保存。
- 数据层（可选）：`data_utils/` 提供数据准备脚本与 Neo4j 导入示例。

项目结构
Telecom-Agent-LangChain/
├── .env
├── .gitignore
├── requirements.txt
├── app.py                      # 启动入口
├── core/
│   ├── __init__.py
│   ├── agent_builder.py        # 构建对话 Agent / Chain
│   ├── llm_services.py         # LLM 与 Embedding 封装（DashScope）
│   └── tool_builder.py         # 工具注册（如使用）
├── ui/
│   ├── __init__.py
│   └── gradio_app.py           # Gradio UI 与交互逻辑
└── data_utils/
    ├── __init__.py
    ├── generate_data.py
    └── import_to_neo4j.py

环境与依赖
- 操作系统：Windows / macOS / Linux
- Python：3.10（建议使用 Conda 虚拟环境）
- 依赖安装：
  - 推荐：`pip install -r requirements.txt`
  - 如果使用 OCR 兜底，建议安装 OCRmyPDF：`pip install ocrmypdf`（需要系统依赖，Windows 可选择 WSL 或跳过）

环境变量配置（.env）
- 必填
  - DASHSCOPE_API_KEY=你的DashScope密钥
- 可选
  - RAG_FAISS_DIR=向量索引持久化目录（留空则默认 `~/.rag_faiss_cache`）
  - 其他与 Neo4j 相关连接串按需在对应文件中配置

快速开始
1. 克隆并进入项目目录
2. 创建并激活环境（示例使用 Conda）
   - conda create -n telecom_agent python=3.10 -y
   - conda activate telecom_agent
3. 安装依赖
   - pip install -r requirements.txt
4. 配置 .env
   - 新建 `.env`，填入 `DASHSCOPE_API_KEY=...`
5. 启动应用
   - python app.py
6. 打开浏览器访问 Gradio 提示的地址（一般是 `http://127.0.0.1:7860`）。

使用说明（UI）
- 上传业务文档
  - 左侧“上传业务文档”支持多选 TXT/PDF/DOCX；选择后点击“处理上传的文件”。
  - 系统会加载→切分（chunk_size / chunk_overlap 可调）→向量化→构建/加载索引。
  - 大文件会提示处理较慢；PDF 遇到非文本将尝试多种加载策略并可选 OCR 兜底（若安装）。
- 对话
  - 右侧对话框中输入问题，例如“RAG 的工作原理是什么”。
  - 回答将基于你已上传的文档进行检索增强。
- 追加上传
  - 右侧“上传更多文档（追加到当前索引）”支持多选，将增量追加到现有索引，立即生效。
- 缓存
  - 可设置“索引持久化目录”；不同 Embedding 指纹会使用不同缓存目录，避免错配。
  - 可点击“清理缓存”清空持久化向量索引并重置到无 RAG 模式。

关键实现与约束
- Embedding（DashScope TextEmbedding）：
  - 已修复 API 入参格式，使用 `input=[text, ...]`。
  - 批量限制：单批最多 25 条，系统自动分批以避免报错。
  - `TongyiEmbeddings` 实现了 `__call__`，以兼容 FAISS 在查询阶段对 embedding 的可调用要求。
- FAISS 索引
  - 单文件：以文件特定 key 持久化。
  - 多文件：使用会话级目录 `combined__{emb_fp}` 合并存储。

常见问题（FAQ）
- 报错：`Embedding API error: InvalidParameter - input.texts should be array`
  - 原因：DashScope 入参格式错误。当前实现已修复为 `input=[...]`。
- 报错：`batch size is invalid, it should not be larger than 25`
  - 原因：单次批量超过 25。当前实现已自动分批处理。
- 报错：`'TongyiEmbeddings' object is not callable`
  - 原因：FAISS 查询调用 embedding 需可调用对象。当前实现已添加 `__call__` 映射到 `embed_query`。
- PDF 未能提取文本
  - 可能是扫描版或受保护 PDF。建议先 OCR 处理或者安装 OCRmyPDF 后重试。

开发与扩展建议
- 可在 `core/tool_builder.py` 增加行业工具，如工单检索、套餐规则查询等。
- `core/agent_builder.py` 中替换/扩展 Agent 策略、提示词与工具调度逻辑。
- 如需替换向量数据库（如 Milvus、PGVector），替换 `FAISS` 相关调用即可。

命令速查
- 启动：
  - python app.py
- 安装依赖：
  - pip install -r requirements.txt
- 清理缓存（UI 内按钮）或手动删除 `RAG_FAISS_DIR` 目录。

许可证
- 根据你所在企业/团队策略选择（当前未明确声明）。
