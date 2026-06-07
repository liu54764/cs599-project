# CS599 Project - AI 学术助手

## 项目简介

基于 Retrieval-Augmented Generation (RAG) 的智能论文问答系统，帮助研究者快速理解学术文献、提取关键信息，并基于知识库进行智能问答。

## 方向

方向一：Agentic AI 原生开发

## 技术栈

- **AI IDE**: Trae CN
- **LLM**: DeepSeek API
- **框架**: LangGraph, LangChain
- **嵌入模型**: sentence-transformers/all-MiniLM-L6-v2
- **向量数据库**: ChromaDB
- **后端框架**: FastAPI
- **前端**: HTML + Tailwind CSS + Font Awesome
- **文档处理**: PyPDF2, pdfplumber

## 目录结构

```
cs599-project/
├── docs/                    # 项目文档
│   └── architecture.md       # 详细架构说明
├── src/                      # 项目源代码
│   ├── document_processor/    # 文档处理模块
│   │   ├── pdf_extractor.py      # PDF文档解析
│   │   ├── text_cleaner.py       # 文本清洗与预处理
│   │   ├── document_manager.py   # 文档管理功能
│   │   └── models.py             # 数据模型定义
│   ├── knowledge_base/        # 知识库模块
│   │   ├── embedding_service.py  # 文本向量化服务
│   │   ├── vector_store.py       # 向量数据库存储
│   │   ├── knowledge_manager.py  # 知识库管理
│   │   └── enhanced_retriever.py # 增强检索器
│   ├── qa_module/             # 问答模块
│   │   ├── qa_engine.py          # 核心问答引擎
│   │   ├── llm_client.py         # LLM客户端
│   │   ├── prompt_templates.py   # 提示词模板
│   │   └── langchain_qa_engine.py # LangChain问答集成
│   ├── frontend/              # 前端界面
│   │   └── index.html            # 主界面文件
│   ├── models/                # 模型文件
│   │   └── embeddings/           # 嵌入模型缓存
│   └── main.py                # 服务主入口
├── data/                     # 数据目录
│   ├── documents/             # 上传的文档存储
│   ├── uploads/               # 上传临时目录
│   └── vector_db/             # 向量数据库
├── .gitignore                # Git忽略配置
├── LICENSE                   # MIT许可证
├── README.md                 # 项目说明文档
└── requirements.txt          # Python依赖列表
```

## 环境搭建

### 1. 依赖安装

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 环境变量配置

在项目根目录创建 `.env` 文件：

```env
# DeepSeek API Key（必填）
DEEPSEEK_API_KEY=your-deepseek-api-key

# 可选配置
# HF_ENDPOINT=https://hf-mirror.com
```

### 3. 启动步骤

```bash
# 进入源代码目录
cd src

# 启动FastAPI服务
python main.py
```

服务启动后访问：
- 前端界面: http://localhost:8000
- API文档: http://localhost:8000/docs

## 功能特性

- ✅ PDF 文档批量上传与解析
- ✅ 文档内容向量化存储（ChromaDB）
- ✅ 基于语义相似度的文档检索
- ✅ 支持多轮对话的 RAG 问答系统
- ✅ 论文精读与摘要生成
- ✅ 多文档对比分析
- ✅ 美观的 Web 界面

## 项目状态

- [x] Proposal
- [x] MVP
- [ ] Final

## 使用说明

1. **上传文档**: 在左侧上传区域拖拽或选择 PDF 文件
2. **查看文档**: 在中间文献列表查看已上传的文档
3. **加入知识库**: 点击「入库」按钮将文档加入向量知识库
4. **智能问答**: 在右侧输入问题，系统基于知识库内容回答
5. **论文精读**: 选择文档，点击「论文精读」生成详细摘要
6. **文献对比**: 勾选多个文档，点击「批量对比」生成对比分析

## 许可证

MIT License
