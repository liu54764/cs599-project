# 系统架构说明

## 整体架构

本项目采用前后端分离的架构设计。

## 核心组件

### 1. 前端 (Frontend)
- 位置：`src/frontend/`
- 技术栈：HTML + Tailwind CSS + Font Awesome
- 功能：
  - 文档上传界面
  - 文献列表展示
  - 知识库统计信息
  - RAG 问答交互
  - 检索结果展示

### 2. 后端 (Backend)
位置：`src/`
技术栈：FastAPI + Python

#### 2.1 文档处理模块 (document_processor)

位置：`src/document_processor/`
- `pdf_extractor.py`：PDF 文档解析
- `text_cleaner.py`：文本
- `document_manager.py`：文档管理
- `models.py`：数据模型

#### 2.2 知识库模块 (knowledge_base)
位置：`src/knowledge_base/`
- `embedding_service.py`：向量化服务
- `vector_store.py`：向量数据库
- `knowledge_manager.py`：知识库管理
- `enhanced_retriever.py`：增强检索器

#### 2.3 问答模块 (qa_module)
位置：`src/qa_module/`
- `qa_engine.py`：问答引擎
- `llm_client.py`：LLM 客户端
- `prompt_templates.py`：提示词模板
- `question_classifier.py`：问题分类器
- `langchain_qa_engine.py`：LangChain 集成

## 工作流程

### 文档处理流程
1. 用户上传 PDF 文档
2. PDF 文档被解析提取文本
3. 文本进行向量化处理
4. 向量化结果存储到向量数据库

### 问答流程
1. 用户输入问题
2. 问题分类（闲聊/知识问题）
3. 从向量数据库检索相关文档片段
4. 将检索结果与问题一起发送给 LLM
5. LLM 生成回答
6. 返回结果给用户

## API 接口

### 文档相关接口
- `POST /upload` - 上传 PDF 文档
- `GET /documents` - 获取文档列表
- `GET /documents/{id}` - 获取文档详情
- `DELETE /documents/{id}` - 删除文档

### 知识库相关接口
- `POST /knowledge-base/add/{id}` - 将文档加入知识库
- `GET /knowledge-base/stats` - 获取知识库统计
- `DELETE /knowledge-base/documents/{id}` - 从知识库删除文档
- `DELETE /knowledge-base/clear` - 清空知识库
- `POST /knowledge-base/search` - 检索知识库

### 问答相关接口
- `POST /qa/ask` - 问答
- `POST /qa/summarize` - 论文摘要
- `POST /qa/compare` - 多文档对比

## 数据存储

### 向量数据库
位置：`data/vector_db/`
- 使用 ChromaDB 存储向量数据

### 模型文件
位置：`src/models/`
- Embedding 模型文件存储在这里
