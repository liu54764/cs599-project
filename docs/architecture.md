# 系统架构说明

## 整体架构

本项目采用前后端分离的架构设计，基于 RAG（Retrieval-Augmented Generation）技术实现智能问答。

## 核心组件

### 1. 前端 (Frontend)
- 位置：`src/frontend/`
- 技术栈：HTML + Tailwind CSS + Font Awesome
- 功能：
  - 文档上传界面
  - 文献列表展示
  - 知识库统计信息
  - RAG 问答交互（智能问答/知识库检索切换）
  - 检索结果展示

### 2. 后端 (Backend)
位置：`src/`
技术栈：FastAPI + Python

#### 2.1 文档处理模块 (document_processor)

位置：`src/document_processor/`
- `pdf_extractor.py`：PDF 文档解析
- `text_cleaner.py`：文本清洗与预处理
- `document_manager.py`：文档管理
- `models.py`：数据模型

#### 2.2 知识库模块 (knowledge_base)

位置：`src/knowledge_base/`
- `embedding_service.py`：向量化服务
- `vector_store.py`：向量数据库
- `knowledge_manager.py`：知识库管理
- `enhanced_retriever.py`：增强检索器（向量+BM25混合检索）
- `config.py`：配置文件

#### 2.3 问答模块 (qa_module)

位置：`src/qa_module/`
- `qa_engine.py`：问答引擎（主入口）
- `llm_client.py`：LLM 客户端
- `prompt_templates.py`：提示词模板
- `langgraph_rag_workflow.py`：LangGraph 工作流（内置对话记忆和历史压缩）

## 工作流程

### 文档处理流程
1. 用户上传 PDF 文档
2. PDF 文档被解析提取文本
3. 文本进行清洗（去重、去除重复字符）
4. 文本进行向量化处理
5. 向量化结果存储到向量数据库

### 问答流程
1. 用户输入问题
2. 问题分类（闲聊/知识问题）
3. 从向量数据库检索相关文档片段（向量+BM25混合检索）
4. 将检索结果与问题一起发送给 LLM
5. LLM 生成回答（流式输出）
6. 返回结果给用户

### LangGraph 工作流

```
用户输入 → 问题分类 → 知识检索 → 生成回答 → 评估置信度 → 历史更新
                ↓                        ↓
            闲聊直接回答              置信度低则重试
                                     ↓
                              历史压缩（超过5条时触发）
```

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
- `POST /qa/ask` - 同步问答
- `POST /qa/ask/stream` - 流式问答
- `POST /qa/summarize` - 论文摘要
- `POST /qa/compare` - 多文档对比

## 检索配置

### RETRIEVAL_CONFIG

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `top_k` | 5 | 返回结果数量 |
| `score_threshold` | 0.4 | 分数阈值 |
| `fetch_multiplier` | 2 | 检索扩展倍数 |
| `use_reranking` | True | 是否启用重排序 |
| `rerank_top_n` | 5 | 重排序后保留数量 |
| `use_rrf` | True | 是否使用RRF融合 |
| `rrf_k` | 60 | RRF参数 |

### ENHANCED_RETRIEVAL_CONFIG

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vector_weight` | 0.5 | 向量检索权重 |
| `bm25_weight` | 0.5 | BM25检索权重 |
| `filter_threshold` | 0.3 | 过滤阈值 |
| `enable_multi_query` | False | 是否启用多查询生成 |
| `title_boost` | 2.0 | 标题/文件名匹配权重加成 |

## 对话记忆

### LangGraph 内置 Memory

- 使用 `MemorySaver` 自动管理对话状态
- 对话历史自动持久化到内存
- 支持多会话（通过 session_id 区分）

### 历史压缩

- 当对话历史超过 5 条时自动触发压缩
- 对早期对话进行总结，保留最近 2 条原始对话
- 节省 token 使用量

## 数据存储

### 向量数据库
位置：`data/vector_db/`
- 使用 ChromaDB 存储向量数据
- 支持持久化存储

### 模型文件
位置：`src/models/`
- Embedding 模型文件存储在这里

### 文档存储
位置：`data/documents/`
- 上传的 PDF 文件存储在这里

## 技术特点

1. **混合检索**：结合向量检索和 BM25 关键词检索
2. **RRF 融合**：使用 Reciprocal Rank Fusion 综合多个检索结果
3. **流式输出**：实时返回回答，无需等待完整生成
4. **对话记忆**：支持多轮对话，自动管理历史
5. **历史压缩**：长对话自动总结，节省资源