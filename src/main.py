import os
# 强制hf镜像，永久生效
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, StreamingResponse
import asyncio
import json
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import shutil
from pathlib import Path

# 导入业务模块
from document_processor import DocumentManager, ProcessedDocument
from knowledge_base import KnowledgeManager
# =====新增：导入问答引擎=====
from qa_module import QAEngine

app = FastAPI(title="研究生学术助手API")

# 配置CORS，允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建必要的目录
UPLOAD_DIR = Path("./data/uploads")
STORAGE_DIR = Path("./data/documents")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
STORAGE_DIR.mkdir(exist_ok=True, parents=True)

# 初始化业务管理器（单例模式）
doc_manager = DocumentManager(storage_dir=str(STORAGE_DIR))
knowledge_manager = KnowledgeManager()
# =====新增：初始化DeepSeek问答引擎=====
qa_engine = QAEngine()


# ==================== Pydantic数据模型 ====================
class DocumentInfo(BaseModel):
    document_id: str
    file_name: str
    file_size: int
    upload_time: str
    title: Optional[str]
    authors: List[str]
    doi: Optional[str]
    processing_status: str


class DocumentDetail(DocumentInfo):
    abstract: Optional[str]
    keywords: List[str]
    full_text: str
    sections: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]


class UploadResponse(BaseModel):
    success: bool
    message: str
    document: Optional[DocumentInfo] = None


class BatchUploadResponse(BaseModel):
    success: int
    failed: int
    documents: List[DocumentInfo]


class KnowledgeBaseStats(BaseModel):
    document_count: int
    chunk_count: int
    embedding_dimension: int


class AddToKBResponse(BaseModel):
    success: bool
    message: str
    chunks_added: int = 0


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    file_name: str
    content: str
    metadata: dict
    score: float
    rank: int


class SearchResponse(BaseModel):
    success: bool
    results: List[SearchResult]
    total: int

# ==========新增QA专用请求/返回模型==========
class ChatMessage(BaseModel):
    role: str  # "user" 或 "assistant"
    content: str

class QARequest(BaseModel):
    query: str
    chat_history: List[ChatMessage] = []  # 对话历史，可选
    use_memory: bool = True  # 是否启用对话记忆

class QAResponse(BaseModel):
    success: bool
    answer: str
    source_documents: List[Dict[str, Any]]
    retrieval_count: int
    question_type: str = "knowledge_with_retrieval"  # 问题类型标识

class PaperSummaryRequest(BaseModel):
    document_id: str

class PaperSummaryResponse(BaseModel):
    success: bool
    summary: str
    source_document_id: str
    chunks_used: int = 0

class PaperComparisonRequest(BaseModel):
    document_ids: List[str]

class PaperComparisonResponse(BaseModel):
    success: bool
    comparison: str
    source_document_ids: List[str]


# ==================== 工具函数 ====================
def doc_to_info(doc: ProcessedDocument) -> DocumentInfo:
    """将ProcessedDocument转换为API响应模型"""
    return DocumentInfo(
        document_id=doc.document_id,
        file_name=doc.file_name,
        file_size=doc.file_size,
        upload_time=doc.upload_time.isoformat(),
        title=doc.metadata.title,
        authors=doc.metadata.authors,
        doi=doc.metadata.doi,
        processing_status=doc.processing_status
    )


def doc_to_detail(doc: ProcessedDocument) -> DocumentDetail:
    """将ProcessedDocument转换为详细信息响应模型"""
    info = doc_to_info(doc)
    return DocumentDetail(
        **info.model_dump(),
        abstract=doc.metadata.abstract,
        keywords=doc.metadata.keywords,
        full_text=doc.full_text,
        sections=[{"title": s.title, "content": s.content, "level": s.level, "page_number": s.page_number} for s in
                  doc.sections],
        tables=doc.tables
    )


# ==================== API路由 ====================
@app.get("/", include_in_schema=False)
async def root():
    """根路径重定向到前端界面"""
    return RedirectResponse(url="/index.html")


@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """上传并处理单个PDF文件"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="只支持PDF文件")

        # 保存上传的临时文件
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 处理文献并保存到存储目录
        doc = doc_manager.upload_document(str(file_path), copy_to_storage=True)

        # 删除临时文件
        os.remove(file_path)

        return UploadResponse(
            success=True,
            message=f"文件 {file.filename} 处理成功",
            document=doc_to_info(doc)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/batch-upload", response_model=BatchUploadResponse)
async def batch_upload(files: List[UploadFile] = File(...)):
    """批量上传并处理多个PDF文件"""
    success_count = 0
    failed_count = 0
    processed_docs = []

    for file in files:
        try:
            if not file.filename.lower().endswith('.pdf'):
                failed_count += 1
                continue

            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            doc = doc_manager.upload_document(str(file_path), copy_to_storage=True)
            os.remove(file_path)

            processed_docs.append(doc_to_info(doc))
            success_count += 1

        except Exception:
            failed_count += 1

    return BatchUploadResponse(
        success=success_count,
        failed=failed_count,
        documents=processed_docs
    )


@app.get("/api/documents", response_model=List[DocumentInfo])
async def list_documents():
    """获取所有已处理的文献列表"""
    docs = doc_manager.list_documents()
    return [doc_to_info(doc) for doc in docs]


@app.get("/api/documents/{document_id}", response_model=DocumentDetail)
async def get_document_detail(document_id: str):
    """获取指定文献的详细信息"""
    doc = doc_manager.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文献不存在")
    return doc_to_detail(doc)


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """删除指定文献（同时从知识库中删除）"""
    # 先从知识库中删除
    try:
        knowledge_manager.remove_document_from_knowledge_base(document_id)
    except Exception:
        pass  # 如果知识库中没有该文献，忽略错误

    # 再从文献管理器中删除
    success = doc_manager.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="文献不存在")

    return {"success": True, "message": "文献删除成功"}


# ==================== 知识库API ====================
@app.post("/api/knowledge-base/add/{document_id}", response_model=AddToKBResponse)
async def add_document_to_kb(document_id: str):
    """将指定文献添加到知识库"""
    doc = doc_manager.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文献不存在")

    try:
        chunks_added = knowledge_manager.add_document_to_knowledge_base(doc)
        return AddToKBResponse(
            success=True,
            message=f"文献 {doc.file_name} 已添加到知识库，生成 {chunks_added} 个向量块",
            chunks_added=chunks_added
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/knowledge-base/batch-add", response_model=AddToKBResponse)
async def batch_add_to_kb(document_ids: List[str]):
    """批量将多个文献添加到知识库"""
    docs = []
    for doc_id in document_ids:
        doc = doc_manager.get_document(doc_id)
        if doc:
            docs.append(doc)

    if not docs:
        raise HTTPException(status_code=404, detail="没有找到有效的文献")

    try:
        total_chunks = knowledge_manager.add_documents_to_knowledge_base(docs)
        return AddToKBResponse(
            success=True,
            message=f"成功将 {len(docs)} 篇文献添加到知识库，生成 {total_chunks} 个向量块",
            chunks_added=total_chunks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/knowledge-base/search", response_model=SearchResponse)
async def search_knowledge_base(request: SearchRequest):
    """检索知识库"""
    try:
        results = knowledge_manager.search_knowledge_base(request.query, request.top_k)

        # 转换为响应模型
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                chunk_id=result.chunk.chunk_id,
                document_id=result.chunk.document_id,
                file_name=result.chunk.file_name,
                content=result.chunk.content,
                metadata=result.chunk.metadata,
                score=result.score,
                rank=result.rank
            ))

        return SearchResponse(
            success=True,
            results=search_results,
            total=len(search_results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/knowledge-base/documents/{document_id}")
async def remove_document_from_kb(document_id: str):
    """从知识库中删除指定文献"""
    try:
        knowledge_manager.remove_document_from_knowledge_base(document_id)
        return {"success": True, "message": "文献已从知识库中删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/knowledge-base/stats", response_model=KnowledgeBaseStats)
async def get_kb_stats():
    """获取知识库统计信息"""
    return knowledge_manager.get_knowledge_base_stats()


@app.delete("/api/knowledge-base/clear")
async def clear_knowledge_base():
    """清空整个知识库（谨慎使用）"""
    try:
        knowledge_manager.clear_knowledge_base()
        return {"success": True, "message": "知识库已清空"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========新增：DeepSeek RAG问答三个接口==========
@app.post("/api/qa/ask", response_model=QAResponse)
async def ask_question(request: QARequest):
    """基于知识库RAG问答，支持多轮对话记忆"""
    try:
        # 如果有对话历史且启用记忆，使用带记忆的问答
        if request.chat_history and request.use_memory:
            # 转换对话历史格式
            history = []
            for i in range(0, len(request.chat_history), 2):
                if i + 1 < len(request.chat_history):
                    user_msg = request.chat_history[i]
                    ai_msg = request.chat_history[i + 1]
                    if user_msg.role == "user" and ai_msg.role == "assistant":
                        history.append((user_msg.content, ai_msg.content))
            
            res = qa_engine.answer_question(
                query=request.query,
                chat_history=history if history else None
            )
        else:
            res = qa_engine.answer_question(request.query)
        
        return QAResponse(
            success=True,
            answer=res["answer"],
            source_documents=res.get("source_documents", []),
            retrieval_count=res.get("retrieval_count", 0),
            question_type=res.get("question_type", "knowledge_with_retrieval")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def generate_stream(request: QARequest):
    """生成流式响应的异步生成器"""
    try:
        history = None
        if request.chat_history and request.use_memory:
            history = []
            for i in range(0, len(request.chat_history), 2):
                if i + 1 < len(request.chat_history):
                    user_msg = request.chat_history[i]
                    ai_msg = request.chat_history[i + 1]
                    if user_msg.role == "user" and ai_msg.role == "assistant":
                        history.append((user_msg.content, ai_msg.content))
        
        for chunk in qa_engine.answer_question_stream(
            query=request.query,
            chat_history=history if history else None
        ):
            data = json.dumps(chunk, ensure_ascii=False)
            yield f"data: {data}\n\n"
            await asyncio.sleep(0)
    except Exception as e:
        error_data = json.dumps({
            "chunk": f"错误: {str(e)}",
            "is_finished": True,
            "source_documents": [],
            "retrieval_count": 0,
            "question_type": "error"
        }, ensure_ascii=False)
        yield f"data: {error_data}\n\n"

@app.post("/api/qa/ask/stream")
async def ask_question_stream(request: QARequest):
    """基于知识库RAG问答（流式输出），支持多轮对话记忆"""
    return StreamingResponse(
        generate_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream; charset=utf-8"
        }
    )

@app.post("/api/qa/summarize", response_model=PaperSummaryResponse)
async def summarize_paper(req:PaperSummaryRequest):
    """单篇论文结构化精读"""
    try:
        res=qa_engine.summarize_paper(req.document_id)
        return PaperSummaryResponse(success=True,**res)
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

@app.post("/api/qa/compare", response_model=PaperComparisonResponse)
async def compare_papers(req:PaperComparisonRequest):
    """多篇论文对比分析"""
    try:
        res=qa_engine.compare_papers(req.document_ids)
        return PaperComparisonResponse(success=True,**res)
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))


# ==================== 静态文件挂载 ====================
# 挂载前端静态文件（必须在所有API路由之后）
app.mount("/", StaticFiles(directory="src/frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 50)
    print("🚀 研究生学术助手 启动成功！")
    print("=" * 50)
    print(f"🌐 前端界面: http://localhost:8000")
    print(f"📚 API文档: http://localhost:8000/docs")
    print(f"🔧 备用文档: http://localhost:8000/redoc")
    print("=" * 50)
    print("提示: 第一次运行会自动下载嵌入模型，请耐心等待")
    print("=" * 50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)