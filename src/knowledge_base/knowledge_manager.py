"""知识库管理器模块

该模块负责管理知识库的核心操作：
- 文档分块与嵌入
- 文档添加与去重
- 知识库检索（支持增强检索）
- 文档块管理

核心组件：
- KnowledgeManager: 知识库管理器，整合分块、嵌入和存储流程
- 支持单例模式确保全局只有一个实例
- 支持增强检索（EnsembleRetriever）提升召回率
"""

from typing import List, Optional, Dict
import threading
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .config import CHUNKING_CONFIG
from .models import DocumentChunk, RetrievalResult
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
import sys
import os

# 添加父目录到路径，确保可以导入 document_processor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from document_processor.models import ProcessedDocument


class KnowledgeManager:
    """知识库管理器，整合分块、嵌入和存储流程

    主要功能：
    1. 文档分块：优先按结构化章节分块，回退到全文分块
    2. 嵌入生成：使用Ollama本地生成文本嵌入
    3. 向量存储：将文档块及其嵌入存储到ChromaDB
    4. 知识检索：支持基础向量检索和增强检索（EnsembleRetriever）
    5. 去重管理：防止重复添加相同文档

    使用单例模式确保全局只有一个实例
    """

    _instance = None
    _initialized = False
    _lock = threading.Lock()  # 添加线程锁，确保单例线程安全

    def __new__(cls):
        """单例模式实现（线程安全版）"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化组件"""
        if self._initialized:
            return

        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNKING_CONFIG["chunk_size"],
            chunk_overlap=CHUNKING_CONFIG["chunk_overlap"],
            separators=CHUNKING_CONFIG["separators"],
            length_function=len
        )
        # 初始化嵌入服务
        self.embedding_service = EmbeddingService()
        # 初始化向量存储
        self.vector_store = VectorStore()
        # 延迟初始化增强检索器（首次使用时创建）
        self.enhanced_retriever = None
        self._initialized = True

    def add_document_to_knowledge_base(self, doc: ProcessedDocument, overwrite: bool = False) -> int:
        """
        将单个处理好的文献添加到知识库

        Args:
            doc: 处理后的文献对象（ProcessedDocument）
            overwrite: 如果文档已存在，是否覆盖（删除旧的再添加）

        Returns:
            添加的块数量（0表示跳过或失败）

        流程：
        1. 检查文档处理状态是否为成功
        2. 检查文档是否已存在
        3. 如果存在且overwrite=True，先添加新块再删除旧块（事务性处理）
        4. 将文档分割成块（优先按结构化章节分块）
        5. 批量生成嵌入
        6. 添加到向量数据库
        """
        if not self._initialized:
            raise RuntimeError("知识库管理器未初始化")

        # 检查文档处理状态
        if doc.processing_status != "success":
            error_msg = doc.error_message or "未知错误"
            print(f"❌ 文档 {doc.file_name} 处理失败，无法添加到知识库: {error_msg}")
            return 0

        # 检查文档是否已存在于知识库中
        existing_chunks = self.vector_store.get_chunks_by_document_id(doc.document_id)
        doc_title = doc.metadata.title or doc.file_name

        if existing_chunks > 0:
            if not overwrite:
                # 跳过重复文档
                print(f"⚠️ 文档 '{doc_title}' 已存在于知识库中，跳过导入（如需覆盖请设置 overwrite=True）")
                return 0
            else:
                print(f"🔄 文档 '{doc_title}' 已存在，将进行覆盖更新")

        try:
            # 1. 将文档分块（优先按结构化章节分块）
            chunks = self._split_document_into_chunks(doc)

            if not chunks:
                print(f"⚠️ 文档 '{doc_title}' 没有可分块的内容，跳过导入")
                return 0

            # 2. 批量生成嵌入
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_service.embed_texts(texts)

            # 3. 将嵌入赋值给块
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding

            # 4. 添加到向量数据库
            self.vector_store.add_chunks(chunks)

            # 5. 如果是覆盖模式，现在才删除旧块（避免旧块已删新块未加的情况）
            if overwrite and existing_chunks > 0:
                self.vector_store.delete_by_document_id(doc.document_id)
                print(f"✅ 已删除文档 '{doc_title}' 的旧版本")

            # 6. 刷新增强检索器（如果已初始化）
            if self.enhanced_retriever is not None:
                try:
                    self.enhanced_retriever.refresh_bm25()
                except Exception as e:
                    print(f"⚠️ 刷新增强检索器失败: {e}")

            print(f"✅ 文档 '{doc_title}' 已成功添加到知识库，共 {len(chunks)} 个块")
            return len(chunks)
        except Exception as e:
            print(f"❌ 添加文档 '{doc_title}' 失败: {e}")
            return 0

    def _split_document_into_chunks(self, doc: ProcessedDocument) -> List[DocumentChunk]:
        """
        将文档分割成多个块（完全匹配ProcessedDocument模型 + 结构化分块优化）

        Args:
            doc: 处理后的文献对象

        Returns:
            文档块列表

        分块策略：
        1. 优先使用结构化章节分块（保留章节边界，检索效果更好）
        2. 如果没有结构化章节，回退到全文分块
        3. 每个块都会包含完整的元数据信息
        """
        chunks = []

        # 优先使用结构化章节分块（效果更好）
        if doc.sections and len(doc.sections) > 0:
            print(f"📑 使用结构化章节分块: {len(doc.sections)} 个章节")

            for section_index, section in enumerate(doc.sections):
                # 跳过空章节
                if not section.content or not section.content.strip():
                    continue

                # 对每个章节单独分块（保留章节边界）
                section_chunks = self.text_splitter.split_text(section.content)

                for i, chunk_content in enumerate(section_chunks):
                    # 构建更丰富的元数据
                    metadata = {
                        "title": doc.metadata.title,
                        "authors": ", ".join(doc.metadata.authors) if doc.metadata.authors else "",
                        "publication_date": doc.metadata.publication_date,
                        "journal": doc.metadata.journal,
                        "volume": doc.metadata.volume,
                        "issue": doc.metadata.issue,
                        "pages": doc.metadata.pages,
                        "doi": doc.metadata.doi,
                        "abstract": doc.metadata.abstract,
                        "keywords": ", ".join(doc.metadata.keywords) if doc.metadata.keywords else "",
                        "publisher": doc.metadata.publisher,
                        "language": doc.metadata.language,
                        "section_title": section.title,
                        "section_level": section.level,
                        "section_page": section.page_number,
                        "file_size": doc.file_size,
                        "upload_time": doc.upload_time.isoformat() if doc.upload_time else "",
                        "processing_status": doc.processing_status
                    }

                    # 过滤掉None值（ChromaDB不接受None作为metadata值）
                    metadata = {k: v for k, v in metadata.items() if v is not None}

                    chunk = DocumentChunk(
                        document_id=doc.document_id,
                        file_name=doc.file_name,
                        chunk_index=len(chunks),  # 全局块索引
                        content=f"【{section.title}】\n{chunk_content.strip()}",  # 保留章节标题
                        metadata=metadata
                    )
                    chunks.append(chunk)

        # 如果没有结构化章节，回退到全文分块
        else:
            print("📄 未找到结构化章节，使用全文分块")

            if not doc.full_text or not doc.full_text.strip():
                print(f"⚠️ 文档 {doc.file_name} 全文为空，跳过分块")
                return []

            split_chunks = self.text_splitter.split_text(doc.full_text)

            for i, chunk_content in enumerate(split_chunks):
                metadata = {
                    "title": doc.metadata.title,
                    "authors": ", ".join(doc.metadata.authors) if doc.metadata.authors else "",
                    "publication_date": doc.metadata.publication_date,
                    "journal": doc.metadata.journal,
                    "volume": doc.metadata.volume,
                    "issue": doc.metadata.issue,
                    "pages": doc.metadata.pages,
                    "doi": doc.metadata.doi,
                    "abstract": doc.metadata.abstract,
                    "keywords": ", ".join(doc.metadata.keywords) if doc.metadata.keywords else "",
                    "publisher": doc.metadata.publisher,
                    "language": doc.metadata.language,
                    "file_size": doc.file_size,
                    "upload_time": doc.upload_time.isoformat() if doc.upload_time else "",
                    "processing_status": doc.processing_status
                }

                # 过滤掉None值
                metadata = {k: v for k, v in metadata.items() if v is not None}

                chunk = DocumentChunk(
                    document_id=doc.document_id,
                    file_name=doc.file_name,
                    chunk_index=i,
                    content=chunk_content.strip(),
                    metadata=metadata
                )
                chunks.append(chunk)

        return chunks

    def search_knowledge_base(self, query: str, top_k: int = None, use_enhanced: bool = True) -> List[RetrievalResult]:
        """
        检索知识库

        Args:
            query: 搜索查询文本
            top_k: 返回结果数量（默认使用配置值）
            use_enhanced: 是否使用增强检索器（EnsembleRetriever）

        Returns:
            检索结果列表，按相似度排序（降序）

        支持两种检索模式：
        1. 增强检索（默认）：结合向量检索和BM25检索，提升召回率
        2. 基础检索：纯向量检索
        """
        if not self._initialized:
            raise RuntimeError("知识库管理器未初始化")

        if use_enhanced:
            return self._search_with_enhanced_retriever(query, top_k)
        else:
            return self.vector_store.search(query, top_k)

    def _search_with_enhanced_retriever(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """使用增强检索器搜索（EnsembleRetriever）"""
        if top_k is None:
            top_k = 5

        # 延迟初始化增强检索器（动态导入避免启动时加载）
        if self.enhanced_retriever is None:
            try:
                from .enhanced_retriever import EnhancedRetriever
                self.enhanced_retriever = EnhancedRetriever()
            except Exception as e:
                print(f"⚠️ 增强检索器初始化失败，使用基础检索: {e}")
                return self.vector_store.search(query, top_k)

        try:
            results = self.enhanced_retriever.search(query, top_k)

            retrieval_results = []
            # 改动：enumerate 生成rank序号，从1开始
            for rank, result in enumerate(results, start=1):
                chunk = DocumentChunk(
                    document_id=result["metadata"].get("document_id", ""),
                    file_name=result["file_name"],
                    chunk_index=result["metadata"].get("chunk_index", 0),
                    content=result["content"],
                    metadata=result["metadata"]
                )
                # 补齐 rank=rank，修复缺参报错
                retrieval_results.append(RetrievalResult(rank=rank, chunk=chunk, score=result["score"]))

            return retrieval_results
        except Exception as e:
            print(f"⚠️ 增强检索失败，使用基础检索: {e}")
            return self.vector_store.search(query, top_k)

    def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """
        获取指定文档的所有块（直接从向量数据库获取，不进行语义检索）

        Args:
            document_id: 文档ID

        Returns:
            文档块列表

        用于论文精读等场景，需要获取文档的完整内容。
        """
        if not self._initialized:
            raise RuntimeError("知识库管理器未初始化")
        return self.vector_store.get_chunks_by_document_id_full(document_id)

    def delete_document_from_knowledge_base(self, document_id: str) -> bool:
        """
        从知识库中删除指定文档

        Args:
            document_id: 文档ID

        Returns:
            是否删除成功
        """
        if not self._initialized:
            raise RuntimeError("知识库管理器未初始化")

        result = self.vector_store.delete_by_document_id(document_id)

        # 刷新增强检索器
        if self.enhanced_retriever is not None:
            try:
                self.enhanced_retriever.refresh_bm25()
            except Exception as e:
                print(f"⚠️ 刷新增强检索器失败: {e}")

        return result

    def get_statistics(self) -> Dict[str, int]:
        """
        获取知识库统计信息

        Returns:
            统计字典，包含文档数、块数等信息
        """
        if not self._initialized:
            raise RuntimeError("知识库管理器未初始化")
        return self.vector_store.get_statistics()

    def get_knowledge_base_stats(self) -> Dict[str, int]:
        """
        获取知识库统计信息（API接口使用的别名方法）

        Returns:
            统计字典，包含文档数、块数等信息
        """
        return self.get_statistics()

    def clear_knowledge_base(self) -> bool:
        """
        清空整个知识库

        Returns:
            是否清空成功
        """
        if not self._initialized:
            raise RuntimeError("知识库管理器未初始化")

        result = self.vector_store.clear_all()

        # 刷新增强检索器
        if self.enhanced_retriever is not None:
            try:
                self.enhanced_retriever.refresh_bm25()
            except Exception as e:
                print(f"⚠️ 刷新增强检索器失败: {e}")

        return result

    def remove_document_from_knowledge_base(self, document_id: str) -> bool:
        """
        从知识库中删除指定文档（API接口使用的别名方法）

        Args:
            document_id: 文档ID

        Returns:
            是否删除成功
        """
        return self.delete_document_from_knowledge_base(document_id)