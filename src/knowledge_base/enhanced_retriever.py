"""增强版检索器，使用 EnsembleRetriever 提升召回率"""
from typing import List, Dict, Any
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from .config import VECTOR_STORE_CONFIG, CHUNKING_CONFIG
from .models import RetrievalResult


class EnhancedRetriever:
    """增强版检索器，结合向量检索和BM25检索"""

    def __init__(self):
        self._initialize_retrievers()

    def _initialize_retrievers(self):
        """初始化各种检索器"""
        from .embedding_service import EmbeddingService
        self.embeddings = EmbeddingService()

        self.vector_store = Chroma(
            persist_directory=VECTOR_STORE_CONFIG["persist_directory"],
            embedding_function=self.embeddings,
            collection_name=VECTOR_STORE_CONFIG["collection_name"]
        )

        # ==========改动1：删除固定k=10，运行时动态传入top_k，消除超限警告==========
        self.vector_retriever = self.vector_store.as_retriever()

        self._init_bm25_retriever()

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever] if self.bm25_retriever else [self.vector_retriever],
            weights=[0.6, 0.4] if self.bm25_retriever else [1.0]
        )

        self.compressor = EmbeddingsFilter(
            embeddings=self.embeddings,
            similarity_threshold=0.3
        )
        # ==========改动2：实例化压缩检索器，启用EmbeddingsFilter过滤（原来定义未使用）==========
        self.compression_retriever = ContextualCompressionRetriever(
            base_retriever=self.ensemble_retriever,
            base_compressor=self.compressor
        )

    def _init_bm25_retriever(self):
        all_docs = self.vector_store.get()
        if all_docs and all_docs["documents"]:
            from langchain.docstore.document import Document
            langchain_docs = []
            for i, doc in enumerate(all_docs["documents"]):
                metadata = all_docs["metadatas"][i] if all_docs["metadatas"] else {}
                langchain_docs.append(Document(page_content=doc, metadata=metadata))
            self.bm25_retriever = BM25Retriever.from_documents(langchain_docs)
            # ==========改动3：去掉全局固定k=10，查询时动态赋值==========
        else:
            self.bm25_retriever = None

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        # 修正获取向量总数
        try:
            total_chunk = self.vector_store._collection.count()
        except Exception:
            all_data = self.vector_store.get()
            total_chunk = len(all_data.get("ids", []))
        fetch_k = min(top_k * 2, total_chunk)  # 限制查询条数不超存量，消除6>5告警

        try:
            if self.bm25_retriever:
                self.vector_retriever.search_kwargs = {"k": fetch_k}
                self.bm25_retriever.k = fetch_k
                ensemble_results = self.compression_retriever.invoke(query)
            else:
                self.vector_retriever.search_kwargs = {"k": fetch_k}
                ensemble_results = self.vector_retriever.invoke(query)
            # 后续原有去重、生成rank代码不变
            # 后续去重、rank逻辑不动
            seen_contents = set()
            unique_results = []
            for doc in ensemble_results:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    unique_results.append(doc)
                    if len(unique_results) >= top_k:
                        break

            for rank, doc in enumerate(unique_results[:top_k], start=1):
                score = 1.0 - (rank / (top_k + 1))
                res_obj = RetrievalResult(rank=rank, chunk=doc.page_content, score=score)
                res_obj.file_name = doc.metadata.get("file_name", "unknown")
                res_obj.metadata = doc.metadata
                item = {
                    "content": res_obj.chunk,
                    "file_name": res_obj.file_name,
                    "score": res_obj.score,
                    "metadata": res_obj.metadata
                }
                results.append(item)

        except Exception as e:
            print(f"增强检索异常: {str(e)}")
            # 降级同样限制k不能超存量
            try:
                safe_k = min(top_k, total_chunk)
                self.vector_retriever.search_kwargs = {"k": safe_k}
                vector_results = self.vector_retriever.invoke(query)
                for rank, doc in enumerate(vector_results[:top_k], start=1):
                    score = 1.0 - (rank / (top_k + 1))
                    res_obj = RetrievalResult(rank=rank, chunk=doc.page_content, score=score)
                    res_obj.file_name = doc.metadata.get("file_name", "unknown")
                    res_obj.metadata = doc.metadata
                    item = {
                        "content": res_obj.chunk,
                        "file_name": res_obj.file_name,
                        "score": res_obj.score,
                        "metadata": res_obj.metadata
                    }
                    results.append(item)
            except Exception as ee:
                print(f"降级检索也失败:{str(ee)}")
        return results

    def refresh_bm25(self):
        self._init_bm25_retriever()
        if self.bm25_retriever:
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.vector_retriever, self.bm25_retriever],
                weights=[0.6, 0.4]
            )
            # 同步刷新压缩检索器
            self.compression_retriever = ContextualCompressionRetriever(
                base_retriever=self.ensemble_retriever,
                base_compressor=self.compressor
            )