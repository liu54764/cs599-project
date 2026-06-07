"""向量数据库模块

该模块封装了 ChromaDB 向量数据库的操作：
- 初始化和管理向量数据库连接
- 添加、删除、查询文档块
- 语义检索相关文档

核心组件：
- VectorStore: Chroma向量数据库封装类
- 支持自定义嵌入函数
- 支持持久化存储

配置说明：
- persist_directory: 向量数据库持久化目录
- collection_name: 集合名称
- embedding_function: 嵌入函数（设置为None时手动提供嵌入）
"""
# 必须在导入 chromadb 之前设置环境变量，禁用遥测
import os

os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

import chromadb
from typing import List, Dict, Optional
from .config import VECTOR_STORE_CONFIG, RETRIEVAL_CONFIG
from .models import DocumentChunk, RetrievalResult
from .embedding_service import EmbeddingService


class VectorStore:
    """Chroma向量数据库封装类

    提供向量数据库的核心操作：
    - 添加文档块（带嵌入向量）
    - 删除指定文档的所有块
    - 语义检索相关文档
    - 获取指定文档的所有块
    - 获取统计信息

    使用持久化存储，数据保存在本地文件系统。
    """

    def __init__(self):
        """初始化向量数据库"""
        # 配置参数
        self.persist_directory = VECTOR_STORE_CONFIG["persist_directory"]
        self.collection_name = VECTOR_STORE_CONFIG["collection_name"]

        # 初始化嵌入服务（用于手动生成嵌入）
        self.embedding_service = EmbeddingService()

        # 初始化Chroma持久化客户端
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # 获取或创建集合（不使用内置嵌入函数，手动提供嵌入）
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "学术文献向量数据库"},
            embedding_function=None  # 使用我们自己的嵌入服务
        )

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        批量添加文档块到向量数据库

        Args:
            chunks: 文档块列表（DocumentChunk）

        流程：
        1. 准备数据：ID、文档内容、元数据、嵌入向量
        2. 过滤元数据中的None值（ChromaDB不接受None）
        3. 如果块没有预先生成的嵌入，自动生成
        4. 添加到集合
        """
        if not chunks:
            return

        # 准备数据
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = []

        for chunk in chunks:
            metadata = {
                "document_id": chunk.document_id,
                "file_name": chunk.file_name,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata
            }
            # 过滤掉 None 值，ChromaDB 不接受 None 作为 metadata 值
            metadata = {k: v for k, v in metadata.items() if v is not None}
            metadatas.append(metadata)

        # 生成嵌入（如果没有预先生成）
        embeddings = []
        for chunk in chunks:
            if chunk.embedding is not None:
                embeddings.append(chunk.embedding)
            else:
                embeddings.append(self.embedding_service.embed_text(chunk.content))

        # 添加到集合
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def search(self, query: str, top_k: int = None, score_threshold: float = None) -> List[RetrievalResult]:
        """
        语义检索相关的文档块

        Args:
            query: 查询文本
            top_k: 返回结果数量（默认使用配置值）
            score_threshold: 相似度阈值（默认使用配置值）

        Returns:
            检索结果列表，按相似度排序（降序）

        流程：
        1. 将查询文本转换为嵌入向量
        2. 在向量数据库中进行相似度搜索
        3. 将结果转换为统一格式返回
        """
        top_k = top_k or RETRIEVAL_CONFIG["top_k"]
        score_threshold = score_threshold or RETRIEVAL_CONFIG["score_threshold"]

        # 生成查询嵌入
        query_embedding = self.embedding_service.embed_text(query)

        # 查询向量数据库
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # 转换结果格式
        retrieval_results = []
        if results and results["documents"]:
            # enumerate生成rank，从1开始
            for rank, i in enumerate(range(len(results["documents"][0])), start=1):
                # 计算相似度分数（距离越小相似度越高）
                distance = results["distances"][0][i]
                score = max(0, 1.0 - distance)  # 转换为相似度分数

                # 应用阈值过滤
                if score >= score_threshold:
                    chunk = DocumentChunk(
                        document_id=results["metadatas"][0][i].get("document_id", ""),
                        file_name=results["metadatas"][0][i].get("file_name", "unknown"),
                        chunk_index=results["metadatas"][0][i].get("chunk_index", 0),
                        content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i]
                    )
                    # 新增 rank=rank，补齐必填参数
                    retrieval_results.append(RetrievalResult(rank=rank, chunk=chunk, score=score))

        return retrieval_results

    def get_chunks_by_document_id(self, document_id: str) -> int:
        """
        获取指定文档的块数量

        Args:
            document_id: 文档ID

        Returns:
            该文档的块数量

        用于检查文档是否已存在于知识库中。
        """
        results = self.collection.get(
            where={"document_id": document_id},
            limit=1
        )
        return len(results["ids"])

    def get_chunks_by_document_id_full(self, document_id: str) -> List[DocumentChunk]:
        """
        获取指定文档的所有块（完整内容）

        Args:
            document_id: 文档ID

        Returns:
            文档块列表

        用于论文精读等场景，需要获取文档的完整内容。
        """
        results = self.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"]
        )

        chunks = []
        if results and results["ids"]:
            # 按chunk_index排序
            indexed_results = list(zip(
                results["ids"],
                results["documents"],
                results["metadatas"]
            ))
            indexed_results.sort(key=lambda x: x[2].get("chunk_index", 0))

            for chunk_id, content, metadata in indexed_results:
                chunk = DocumentChunk(
                    document_id=metadata.get("document_id", ""),
                    file_name=metadata.get("file_name", "unknown"),
                    chunk_index=metadata.get("chunk_index", 0),
                    content=content,
                    metadata=metadata
                )
                chunks.append(chunk)

        return chunks

    def delete_by_document_id(self, document_id: str) -> bool:
        """
        删除指定文档的所有块（修复ChromaDB v0.4+ API不兼容问题）

        Args:
            document_id: 文档ID

        Returns:
            是否删除成功
        """
        try:
            # 第一步：查询出所有属于该文档的chunk_id（只查ID，提高效率）
            results = self.collection.get(
                where={"document_id": document_id},
                include=[]
            )

            if not results["ids"]:
                print(f"⚠️ 未找到文档ID为 {document_id} 的任何块")
                return False

            # 第二步：根据chunk_id批量删除（这是v0.4+唯一支持的删除方式）
            self.collection.delete(ids=results["ids"])
            print(f"✅ 成功删除文档 {document_id} 的 {len(results['ids'])} 个块")
            return True
        except Exception as e:
            print(f"❌ 删除文档失败: {e}")
            return False

    def get_statistics(self) -> Dict[str, int]:
        """
        获取知识库统计信息

        Returns:
            统计字典：{"document_count": 文档数, "chunk_count": 块数, "embedding_dimension": 嵌入维度}
        """
        # 获取所有文档ID并去重
        results = self.collection.get(include=["metadatas"])
        document_ids = set()

        if results and results["metadatas"]:
            for metadata in results["metadatas"]:
                doc_id = metadata.get("document_id")
                if doc_id:
                    document_ids.add(doc_id)

        return {
            "document_count": len(document_ids),
            "chunk_count": self.collection.count(),
            "embedding_dimension": getattr(self.embedding_service, 'embedding_dim', 384)
        }

    def clear_all(self) -> bool:
        """
        清空整个向量数据库集合（修复ChromaDB v0.4+ API不兼容问题）

        Returns:
            是否清空成功
        """
        try:
            # 获取所有chunk_id
            all_ids = self.collection.get(include=[])["ids"]

            if not all_ids:
                print("⚠️ 知识库已经是空的")
                return True

            # 批量删除所有数据
            self.collection.delete(ids=all_ids)
            print(f"✅ 成功清空知识库，共删除 {len(all_ids)} 个块")
            return True
        except Exception as e:
            print(f"❌ 清空知识库失败: {e}")
            return False