"""增强版检索器，使用 EnsembleRetriever + 重排序提升召回率和精度"""
import time
import logging
from functools import lru_cache
from typing import List, Dict, Any, Optional
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from .config import VECTOR_STORE_CONFIG, CHUNKING_CONFIG, RETRIEVAL_CONFIG, ENHANCED_RETRIEVAL_CONFIG
from .models import RetrievalResult

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnhancedRetriever:
    """增强版检索器，结合向量检索、BM25检索和重排序"""

    def __init__(self):
        self._initialize_retrievers()
        self._initialize_reranker()
        self._cache = {}  # 检索缓存
        self._cache_max_size = 100  # 缓存最大条目数

    def _rrf_fusion(self, results_list: List[List], k: int = 60) -> List:
        """
        使用RRF（Reciprocal Rank Fusion）融合多个检索结果列表
        
        Args:
            results_list: 多个检索结果列表的列表
            k: RRF参数，通常在60-100之间
        
        Returns:
            融合后的结果列表，按RRF分数降序排列
        """
        if len(results_list) == 1:
            return results_list[0]
        
        # 构建文档到排名的映射
        doc_rank_map = {}
        doc_content_map = {}
        
        for results in results_list:
            for rank, doc in enumerate(results, start=1):
                content = doc.page_content
                doc_content_map[content] = doc
                if content not in doc_rank_map:
                    doc_rank_map[content] = []
                doc_rank_map[content].append(rank)
        
        # 计算RRF分数
        scored_docs = []
        for content, ranks in doc_rank_map.items():
            rrf_score = sum(1.0 / (k + rank) for rank in ranks)
            scored_docs.append((doc_content_map[content], rrf_score))
        
        # 按分数降序排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs]

    def _title_boost(self, query: str, results: List) -> List:
        """
        对标题/文件名匹配的结果进行加权提升
        
        Args:
            query: 查询文本
            results: 检索结果列表
        
        Returns:
            加权后的结果列表
        """
        boost_factor = ENHANCED_RETRIEVAL_CONFIG.get("title_boost", 1.5)
        if boost_factor <= 1.0:
            return results
        
        query_tokens = set(query.lower().replace(" ", ""))
        
        scored_results = []
        for doc in results:
            # 检查标题/文件名匹配
            title_match = False
            boost_score = 1.0
            
            if hasattr(doc, 'metadata') and doc.metadata:
                # 检查文件名
                file_name = doc.metadata.get("file_name", "").lower()
                title = doc.metadata.get("title", "").lower()
                section_title = doc.metadata.get("section_title", "").lower()
                
                # 检查是否有查询词出现在标题/文件名中
                for token in query_tokens:
                    if len(token) >= 2:  # 至少2个字符
                        if token in file_name or token in title or token in section_title:
                            title_match = True
                            break
            
            if title_match:
                boost_score = boost_factor
            
            scored_results.append((doc, boost_score))
        
        # 按加权分数排序（保持原有顺序但提升标题匹配的结果）
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_results]

    def _generate_query_variants(self, query: str) -> List[str]:
        """生成查询变体（多查询检索）"""
        variants = [query]
        
        if not ENHANCED_RETRIEVAL_CONFIG.get("enable_multi_query", False):
            return variants
        
        max_variants = ENHANCED_RETRIEVAL_CONFIG.get("max_query_variants", 3)
        
        try:
            # 基于关键词生成变体
            # 1. 移除疑问词
            question_words = ["什么", "为什么", "如何", "怎样", "哪里", "哪个", "谁", "何时", "是否"]
            variant1 = query
            for qw in question_words:
                variant1 = variant1.replace(qw, "")
            variant1 = variant1.strip("？?。. ")
            if variant1 and variant1 != query:
                variants.append(variant1)
            
            # 2. 添加同义词/扩展词（简单实现）
            if len(variants) < max_variants:
                expansions = {
                    "方法": ["算法", "技术", "策略", "方式"],
                    "实验": ["实验", "试验", "测试", "验证"],
                    "研究": ["研究", "探讨", "分析", "调研"],
                    "结果": ["结果", "结论", "发现", "成果"],
                    "系统": ["系统", "框架", "平台", "架构"],
                    "模型": ["模型", "算法", "方法", "框架"],
                }
                for word, synonyms in expansions.items():
                    if word in query:
                        for synonym in synonyms[:max_variants - len(variants)]:
                            variant = query.replace(word, synonym)
                            if variant not in variants:
                                variants.append(variant)
                    if len(variants) >= max_variants:
                        break
            
            # 3. 添加简洁版（去除标点和多余空格）
            if len(variants) < max_variants:
                clean_query = re.sub(r'[，,。.？?！!、；;：:]', ' ', query)
                clean_query = re.sub(r'\s+', ' ', clean_query).strip()
                if clean_query and clean_query != query and clean_query not in variants:
                    variants.append(clean_query)
            
            logger.debug(f"生成查询变体: {variants}")
            
        except Exception as e:
            logger.warning(f"生成查询变体失败: {str(e)}")
        
        return variants[:max_variants]

    def _calculate_score(self, doc, rank: int, top_k: int) -> float:
        """更精确的相似度分数计算"""
        # 优先使用元数据中的相似度
        if hasattr(doc, 'metadata') and doc.metadata:
            if 'similarity' in doc.metadata:
                return float(doc.metadata['similarity'])
            if 'score' in doc.metadata:
                return float(doc.metadata['score'])
        
        # 使用距离计算（Chroma返回的distance是余弦距离，范围0-2）
        if hasattr(doc, 'distance'):
            # 归一化距离为相似度分数（余弦距离0表示完全相同，2表示完全不同）
            distance = float(doc.distance)
            return max(0.0, min(1.0, 1.0 - (distance / 2.0)))
        
        # 作为最后手段，使用排名估算（给予较高的基础分）
        base_score = max(0.15, 1.0 - (rank / (top_k * 1.5)))
        return base_score

    def _get_cache_key(self, query: str, top_k: int) -> str:
        """生成缓存键"""
        return f"{query}_{top_k}"

    def _add_to_cache(self, query: str, top_k: int, results: List[Dict[str, Any]]):
        """添加结果到缓存"""
        key = self._get_cache_key(query, top_k)
        
        # 清理过期缓存（FIFO策略）
        if len(self._cache) >= self._cache_max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = {
            'results': results,
            'timestamp': time.time()
        }

    def _get_from_cache(self, query: str, top_k: int) -> Optional[List[Dict[str, Any]]]:
        """从缓存获取结果（缓存有效期5分钟）"""
        key = self._get_cache_key(query, top_k)
        cached = self._cache.get(key)
        
        if cached:
            # 检查缓存是否过期（5分钟）
            if time.time() - cached['timestamp'] < 300:
                logger.debug(f"命中缓存: {query[:30]}...")
                return cached['results']
            else:
                del self._cache[key]
        
        return None

    def _initialize_retrievers(self):
        """初始化各种检索器"""
        from .embedding_service import EmbeddingService
        self.embeddings = EmbeddingService()

        import os
        os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
        self.vector_store = Chroma(
            persist_directory=VECTOR_STORE_CONFIG["persist_directory"],
            embedding_function=self.embeddings,
            collection_name=VECTOR_STORE_CONFIG["collection_name"]
        )

        self.vector_retriever = self.vector_store.as_retriever()
        self._init_bm25_retriever()

        # 使用配置中的权重
        weights = [
            ENHANCED_RETRIEVAL_CONFIG["vector_weight"],
            ENHANCED_RETRIEVAL_CONFIG["bm25_weight"]
        ] if self.bm25_retriever else [1.0]

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever] if self.bm25_retriever else [self.vector_retriever],
            weights=weights
        )

        # 使用配置中的过滤阈值
        self.compressor = EmbeddingsFilter(
            embeddings=self.embeddings,
            similarity_threshold=ENHANCED_RETRIEVAL_CONFIG["filter_threshold"]
        )
        
        self.compression_retriever = ContextualCompressionRetriever(
            base_retriever=self.ensemble_retriever,
            base_compressor=self.compressor
        )

    def _initialize_reranker(self):
        """初始化重排序器（使用 sentence-transformers 实现）"""
        self.reranker = None
        if RETRIEVAL_CONFIG.get("use_reranking", True):
            try:
                from sentence_transformers import CrossEncoder
                
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                print("✅ 重排序器初始化成功")
            except ImportError:
                print("⚠️ sentence-transformers 未安装 CrossEncoder 功能，将跳过重排序")
            except Exception as e:
                print(f"⚠️ 重排序器初始化失败，将跳过重排序: {str(e)}")

    def _rerank_results(self, query: str, documents: List) -> List:
        """对检索结果进行重排序"""
        if not self.reranker or len(documents) <= 1:
            return documents

        try:
            pairs = [(query, doc.page_content) for doc in documents]
            scores = self.reranker.predict(pairs)
            
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            rerank_top_n = RETRIEVAL_CONFIG.get("rerank_top_n", 5)
            return [doc for doc, score in scored_docs[:rerank_top_n]]
        except Exception as e:
            print(f"⚠️ 重排序失败，使用原始排序: {str(e)}")
            return documents

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
        """执行检索，支持缓存、多查询和多种检索策略"""
        start_time = time.time()
        results: List[Dict[str, Any]] = []
        
        # 检查缓存
        cached_results = self._get_from_cache(query, top_k)
        if cached_results:
            logger.info(f"检索完成(缓存): 查询='{query[:50]}...' 结果数={len(cached_results)} 耗时=0s")
            return cached_results
        
        try:
            total_chunk = self.vector_store._collection.count()
        except Exception as e:
            logger.debug(f"获取向量库数量失败，使用备用方法: {str(e)}")
            all_data = self.vector_store.get()
            total_chunk = len(all_data.get("ids", []))
        
        logger.debug(f"向量库共有 {total_chunk} 个分片")
        
        # 使用配置中的查询扩展倍数
        fetch_multiplier = RETRIEVAL_CONFIG.get("fetch_multiplier", 2)
        fetch_k = min(top_k * fetch_multiplier, max(total_chunk, 1))

        try:
            # 生成查询变体
            query_variants = self._generate_query_variants(query)
            logger.debug(f"使用 {len(query_variants)} 个查询变体")
            
            # 分别使用向量检索和BM25检索，收集多个结果列表用于RRF融合
            results_list = []
            
            # 添加向量检索结果
            self.vector_retriever.search_kwargs = {"k": fetch_k}
            vector_results = self.vector_retriever.invoke(query)
            results_list.append(vector_results)
            logger.debug(f"向量检索获取 {len(vector_results)} 条结果")
            
            # 添加BM25检索结果（如果可用）
            if self.bm25_retriever:
                self.bm25_retriever.k = fetch_k
                bm25_results = self.bm25_retriever.invoke(query)
                results_list.append(bm25_results)
                logger.debug(f"BM25检索获取 {len(bm25_results)} 条结果")
            
            # 如果启用多查询，添加变体查询结果
            if len(query_variants) > 1:
                for variant in query_variants[1:]:
                    self.vector_retriever.search_kwargs = {"k": fetch_k}
                    variant_results = self.vector_retriever.invoke(variant)
                    results_list.append(variant_results)
                    logger.debug(f"查询变体 '{variant[:20]}...' 获取 {len(variant_results)} 条结果")
            
            # 使用RRF融合多个检索结果
            if RETRIEVAL_CONFIG.get("use_rrf", False) and len(results_list) > 1:
                rrf_k = RETRIEVAL_CONFIG.get("rrf_k", 60)
                fused_results = self._rrf_fusion(results_list, rrf_k)
                logger.debug(f"RRF融合后保留 {len(fused_results)} 条结果")
            else:
                # 简单合并
                all_results = []
                for results in results_list:
                    all_results.extend(results)
                fused_results = all_results
            
            # 去重
            seen_contents = set()
            unique_results = []
            for doc in fused_results:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    unique_results.append(doc)
            
            logger.debug(f"去重后保留 {len(unique_results)} 条结果")

            # 标题/文件名加权提升
            unique_results = self._title_boost(query, unique_results)
            logger.debug("已执行标题加权")

            # 重排序（使用原始查询进行重排序）
            if RETRIEVAL_CONFIG.get("use_reranking", True) and len(unique_results) > 1:
                unique_results = self._rerank_results(query, unique_results)
                logger.debug("已执行重排序")

            # 生成最终结果
            for rank, doc in enumerate(unique_results[:top_k], start=1):
                # 使用改进的分数计算方法
                score = self._calculate_score(doc, rank, top_k)
                
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

            # 添加到缓存
            if results:
                self._add_to_cache(query, top_k, results)

        except Exception as e:
            logger.error(f"增强检索异常: {str(e)}", exc_info=True)
            try:
                safe_k = min(top_k, max(total_chunk, 1))
                logger.warning(f"降级到基础向量检索, k={safe_k}")
                self.vector_retriever.search_kwargs = {"k": safe_k}
                vector_results = self.vector_retriever.invoke(query)
                
                # 重排序（如果可用）
                if RETRIEVAL_CONFIG.get("use_reranking", True) and len(vector_results) > 1:
                    vector_results = self._rerank_results(query, vector_results)
                
                for rank, doc in enumerate(vector_results[:top_k], start=1):
                    score = self._calculate_score(doc, rank, top_k)
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
                logger.error(f"降级检索也失败: {str(ee)}", exc_info=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"检索完成: 查询='{query[:50]}...' 结果数={len(results)} 耗时={elapsed_time:.3f}s")
        
        return results

    def refresh_bm25(self):
        """刷新BM25检索器（当有新文档入库时调用）"""
        logger.info("刷新BM25检索器...")
        self._init_bm25_retriever()
        
        # 清空缓存（文档发生变化，缓存失效）
        self._cache = {}
        logger.debug("已清空检索缓存")
        
        if self.bm25_retriever:
            weights = [
                ENHANCED_RETRIEVAL_CONFIG["vector_weight"],
                ENHANCED_RETRIEVAL_CONFIG["bm25_weight"]
            ]
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.vector_retriever, self.bm25_retriever],
                weights=weights
            )
            # 同步刷新压缩检索器
            self.compression_retriever = ContextualCompressionRetriever(
                base_retriever=self.ensemble_retriever,
                base_compressor=self.compressor
            )
            logger.info("BM25检索器刷新成功")
        else:
            logger.info("BM25检索器初始化失败（向量库为空）")