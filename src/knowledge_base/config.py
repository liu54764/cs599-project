import os
from pathlib import Path

# 基础目录
BASE_DIR = Path(__file__).parent.parent.parent

# 嵌入模型配置
EMBEDDING_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",  # 轻量高效的开源嵌入模型
    "device": "cpu",                   # 可改为"cuda"如果有GPU
    "cache_folder": str(BASE_DIR / "src" / "models" / "embeddings"),  # 模型缓存目录
}

# 文档分块配置（优化后）
CHUNKING_CONFIG = {
    "chunk_size": 384,        # 每个块的字符数（优化：更细粒度提升检索精度）
    "chunk_overlap": 64,      # 块之间的重叠字符数（优化：减少冗余，保持上下文连贯）
    "separators": [           # 分块分隔符优先级（优化：增加中文分隔符）
        "\n\n\n", 
        "\n\n", 
        "\n", 
        "。", 
        "！", 
        "？", 
        ". ", 
        ".", 
        " ", 
        ""
    ],
    "add_start_index": True,   # 是否在块内容前添加起始位置索引
}

# 向量数据库配置
VECTOR_STORE_CONFIG = {
    "persist_directory": str(BASE_DIR / "data" / "vector_db"),  # 向量数据库持久化目录
    "collection_name": "academic_papers",  # 集合名称
}

# 检索配置（平衡速度和召回率）
RETRIEVAL_CONFIG = {
    "top_k": 5,              # 返回结果数量
    "score_threshold": 0.4,  # 提高阈值，过滤低相关结果
    "fetch_multiplier": 2,   # 降低扩展倍数
    "use_reranking": True,   # 启用重排序
    "rerank_top_n": 5,       # 重排序后保留数量
    "use_rrf": True,         # 使用RRF融合
    "rrf_k": 60,             # RRF参数
}

# 增强检索配置（EnsembleRetriever + 重排序）
ENHANCED_RETRIEVAL_CONFIG = {
    "vector_weight": 0.5,     # 平衡向量和BM25权重
    "bm25_weight": 0.5,      # 平衡权重
    "filter_threshold": 0.3,  # 提高过滤阈值
    "enable_multi_query": False,  # 禁用多查询，减少检索次数
    "max_query_variants": 3,      # 多查询变体数量
    "title_boost": 2.0,       # 标题/文件名匹配权重加成
}


def ensure_directories():
    """确保所有必要的目录都存在"""
    dirs = [
        EMBEDDING_CONFIG["cache_folder"],
        VECTOR_STORE_CONFIG["persist_directory"],
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


# 自动创建目录
ensure_directories()
