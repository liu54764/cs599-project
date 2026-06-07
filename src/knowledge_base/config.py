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

# 文档分块配置
CHUNKING_CONFIG = {
    "chunk_size": 512,        # 每个块的字符数
    "chunk_overlap": 100,     # 块之间的重叠字符数
    "separators": ["\n\n", "\n", ". ", " ", ""],  # 分块分隔符优先级
}

# 向量数据库配置
VECTOR_STORE_CONFIG = {
    "persist_directory": str(BASE_DIR / "data" / "vector_db"),  # 向量数据库持久化目录
    "collection_name": "academic_papers",  # 集合名称
}

# 检索配置
RETRIEVAL_CONFIG = {
    "top_k": 5,              # 默认返回最相关的5个块
    "score_threshold": 0.5,  # 相似度阈值，低于此值的结果将被过滤
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
