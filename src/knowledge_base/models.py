from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import uuid

@dataclass
class DocumentChunk:
    """文档块模型，代表向量化的最小单位"""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""  # 关联的原始文献ID
    file_name: str = ""    # 原始文件名
    chunk_index: int = 0   # 在文档中的块索引
    content: str = ""      # 块的文本内容
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    embedding: Optional[List[float]] = None  # 向量嵌入

@dataclass
class RetrievalResult:
    """检索结果模型"""
    chunk: DocumentChunk
    score: float  # 相似度分数（0-1，越高越相关）
    rank: int     # 排名