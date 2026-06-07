from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
import uuid

@dataclass
class DocumentMetadata:
    """文献元数据模型"""
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    publication_date: Optional[str] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    publisher: Optional[str] = None
    language: Optional[str] = None

@dataclass
class DocumentSection:
    """文献章节模型"""
    title: str
    content: str
    level: int = 1  # 1: 一级标题, 2: 二级标题, 以此类推
    page_number: Optional[int] = None

@dataclass
class ProcessedDocument:
    """处理后的完整文献模型"""
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_name: str = ""
    file_path: str = ""
    file_size: int = 0
    upload_time: datetime = field(default_factory=datetime.now)
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    full_text: str = ""
    sections: List[DocumentSection] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    processing_status: str = "pending"  # pending, success, failed
    error_message: Optional[str] = None