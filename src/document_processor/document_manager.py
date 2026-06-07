import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from .models import ProcessedDocument, DocumentMetadata, DocumentSection
from .pdf_extractor import PDFExtractor
from .config import SUPPORTED_FILE_TYPES


class DocumentManager:
    """文献管理器，负责批量处理和管理文献文件，支持数据持久化"""

    def __init__(self, storage_dir: str = "./data/documents"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)

        # 元数据存储目录
        self.metadata_dir = self.storage_dir / ".metadata"
        self.metadata_dir.mkdir(exist_ok=True)

        self.pdf_extractor = PDFExtractor()
        self.processed_documents: Dict[str, ProcessedDocument] = {}

        # 服务启动时自动加载已有的文献数据
        self._load_existing_documents()

    def upload_document(self, file_path: str, copy_to_storage: bool = True) -> ProcessedDocument:
        """
        上传并处理单个文献
        Args:
            file_path: 源文件路径
            copy_to_storage: 是否将文件复制到存储目录
        Returns:
            处理后的文档对象
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in SUPPORTED_FILE_TYPES:
            raise ValueError(f"不支持的文件类型: {file_ext}，支持的类型: {SUPPORTED_FILE_TYPES}")

        # 复制文件到存储目录
        if copy_to_storage:
            dest_path = self.storage_dir / os.path.basename(file_path)
            shutil.copy2(file_path, dest_path)
            process_path = str(dest_path)
        else:
            process_path = file_path

        # 提取文档内容
        doc = self.pdf_extractor.extract(process_path)

        # 保存到内存和磁盘
        self.processed_documents[doc.document_id] = doc
        self._save_document_metadata(doc)

        return doc

    def batch_upload(self, directory_path: str, recursive: bool = False) -> List[ProcessedDocument]:
        """
        批量上传目录中的所有文献
        Args:
            directory_path: 目录路径
            recursive: 是否递归处理子目录
        Returns:
            处理后的文档列表
        """
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"不是有效的目录: {directory_path}")

        processed_docs = []
        failed_docs = []

        # 遍历目录中的所有文件
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in SUPPORTED_FILE_TYPES:
                    file_path = os.path.join(root, file)
                    try:
                        doc = self.upload_document(file_path)
                        processed_docs.append(doc)
                        print(f"成功处理: {file}")
                    except Exception as e:
                        failed_docs.append({"file": file, "error": str(e)})
                        print(f"处理失败: {file}，错误: {str(e)}")

            if not recursive:
                break

        print(f"\n批量处理完成: 成功 {len(processed_docs)} 个，失败 {len(failed_docs)} 个")
        return processed_docs

    def get_document(self, document_id: str) -> Optional[ProcessedDocument]:
        """根据文档ID获取处理后的文档"""
        return self.processed_documents.get(document_id)

    def list_documents(self) -> List[ProcessedDocument]:
        """列出所有已处理的文档"""
        return list(self.processed_documents.values())

    def delete_document(self, document_id: str) -> bool:
        """删除指定文献"""
        if document_id in self.processed_documents:
            doc = self.processed_documents[document_id]

            # 删除PDF文件
            if os.path.exists(doc.file_path) and self.storage_dir in Path(doc.file_path).parents:
                os.remove(doc.file_path)

            # 删除元数据文件
            metadata_file = self.metadata_dir / f"{document_id}.json"
            if metadata_file.exists():
                os.remove(metadata_file)

            # 从内存中删除
            del self.processed_documents[document_id]

            return True
        return False

    def _save_document_metadata(self, doc: ProcessedDocument) -> None:
        """将文献元数据和处理结果保存到磁盘"""
        metadata_file = self.metadata_dir / f"{doc.document_id}.json"

        # 将dataclass转换为可序列化的字典
        doc_dict = {
            "document_id": doc.document_id,
            "file_name": doc.file_name,
            "file_path": doc.file_path,
            "file_size": doc.file_size,
            "upload_time": doc.upload_time.isoformat(),
            "metadata": {
                "title": doc.metadata.title,
                "authors": doc.metadata.authors,
                "publication_date": doc.metadata.publication_date,
                "journal": doc.metadata.journal,
                "volume": doc.metadata.volume,
                "issue": doc.metadata.issue,
                "pages": doc.metadata.pages,
                "doi": doc.metadata.doi,
                "abstract": doc.metadata.abstract,
                "keywords": doc.metadata.keywords,
                "publisher": doc.metadata.publisher,
                "language": doc.metadata.language
            },
            "full_text": doc.full_text,
            "sections": [
                {
                    "title": s.title,
                    "content": s.content,
                    "level": s.level,
                    "page_number": s.page_number
                } for s in doc.sections
            ],
            "tables": doc.tables,
            "raw_text": doc.raw_text,
            "processing_status": doc.processing_status,
            "error_message": doc.error_message
        }

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(doc_dict, f, ensure_ascii=False, indent=2)

    def _load_existing_documents(self) -> None:
        """服务启动时自动加载已有的文献数据"""
        print("正在加载已有的文献数据...")

        # 扫描元数据目录
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    doc_dict = json.load(f)

                # 重建ProcessedDocument对象
                metadata = DocumentMetadata(**doc_dict["metadata"])
                sections = [DocumentSection(**s) for s in doc_dict["sections"]]

                doc = ProcessedDocument(
                    document_id=doc_dict["document_id"],
                    file_name=doc_dict["file_name"],
                    file_path=doc_dict["file_path"],
                    file_size=doc_dict["file_size"],
                    upload_time=datetime.fromisoformat(doc_dict["upload_time"]),
                    metadata=metadata,
                    full_text=doc_dict["full_text"],
                    sections=sections,
                    tables=doc_dict["tables"],
                    raw_text=doc_dict["raw_text"],
                    processing_status=doc_dict["processing_status"],
                    error_message=doc_dict["error_message"]
                )

                # 检查PDF文件是否还存在
                if os.path.exists(doc.file_path):
                    self.processed_documents[doc.document_id] = doc
                else:
                    # 如果PDF文件已丢失，删除对应的元数据
                    os.remove(metadata_file)
                    print(f"警告: 文献 {doc.file_name} 的PDF文件已丢失，已删除元数据")

            except Exception as e:
                print(f"加载元数据文件 {metadata_file.name} 失败: {str(e)}")
                # 损坏的元数据文件可以选择删除或保留
                # os.remove(metadata_file)

        print(f"成功加载 {len(self.processed_documents)} 篇文献")