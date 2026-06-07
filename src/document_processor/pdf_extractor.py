import os
import re
import hashlib
from typing import List, Dict, Optional, Any
import pdfplumber
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from .models import ProcessedDocument, DocumentMetadata, DocumentSection
from .text_cleaner import TextCleaner
from .config import PDF_EXTRACT_CONFIG


class PDFExtractor:
    """PDF提取器，负责从PDF文件中提取文本、元数据和结构"""

    def __init__(self, extract_config: dict = None, clean_config: dict = None):
        self.extract_config = extract_config or PDF_EXTRACT_CONFIG
        self.text_cleaner = TextCleaner(clean_config)

    def extract(self, file_path: str) -> ProcessedDocument:
        """
        提取PDF文件的所有内容
        Args:
            file_path: PDF文件的完整路径
        Returns:
            ProcessedDocument对象
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"不支持的文件类型: {file_path}")

        doc = ProcessedDocument()
        doc.file_name = os.path.basename(file_path)
        doc.file_path = file_path
        doc.file_size = os.path.getsize(file_path)

        try:
            # 检查PDF是否加密
            self._check_encryption(file_path)

            # 第一步：使用pdfplumber提取原始文本和表格
            with pdfplumber.open(file_path) as pdf:
                raw_text = ""
                tables = []

                for page_num, page in enumerate(pdf.pages, 1):
                    # 提取文本
                    try:
                        page_text = page.extract_text(
                            x_tolerance=self.extract_config["char_margin"],
                            y_tolerance=self.extract_config["line_margin"]
                        )
                        if page_text:
                            raw_text += f"\n--- Page {page_num} ---\n{page_text}"
                    except Exception as e:
                        raw_text += f"\n--- Page {page_num} ---\n[文本提取失败: {str(e)}]"

                    # 提取表格
                    if self.extract_config["extract_tables"]:
                        try:
                            page_tables = page.extract_tables()
                            for table in page_tables:
                                if table and len(table) > 0:
                                    tables.append({
                                        "page_number": page_num,
                                        "data": table,
                                        "markdown": self._table_to_markdown(table)
                                    })
                        except Exception:
                            pass  # 表格提取失败不影响整体处理

                doc.raw_text = raw_text
                doc.tables = tables

            # 第二步：使用PyPDF2提取基本元数据
            metadata = self._extract_pdf_metadata(file_path)
            doc.metadata = metadata

            # 第三步：清洗文本
            doc.full_text = self.text_cleaner.clean(doc.raw_text)

            # 第四步：尝试从文本中提取更准确的元数据（不覆盖已有值）
            self._extract_metadata_from_text(doc)

            # 第五步：尝试识别文档结构（章节）
            doc.sections = self._extract_sections(doc.full_text)

            doc.processing_status = "success"

        except Exception as e:
            doc.processing_status = "failed"
            doc.error_message = str(e)
            raise

        return doc

    def _check_encryption(self, file_path: str) -> None:
        """检查PDF是否加密"""
        try:
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                if reader.is_encrypted:
                    # 尝试用空密码解密
                    if not reader.decrypt(''):
                        raise ValueError("PDF文件已加密，需要密码才能访问")
        except PdfReadError as e:
            raise ValueError(f"PDF文件读取错误: {str(e)}")

    def _extract_pdf_metadata(self, file_path: str) -> DocumentMetadata:
        """从PDF文件中提取元数据"""
        metadata = DocumentMetadata()

        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                pdf_info = pdf_reader.metadata

                if pdf_info:
                    if pdf_info.title:
                        metadata.title = pdf_info.title
                    if pdf_info.author:
                        metadata.authors = [a.strip() for a in pdf_info.author.split(',')]
                    if pdf_info.subject:
                        metadata.abstract = pdf_info.subject
                    if pdf_info.creator:
                        metadata.publisher = pdf_info.creator
        except Exception:
            pass  # 元数据提取失败不影响整体处理

        return metadata

    def _extract_metadata_from_text(self, doc: ProcessedDocument) -> None:
        """从文本中提取更准确的元数据（启发式方法，不覆盖已有值）"""
        text = doc.full_text
        lines = text.split('\n')

        # 提取标题（通常是前几行中最长的行）
        if not doc.metadata.title and len(lines) > 0:
            # 前10行中找最长的非空行作为标题候选
            title_candidates = [line.strip() for line in lines[:10] if len(line.strip()) > 10]
            if title_candidates:
                doc.metadata.title = max(title_candidates, key=len)

        # 提取DOI（如果没有）
        if not doc.metadata.doi:
            doi_pattern = re.compile(r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b', re.IGNORECASE)
            doi_match = doi_pattern.search(text)
            if doi_match:
                doc.metadata.doi = doi_match.group(0)

        # 提取关键词（如果没有）
        if not doc.metadata.keywords:
            keywords_pattern = re.compile(r'(?:关键词|Keywords?|Key\s*Words?)\s*[:：]?\s*(.*?)(?:\n\n|\n(?:1\.|Introduction|摘要|Abstract)|\Z)', re.IGNORECASE | re.DOTALL)
            keywords_match = keywords_pattern.search(text)
            if keywords_match:
                keywords_str = keywords_match.group(1).strip()
                doc.metadata.keywords = [k.strip() for k in re.split(r'[,;，；]', keywords_str) if k.strip()]

        # 提取摘要（如果没有）
        if not doc.metadata.abstract:
            # 支持中英文摘要
            abstract_pattern = re.compile(
                r'(?:摘要|Abstract)\s*[:：]?\s*(.*?)(?:\n\n|\n(?:1\.|Introduction|关键词|Keywords)|\Z)',
                re.IGNORECASE | re.DOTALL
            )
            abstract_match = abstract_pattern.search(text)
            if abstract_match:
                doc.metadata.abstract = abstract_match.group(1).strip()

    def _extract_sections(self, text: str) -> List[DocumentSection]:
        """从文本中提取章节结构（支持中英文格式）"""
        sections = []

        # 匹配多种章节格式
        patterns = [
            # 英文格式: 1. Introduction, 2. Related Work
            re.compile(r'^(\d+)\.\s+([A-Z][\w\s]+)$', re.MULTILINE),
            # 英文格式无点号: 1 Introduction, 2 Related Work
            re.compile(r'^(\d+)\s+([A-Z][A-Za-z\s]+)$', re.MULTILINE),
            # 中文格式: 一、引言, 二、相关工作
            re.compile(r'^([一二三四五六七八九十]+)[、．.]\s*(.+)$', re.MULTILINE),
            # 中文格式: 1. 引言, 2. 相关工作
            re.compile(r'^(\d+)[、．.]\s*(.+)$', re.MULTILINE),
        ]

        # 找到所有匹配
        all_matches = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                all_matches.append((match.start(), match.end(), match.group(0)))

        # 按位置排序
        all_matches.sort(key=lambda x: x[0])

        # 去重（避免同一位置被多个模式匹配）
        unique_matches = []
        last_end = -1
        for start, end, content in all_matches:
            if start > last_end:
                unique_matches.append((start, end, content))
                last_end = end

        # 提取章节内容
        for i, (start, end, title) in enumerate(unique_matches):
            title = title.strip()

            # 确定章节结束位置
            if i < len(unique_matches) - 1:
                next_start = unique_matches[i + 1][0]
            else:
                next_start = len(text)

            section_content = text[end:next_start].strip()

            # 只保留有实际内容的章节
            if len(section_content) > 50:
                sections.append(DocumentSection(
                    title=title,
                    content=section_content,
                    level=1
                ))

        return sections

    def _table_to_markdown(self, table: List[List[Optional[str]]]) -> str:
        """将表格数据转换为Markdown格式"""
        if not table or len(table) == 0:
            return ""

        # 处理空值
        processed_table = []
        for row in table:
            processed_row = [str(cell).strip() if cell else "" for cell in row]
            processed_table.append(processed_row)

        # 确保所有行的列数一致
        max_cols = max(len(row) for row in processed_table)
        for row in processed_table:
            while len(row) < max_cols:
                row.append("")

        # 生成Markdown表格
        markdown = "| " + " | ".join(processed_table[0]) + " |\n"
        markdown += "| " + " | ".join(["---"] * max_cols) + " |\n"

        for row in processed_table[1:]:
            markdown += "| " + " | ".join(row) + " |\n"

        return markdown

    def get_file_hash(self, file_path: str) -> str:
        """计算文件的MD5哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
