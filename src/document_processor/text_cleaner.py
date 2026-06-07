import re
from typing import List, Set
from .config import TEXT_CLEAN_CONFIG


class TextCleaner:
    """文本清洗器，负责处理PDF提取后的原始文本"""

    def __init__(self, config: dict = None):
        self.config = config or TEXT_CLEAN_CONFIG

    def clean(self, text: str) -> str:
        """执行完整的文本清洗流程"""
        if not text:
            return ""

        cleaned_text = text

        if self.config["fix_hyphenation"]:
            cleaned_text = self._fix_hyphenation(cleaned_text)

        if self.config["remove_header_footer"]:
            cleaned_text = self._remove_headers_footers(cleaned_text)

        if self.config["remove_page_numbers"]:
            cleaned_text = self._remove_page_numbers(cleaned_text)

        if self.config["remove_extra_spaces"]:
            cleaned_text = self._remove_extra_spaces(cleaned_text)

        if self.config["remove_blank_lines"]:
            cleaned_text = self._remove_blank_lines(cleaned_text)

        return cleaned_text.strip()

    def _fix_hyphenation(self, text: str) -> str:
        """修复单词被连字符分割在两行的情况"""
        # 匹配 "word-\nword" 或 "word-\r\nword" 格式
        hyphen_pattern = re.compile(r'(\w+)-\r?\n(\w+)')
        return hyphen_pattern.sub(r'\1\2', text)

    def _remove_headers_footers(self, text: str) -> str:
        """移除重复出现的页眉页脚"""
        lines = text.split('\n')
        if len(lines) < 10:
            return text

        # 查找页面分隔符（如 "--- Page N ---"）
        page_separators = []
        for i, line in enumerate(lines):
            if re.match(r'^---\s*Page\s+\d+\s*---$', line.strip()):
                page_separators.append(i)

        # 如果没有找到页面分隔符，使用简单方法
        if len(page_separators) < 2:
            return self._simple_header_footer_removal(text)

        # 收集每个页面的首行和尾行
        header_candidates: Set[str] = set()
        footer_candidates: Set[str] = set()

        for i, sep_idx in enumerate(page_separators):
            # 页面开始是分隔符后一行
            page_start = sep_idx + 1
            # 页面结束是下一个分隔符前一行，或文档末尾
            if i < len(page_separators) - 1:
                page_end = page_separators[i + 1] - 1
            else:
                page_end = len(lines) - 1

            # 收集前3行作为页眉候选
            for j in range(page_start, min(page_start + 3, page_end + 1)):
                stripped = lines[j].strip()
                if len(stripped) > 3:
                    header_candidates.add(stripped)

            # 收集后3行作为页脚候选
            for j in range(max(page_start, page_end - 2), page_end + 1):
                stripped = lines[j].strip()
                if len(stripped) > 3:
                    footer_candidates.add(stripped)

        # 找出在多个页面都出现的行
        header_counts = {}
        footer_counts = {}
        for i, sep_idx in enumerate(page_separators):
            page_start = sep_idx + 1
            if i < len(page_separators) - 1:
                page_end = page_separators[i + 1] - 1
            else:
                page_end = len(lines) - 1

            for j in range(page_start, min(page_start + 3, page_end + 1)):
                stripped = lines[j].strip()
                if stripped in header_candidates:
                    header_counts[stripped] = header_counts.get(stripped, 0) + 1

            for j in range(max(page_start, page_end - 2), page_end + 1):
                stripped = lines[j].strip()
                if stripped in footer_candidates:
                    footer_counts[stripped] = footer_counts.get(stripped, 0) + 1

        # 出现次数超过页面数一半的认为是页眉页脚
        num_pages = len(page_separators)
        headers_to_remove = {k for k, v in header_counts.items() if v >= num_pages / 2}
        footers_to_remove = {k for k, v in footer_counts.items() if v >= num_pages / 2}

        # 移除所有匹配的页眉页脚
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped not in headers_to_remove and stripped not in footers_to_remove:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _simple_header_footer_removal(self, text: str) -> str:
        """简单页眉页脚移除（当没有页面分隔符时使用）"""
        lines = text.split('\n')
        if len(lines) < 10:
            return text

        # 统计前10行和后10行中重复出现的行
        header_candidates = lines[:10]
        footer_candidates = lines[-10:]

        header_counts = {}
        footer_counts = {}

        for line in header_candidates:
            stripped = line.strip()
            if len(stripped) > 3:
                header_counts[stripped] = header_counts.get(stripped, 0) + 1

        for line in footer_candidates:
            stripped = line.strip()
            if len(stripped) > 3:
                footer_counts[stripped] = footer_counts.get(stripped, 0) + 1

        headers_to_remove = {k for k, v in header_counts.items() if v >= 2}
        footers_to_remove = {k for k, v in footer_counts.items() if v >= 2}

        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped not in headers_to_remove and stripped not in footers_to_remove:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _remove_page_numbers(self, text: str) -> str:
        """移除页码"""
        # 匹配单独一行的数字（页码）
        page_number_pattern = re.compile(r'^\s*\d+\s*$', re.MULTILINE)
        return page_number_pattern.sub('', text)

    def _remove_extra_spaces(self, text: str) -> str:
        """移除多余的空格和制表符"""
        # 替换多个空格为单个空格
        text = re.sub(r' +', ' ', text)
        # 替换多个制表符为单个空格
        text = re.sub(r'\t+', ' ', text)
        return text

    def _remove_blank_lines(self, text: str) -> str:
        """移除空行和只包含空白字符的行"""
        # 先移除只包含空白字符的行
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        return '\n'.join(non_empty_lines)
