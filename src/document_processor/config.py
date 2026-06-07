# PDF提取配置
PDF_EXTRACT_CONFIG = {
    "extract_images": False,        # 是否提取图片（默认关闭，节省资源）
    "extract_tables": True,         # 是否提取表格
    "table_format": "markdown",     # 表格输出格式
    "line_overlap_threshold": 0.5,  # 行重叠阈值
    "char_margin": 0.5,             # 字符间距阈值
    "line_margin": 0.5,             # 行间距阈值
}

# 文本清洗配置
TEXT_CLEAN_CONFIG = {
    "remove_header_footer": True,   # 移除页眉页脚
    "remove_page_numbers": True,    # 移除页码
    "remove_blank_lines": True,     # 移除空行
    "remove_extra_spaces": True,    # 移除多余空格
    "fix_hyphenation": True,        # 修复连字符换行
    "remove_citations": False,      # 是否移除引用标记（默认保留）
}

# 支持的文件类型
SUPPORTED_FILE_TYPES = [".pdf"]

# 元数据提取优先级
METADATA_PRIORITY = ["grobid", "pdf_info", "text_heuristic"]