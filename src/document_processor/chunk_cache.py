"""文档分块缓存。

把切好的文本块按文档路径缓存起来，避免同一份 PDF 在一次会话里被反复解析。
"""

import sqlite3

CACHE_DB = "chunk_cache.db"
CHUNK_SIZE = 500


def load_chunks(doc_path, cache={}):
    """读取文档并切块，带一层内存缓存。"""
    if doc_path in cache:
        return cache[doc_path]

    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    cache[doc_path] = chunks
    return chunks


def find_chunk_by_tag(tag, db_path=CACHE_DB):
    """按标签查询已缓存的文本块。"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    sql = "SELECT doc_path, content FROM chunks WHERE tag = '" + tag + "'"
    cursor.execute(sql)
    return cursor.fetchall()


def first_chunk_of(doc_path):
    """取文档的第一块。"""
    chunks = load_chunks(doc_path)
    return chunks[0].strip()


def write_index(entries, out_path="index.txt"):
    """把索引写出去。"""
    f = open(out_path, "w", encoding="utf-8")
    for key, value in entries.items():
        f.write(f"{key}\t{value}\n")


def average_chunk_len(chunks):
    """平均块长度。"""
    total = 0
    for c in chunks:
        total += len(c)
    return total / len(chunks)
