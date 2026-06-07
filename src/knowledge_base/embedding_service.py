import os
import threading
from typing import List, Optional
from langchain_core.embeddings import Embeddings
import ollama


class EmbeddingService(Embeddings):
    """统一的嵌入生成服务，使用Ollama本地运行，兼容LangChain规范"""

    _instance = None
    _initialized = False
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def _initialize(self):
        if self._initialized:
            return
        try:
            print(f"正在初始化Ollama嵌入模型: all-minilm:l6-v2...")
            test_embedding = ollama.embed(
                model="all-minilm:l6-v2",
                input="test"
            )["embeddings"][0]
            self.embedding_dim = len(test_embedding)
            self._initialized = True
            print(f"✅ Ollama嵌入模型初始化完成，维度: {self.embedding_dim}")
        except Exception as e:
            print(f"❌ Ollama嵌入服务初始化失败: {str(e)}")
            print("💡 请确保Ollama桌面版已启动（右下角有图标）")
            raise RuntimeError(f"无法初始化嵌入服务: {str(e)}")

    def __init__(self):
        if not self._initialized:
            self._initialize()

    def embed_text(self, text: str) -> List[float]:
        if not self._initialized:
            raise RuntimeError("嵌入服务未初始化")
        try:
            return ollama.embed(
                model="all-minilm:l6-v2",
                input=text
            )["embeddings"][0]
        except Exception as e:
            raise RuntimeError(f"生成嵌入失败: {str(e)}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not self._initialized:
            raise RuntimeError("嵌入服务未初始化")
        if not texts:
            return []
        try:
            return ollama.embed(
                model="all-minilm:l6-v2",
                input=texts
            )["embeddings"]
        except Exception as e:
            raise RuntimeError(f"批量生成嵌入失败: {str(e)}")

    def embed_query(self, text: str) -> List[float]:
        return self.embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts)

    @classmethod
    def is_initialized(cls) -> bool:
        return cls._initialized