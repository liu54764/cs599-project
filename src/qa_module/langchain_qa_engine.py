"""基于 LangChain 的增强版问答引擎"""
from typing import List, Dict, Any, Optional
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from knowledge_base.config import VECTOR_STORE_CONFIG
from knowledge_base.embedding_service import EmbeddingService
import threading


class OllamaEmbeddingsWrapper:
    """兼容LangChain的Ollama嵌入函数包装器"""

    def __init__(self, embedding_service):
        self.embedding_service = embedding_service

    def embed_documents(self, texts):
        return self.embedding_service.embed_texts(texts)

    def embed_query(self, text):
        return self.embedding_service.embed_text(text)


class LLMClientWrapper:
    """兼容LangChain的LLM客户端包装器"""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.temperature = 0.1
        self.max_tokens = 2048

    def __call__(self, prompt: str) -> str:
        return self.llm_client.complete(prompt)


class LangChainQAEngine:
    """基于 LangChain 的问答引擎"""

    _instance = None
    _initialized = False
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._initialize()
                    self._initialized = True

    def _initialize(self):
        """初始化 LangChain 组件"""
        # 统一使用项目的EmbeddingService（解决嵌入模型不一致问题）
        self.embedding_service = EmbeddingService()
        self.embeddings = OllamaEmbeddingsWrapper(self.embedding_service)

        self.vector_store = Chroma(
            persist_directory=VECTOR_STORE_CONFIG["persist_directory"],
            embedding_function=self.embeddings,
            collection_name=VECTOR_STORE_CONFIG["collection_name"]
        )

        # 使用项目统一的LLMClient，而不是自己初始化ChatOpenAI
        from .llm_client import LLMClient
        self.llm_client = LLMClient()
        self.llm = LLMClientWrapper(self.llm_client)

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        self._build_chains()

    def _build_chains(self):
        """构建 LangChain 链"""
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 8}  # 移除score_threshold，ChromaDB的距离不是相似度
        )

        qa_prompt = PromptTemplate(
            template="""
你是一位专业的学术助手。请基于提供的上下文文档，回答用户的问题。

上下文：
{context}

用户问题：
{question}

回答要求：
1. 严格基于上下文内容进行回答
2. 如果上下文没有相关信息，请说"知识库中未找到相关内容"
3. 回答要准确、简洁，使用自然语言
4. 如果问题是闲聊类（如问候、天气等），可以直接回答，不必局限于上下文
            """,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_prompt}
        )

        self.conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )

    def answer_with_memory(self, query: str, chat_history: List = None) -> Dict[str, Any]:
        """
        带记忆的问答（支持多轮对话）

        Args:
            query: 用户问题
            chat_history: 可选的对话历史列表，格式为 [(question1, answer1), (question2, answer2)]

        Returns:
            包含回答和来源的字典
        """
        if chat_history:
            for q, a in chat_history:
                self.memory.chat_memory.add_user_message(q)
                self.memory.chat_memory.add_ai_message(a)

        result = self.conversational_chain({"question": query})

        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                # 转换ChromaDB的距离为相似度分数
                distance = doc.metadata.get("distance", 1.0)
                similarity = max(0, 1.0 - distance)

                sources.append({
                    "file_name": doc.metadata.get("file_name", "unknown"),
                    "content": doc.page_content[:200] + "...",
                    "score": similarity
                })

        return {
            "answer": result["answer"],
            "source_documents": sources,
            "retrieval_count": len(sources),
            "question_type": "knowledge_with_retrieval" if sources else "knowledge_internal"
        }

    def answer_with_sources(self, query: str) -> Dict[str, Any]:
        """
        不带记忆的问答，但返回来源信息

        Args:
            query: 用户问题

        Returns:
            包含回答和来源的字典
        """
        result = self.qa_chain({"question": query})

        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                distance = doc.metadata.get("distance", 1.0)
                similarity = max(0, 1.0 - distance)

                sources.append({
                    "file_name": doc.metadata.get("file_name", "unknown"),
                    "content": doc.page_content[:200] + "...",
                    "score": similarity
                })

        return {
            "answer": result["answer"],
            "sources": result.get("sources", ""),
            "source_documents": sources,
            "retrieval_count": len(sources)
        }

    def clear_memory(self):
        """清空对话记忆"""
        self.memory.clear()

    def get_memory_summary(self) -> str:
        """获取对话记忆摘要"""
        return str(self.memory.chat_memory)