"""基于 LangChain LCEL 的现代问答引擎"""
from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document

from knowledge_base.config import VECTOR_STORE_CONFIG
from knowledge_base.embedding_service import EmbeddingService


class LCELQAEngine:
    """基于 LangChain LCEL 的问答引擎"""

    def __init__(self):
        self._initialize()

    def _initialize(self):
        """初始化 LCEL 组件"""
        # 统一使用项目的 EmbeddingService
        self.embedding_service = EmbeddingService()

        # 创建兼容 LangChain 的嵌入函数包装器
        class EmbeddingsWrapper:
            def __init__(self, service):
                self.service = service

            def embed_documents(self, texts):
                return self.service.embed_texts(texts)

            def embed_query(self, text):
                return self.service.embed_text(text)

        self.embeddings = EmbeddingsWrapper(self.embedding_service)

        # 初始化向量存储
        from langchain_community.vectorstores import Chroma
        self.vector_store = Chroma(
            persist_directory=VECTOR_STORE_CONFIG["persist_directory"],
            embedding_function=self.embeddings,
            collection_name=VECTOR_STORE_CONFIG["collection_name"]
        )

        # 创建检索器
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 8}
        )

        # 初始化 LLM
        from .llm_client import LLMClient
        self.llm_client = LLMClient()

        # 创建兼容 LCEL 的 LLM 包装器
        class LLMWrapper:
            def __init__(self, client):
                self.client = client

            def invoke(self, prompt):
                # 处理不同类型的输入
                if hasattr(prompt, 'content'):
                    text = prompt.content
                elif isinstance(prompt, str):
                    text = prompt
                else:
                    text = str(prompt)
                return self.client.complete(text)

            async def ainvoke(self, prompt):
                return self.invoke(prompt)

        self.llm = LLMWrapper(self.llm_client)

        # 构建 LCEL 链
        self._build_lcel_chains()

    def _build_lcel_chains(self):
        """构建 LCEL 链"""
        # 格式化文档函数
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 带记忆的 RAG 链（多轮对话）
        from langchain.memory import ConversationBufferMemory
        from langchain_core.runnables import RunnableLambda

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        memory_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位专业的学术助手。请基于提供的上下文文档和对话历史，回答用户的问题。

上下文：
{context}

对话历史：
{chat_history}

回答要求：
1. 严格基于上下文内容进行回答
2. 如果上下文没有相关信息，请说"知识库中未找到相关内容"
3. 回答要准确、简洁，使用自然语言
4. 如果问题是闲聊类（如问候、天气等），可以直接回答，不必局限于上下文
5. 注意引用对话历史中的信息"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}")
        ])

        def load_memory(_):
            return self.memory.load_memory_variables({})["chat_history"]

        def save_memory(inputs, outputs):
            self.memory.save_context(
                {"input": inputs["question"]},
                {"output": outputs}
            )
            return outputs

        self.conversational_rag_chain = (
            RunnableParallel({
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": RunnableLambda(load_memory)
            })
            | memory_prompt
            | self.llm
            | StrOutputParser()
            | RunnableLambda(save_memory)
        )



    def answer_with_memory(self, query: str, chat_history: Optional[List] = None) -> Dict[str, Any]:
        """
        带记忆的问答（支持多轮对话）

        Args:
            query: 用户问题
            chat_history: 可选的对话历史列表，格式为 [(question1, answer1), (question2, answer2)]

        Returns:
            包含回答和来源的字典
        """
        # 加载历史对话到记忆中
        if chat_history:
            self.clear_memory()
            for q, a in chat_history:
                self.memory.save_context({"input": q}, {"output": a})

        # 执行问答
        answer = self.conversational_rag_chain.invoke(query)

        # 获取检索到的文档
        sources = []
        try:
            docs = self.retriever.invoke(query)
            for doc in docs:
                distance = doc.metadata.get("distance", 1.0)
                similarity = max(0, 1.0 - distance)
                sources.append({
                    "file_name": doc.metadata.get("file_name", "unknown"),
                    "content": doc.page_content[:200] + "...",
                    "score": similarity
                })
        except Exception:
            pass

        return {
            "answer": answer,
            "source_documents": sources,
            "retrieval_count": len(sources),
            "question_type": "knowledge_with_retrieval" if sources else "knowledge_internal"
        }

    def answer_stream(self, query: str) -> Any:
        """
        流式问答（单轮）

        Args:
            query: 用户问题

        Yields:
            回答片段
        """
        # 先获取上下文
        docs = self.retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""你是一位专业的学术助手。请基于提供的上下文文档，回答用户的问题。

上下文：
{context}

用户问题：
{query}

回答要求：
1. 严格基于上下文内容进行回答
2. 如果上下文没有相关信息，请说"知识库中未找到相关内容"
3. 回答要准确、简洁，使用自然语言"""

        # 使用流式生成
        for chunk in self.llm_client.stream_complete(prompt):
            yield chunk

    def clear_memory(self):
        """清空对话记忆"""
        self.memory.clear()

    def get_memory_summary(self) -> str:
        """获取对话记忆摘要"""
        return str(self.memory.chat_memory)