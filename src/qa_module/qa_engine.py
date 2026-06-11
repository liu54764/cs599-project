"""问答引擎模块

该模块提供核心的问答功能，支持多种回答策略：
- 基于 LangGraph 的复杂工作流（默认）
- 基于 LangChain 的标准 RAG 链
- 自研的基础问答逻辑（回退方案）

核心组件：
- QAEngine: 主问答引擎，整合检索和生成流程
- 支持多轮对话记忆
- 智能问题分类（闲聊/知识类）
- 支持流式输出
"""
from typing import List, Dict, Any, Optional, Generator
import threading
from .config import QA_CONFIG
from .llm_client import LLMClient
from .prompt_templates import (
    ACADEMIC_QA_SYSTEM_PROMPT,
    PAPER_SUMMARY_SYSTEM_PROMPT,
    CROSS_PAPER_COMPARISON_SYSTEM_PROMPT,
    build_qa_prompt
)
from .question_classifier import QuestionClassifier
from knowledge_base import KnowledgeManager, RetrievalResult

# 尝试导入 LCEL QA 引擎（最高优先级）
try:
    from .lcel_qa_engine import LCELQAEngine

    LCEL_AVAILABLE = True
except ImportError:
    LCEL_AVAILABLE = False

# 尝试导入 LangGraph 工作流
try:
    from .langgraph_rag_workflow import LangGraphRAGWorkflow

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# 尝试导入多Agent分类器
try:
    from .multi_agent_classifier import MultiAgentQuestionClassifier

    MULTI_AGENT_AVAILABLE = True
except ImportError:
    MULTI_AGENT_AVAILABLE = False


class QAEngine:
    """问答引擎核心，整合检索和生成流程

    支持三种问答模式（优先级从高到低）：
    1. LCEL 模式（默认）：基于 LangChain Expression Language 的现代链式结构
    2. LangGraph 工作流模式：复杂工作流编排，支持条件分支和重试
    3. 自研模式：基础的问答逻辑，作为回退方案

    使用单例模式确保全局只有一个实例
    """

    _instance = None
    _initialized = False
    _lock = threading.Lock()

    def __new__(cls):
        """单例模式实现（线程安全版）"""
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
        """初始化组件"""
        self.llm_client = LLMClient()
        self.knowledge_manager = KnowledgeManager()
        self.config = QA_CONFIG
        self.langgraph_workflow = None
        self.langchain_engine = None

    def answer_question(self, query: str, chat_history: List = None,
                        use_langgraph: bool = True, use_langchain: bool = False) -> Dict[str, Any]:
        """
        回答用户的自然语言问题，支持多轮对话记忆

        智能策略：
        1. 判断问题类型（闲聊类/知识类）
        2. 如果是闲聊类，直接用LLM回答
        3. 如果是知识类，先检索知识库
        4. 如果知识库有结果，结合知识库内容回答
        5. 如果知识库无结果，用LLM内置知识回答
        6. 如果仍无法回答，返回"无法回答"

        Args:
            query: 用户问题
            chat_history: 对话历史，格式为 [(question1, answer1), (question2, answer2)]
            use_langgraph: 是否使用LangGraph工作流（默认启用）
            use_langchain: 是否使用LangChain引擎（优先级低于langgraph）
        Returns:
            包含回答和来源的字典，结构如下：
            {
                "answer": str,           # 回答内容
                "source_documents": list, # 来源文档列表
                "retrieval_count": int,  # 检索到的文档数量
                "question_type": str,    # 问题类型
                "confidence": float      # 回答置信度
            }
        """
        # 优先级：LCEL > LangGraph > 自研
        if LCEL_AVAILABLE:
            return self._answer_with_lcel(query, chat_history)
        elif use_langgraph and LANGGRAPH_AVAILABLE:
            return self._answer_with_langgraph(query, chat_history)

        # 自研模式（基础逻辑）
        return self._answer_with_basic_logic(query, chat_history)

    def answer_question_stream(self, query: str, chat_history: List = None) -> Generator[Dict[str, Any], None, None]:
        """
        流式回答用户的自然语言问题，支持多轮对话记忆

        Args:
            query: 用户问题
            chat_history: 对话历史，格式为 [(question1, answer1), (question2, answer2)]
        Yields:
            字典，包含流式回答块和来源信息
        """
        # 构建上下文感知的查询
        context_aware_query = self._build_context_aware_query(query, chat_history)

        # 判断问题类型（优先使用多Agent分类器）
        if MULTI_AGENT_AVAILABLE:
            try:
                classifier = MultiAgentQuestionClassifier()
                classification_result = classifier.classify(context_aware_query, chat_history)
                question_type = classification_result["classification"]

                if classification_result.get("final_answer"):
                    for chunk in self._stream_text(classification_result["final_answer"]):
                        yield {"chunk": chunk, "is_finished": False}
                    yield {
                        "chunk": "",
                        "is_finished": True,
                        "source_documents": classification_result.get("retrieval_results", []),
                        "retrieval_count": len(classification_result.get("retrieval_results", [])),
                        "question_type": question_type
                    }
                    return
            except Exception:
                question_type = QuestionClassifier.classify(context_aware_query)
        else:
            question_type = QuestionClassifier.classify(context_aware_query)

        # 闲聊类问题，直接用LLM回答
        if question_type == 'chat':
            prompt = self._build_chat_prompt(query, chat_history)
            for chunk in self.llm_client.stream_complete(
                prompt=prompt,
                system_prompt="""你是一个友好的聊天助手，擅长进行自然、有趣的对话。请用中文回答，保持口语化和亲切感。"""
            ):
                yield {"chunk": chunk, "is_finished": False}
            yield {
                "chunk": "",
                "is_finished": True,
                "source_documents": [],
                "retrieval_count": 0,
                "question_type": "chat"
            }
            return

        # 知识类问题，先检索知识库
        retrieval_results = self.knowledge_manager.search_knowledge_base(
            context_aware_query,
            top_k=self.config["retrieval_top_k"]
        )

        if retrieval_results:
            context_documents = self._process_retrieval_results(retrieval_results)
            prompt = build_qa_prompt(query, context_documents, chat_history)

            for chunk in self.llm_client.stream_complete(
                prompt=prompt,
                system_prompt=ACADEMIC_QA_SYSTEM_PROMPT
            ):
                yield {"chunk": chunk, "is_finished": False}

            yield {
                "chunk": "",
                "is_finished": True,
                "source_documents": context_documents,
                "retrieval_count": len(retrieval_results),
                "question_type": "knowledge_with_retrieval"
            }
        else:
            # 无检索结果，使用LLM内置知识回答
            prompt = self._build_internal_knowledge_prompt(query, chat_history)
            full_answer = ""

            for chunk in self.llm_client.stream_complete(
                prompt=prompt,
                system_prompt="""你是一位知识渊博的助手。请基于你的内置知识回答用户的问题。"""
            ):
                full_answer += chunk
                yield {"chunk": chunk, "is_finished": False}

            if "无法回答" in full_answer or "不知道" in full_answer or "抱歉" in full_answer:
                question_type = "knowledge_no_result"
            else:
                question_type = "knowledge_internal"

            yield {
                "chunk": "",
                "is_finished": True,
                "source_documents": [],
                "retrieval_count": 0,
                "question_type": question_type
            }

    def _stream_text(self, text: str, chunk_size: int = 50) -> Generator[str, None, None]:
        """将文本按块流式输出"""
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]

    def _build_chat_prompt(self, query: str, chat_history: List = None) -> str:
        """构建闲聊提示词"""
        history_context = ""
        if chat_history and len(chat_history) > 0:
            history_context = "对话历史（按时间顺序）：\n"
            for idx, (q, a) in enumerate(chat_history[-5:], 1):
                history_context += f"[{idx}] 用户：{q}\n"
                history_context += f"     助手：{a}\n"
            history_context += "\n"

        return f"""{history_context}请基于上述对话历史，回答用户当前的问题：\n\n用户问题：{query}\n\n重要提示：1. 如果用户的问题涉及之前对话中提到的内容，请仔细查看对话历史并引用正确的信息。2. 回答要准确、自然，符合上下文。"""

    def _build_internal_knowledge_prompt(self, query: str, chat_history: List = None) -> str:
        """构建内置知识回答提示词"""
        history_context = ""
        if chat_history and len(chat_history) > 0:
            history_context = "\n\n对话历史：\n"
            for q, a in chat_history[-3:]:
                history_context += f"用户：{q}\n助手：{a}\n"

        return f"""{history_context}知识库中未检索到与当前问题直接相关的文献内容，将基于内置知识回答。\n\n用户问题：{query}\n\n如果你的知识中也没有相关信息，请直接说："抱歉，我无法回答这个问题。"不要编造信息，不要猜测，保持回答真实可靠。"""

    def _answer_with_lcel(self, query: str, chat_history: List = None) -> Dict[str, Any]:
        """使用 LCEL 引擎回答问题（最高优先级）"""
        if not hasattr(self, 'lcel_engine') or self.lcel_engine is None:
            try:
                from .lcel_qa_engine import LCELQAEngine
                self.lcel_engine = LCELQAEngine()
            except Exception as e:
                print(f"LCEL引擎初始化失败，回退到LangGraph: {e}")
                if LANGGRAPH_AVAILABLE:
                    return self._answer_with_langgraph(query, chat_history)
                return self._answer_with_basic_logic(query, chat_history)

        try:
            # 如果有对话历史，使用带记忆的问答
            if chat_history:
                result = self.lcel_engine.answer_with_memory(query, chat_history)
            else:
                result = self.lcel_engine.answer_with_sources(query)

            return {
                "answer": result.get("answer", ""),
                "source_documents": result.get("source_documents", []),
                "retrieval_count": len(result.get("source_documents", [])),
                "question_type": result.get("question_type", "knowledge_with_retrieval"),
                "confidence": 0.8
            }
        except Exception as e:
            print(f"LCEL引擎执行失败，回退到LangGraph: {e}")
            if LANGGRAPH_AVAILABLE:
                return self._answer_with_langgraph(query, chat_history)
            return self._answer_with_basic_logic(query, chat_history)

    def _answer_with_basic_logic(self, query: str, chat_history: List = None) -> Dict[str, Any]:
        """使用自研基础逻辑回答问题"""
        # 构建上下文感知的查询（使用LLM重写，解决指代消解问题）
        context_aware_query = self._build_context_aware_query(query, chat_history)

        # 1. 判断问题类型（优先使用多Agent分类器）
        if MULTI_AGENT_AVAILABLE:
            try:
                classifier = MultiAgentQuestionClassifier()
                classification_result = classifier.classify(context_aware_query, chat_history)
                question_type = classification_result["classification"]
                confidence = classification_result["confidence"]
                needs_retrieval = classification_result["needs_retrieval"]

                print(f"【多Agent分类】问题类型: {question_type}, 置信度: {confidence:.2f}, 需要检索: {needs_retrieval}")
                for info in classification_result.get("agent_info", []):
                    print(f"  {info}")

                # 如果多Agent已经生成了回答，直接返回
                if classification_result.get("final_answer"):
                    return {
                        "answer": classification_result["final_answer"],
                        "source_documents": classification_result.get("retrieval_results", []),
                        "retrieval_count": len(classification_result.get("retrieval_results", [])),
                        "question_type": question_type,
                        "confidence": confidence
                    }
            except Exception as e:
                print(f"多Agent分类器失败，使用基础分类器: {e}")
                question_type = QuestionClassifier.classify(context_aware_query)
                confidence = 0.7
                needs_retrieval = question_type == 'knowledge'
        else:
            question_type = QuestionClassifier.classify(context_aware_query)
            confidence = 0.7
            needs_retrieval = question_type == 'knowledge'

        print(f"问题类型: {question_type}, 问题: {query}")

        # 2. 如果是闲聊类问题，直接用LLM回答，不检索知识库
        if question_type == 'chat':
            answer = self._answer_chat(query, chat_history)
            return {
                "answer": answer,
                "source_documents": [],
                "retrieval_count": 0,
                "question_type": "chat",
                "confidence": confidence
            }

        # 3. 知识类/技术类问题，先检索知识库
        retrieval_results = self.knowledge_manager.search_knowledge_base(
            context_aware_query,
            top_k=self.config["retrieval_top_k"]
        )

        # 4. 检查检索结果
        if retrieval_results:
            # 有检索结果，结合知识库回答
            return self._answer_with_knowledge(query, retrieval_results, chat_history)
        else:
            # 无检索结果，尝试用LLM内置知识回答
            return self._answer_with_internal_knowledge(query, chat_history)

    def _build_context_aware_query(self, query: str, chat_history: List = None) -> str:
        """构建上下文感知的查询（使用LLM重写，解决指代消解问题）"""
        if not chat_history or len(chat_history) == 0:
            return query

        # 只取最近3轮对话
        recent_history = chat_history[-3:]

        history_str = ""
        for q, a in recent_history:
            history_str += f"用户：{q}\n助手：{a}\n"

        rewrite_prompt = f"""
对话历史：
{history_str}

当前问题：{query}

请将当前问题重写为一个完整、独立的问题，使其不依赖于对话上下文也能被理解。
如果当前问题已经是独立的，直接返回原问题。
只返回重写后的问题，不要添加任何其他内容。
"""

        try:
            rewritten_query = self.llm_client.complete(rewrite_prompt)
            print(f"原始查询: {query}")
            print(f"重写后查询: {rewritten_query}")
            return rewritten_query.strip()
        except Exception as e:
            print(f"查询重写失败，使用原始查询: {e}")
            return query

    def _answer_chat(self, query: str, chat_history: List = None) -> str:
        """回答闲聊类问题"""
        history_context = ""
        if chat_history and len(chat_history) > 0:
            history_context = "对话历史（按时间顺序）：\n"
            for idx, (q, a) in enumerate(chat_history[-5:], 1):  # 取最近5轮
                history_context += f"[{idx}] 用户：{q}\n"
                history_context += f"     助手：{a}\n"
            history_context += "\n"

        chat_prompt = f"""
{history_context}

请基于上述对话历史，回答用户当前的问题：

用户问题：{query}

重要提示：
1. 如果用户的问题涉及之前对话中提到的内容（如"我之前说什么了？"、"它是什么？"等），请仔细查看对话历史并引用正确的信息。
2. 如果用户问"我之前说我有几个猫？"，请从对话历史中查找用户之前提到的猫的数量。
3. 回答要准确、自然，符合上下文。
"""

        system_prompt = """
你是一个友好的聊天助手，擅长进行自然、有趣的对话。
请仔细查看提供的对话历史，理解上下文后再回答问题。
如果用户的问题涉及之前提到的内容，请务必引用对话历史中的信息。
请用中文回答，保持口语化和亲切感。
"""

        return self.llm_client.complete(
            prompt=chat_prompt,
            system_prompt=system_prompt
        )

    def _answer_with_knowledge(self, query: str, retrieval_results: List[RetrievalResult], chat_history: List = None) -> \
    Dict[str, Any]:
        """结合知识库内容和内置知识回答问题"""
        # 处理检索结果，构建上下文
        context_documents = self._process_retrieval_results(retrieval_results)

        # 统一使用build_qa_prompt函数构建提示词
        prompt = build_qa_prompt(query, context_documents, chat_history)

        # 调用LLM生成回答，结合知识库和内置知识
        answer = self.llm_client.complete(
            prompt=prompt,
            system_prompt=ACADEMIC_QA_SYSTEM_PROMPT
        )

        return {
            "answer": answer,
            "source_documents": context_documents,
            "retrieval_count": len(retrieval_results),
            "question_type": "knowledge_with_retrieval",
            "confidence": 0.8
        }

    def _answer_with_internal_knowledge(self, query: str, chat_history: List = None) -> Dict[str, Any]:
        """使用LLM内置知识回答（知识库无结果时）"""
        history_context = ""
        if chat_history and len(chat_history) > 0:
            history_context = "\n\n对话历史：\n"
            for q, a in chat_history[-3:]:
                history_context += f"用户：{q}\n助手：{a}\n"

        internal_prompt = f"""
        {history_context}
        知识库中未检索到与当前问题直接相关的文献内容，将基于内置知识回答。

        用户问题：{query}

        如果你的知识中也没有相关信息，请直接说："抱歉，我无法回答这个问题。"
        不要编造信息，不要猜测，保持回答真实可靠。
        """

        system_prompt = """
        你是一位知识渊博的助手。请基于你的内置知识回答用户的问题。

        规则：
        1. 如果你知道答案，请直接回答，确保信息真实可靠
        2. 如果不确定或不知道，请说："抱歉，我无法回答这个问题。"
        3. 不要编造信息，保持诚实
        4. 回答要准确、简洁
        """

        answer = self.llm_client.complete(
            prompt=internal_prompt,
            system_prompt=system_prompt
        )

        # 检查是否是"无法回答"的情况
        if "无法回答" in answer or "不知道" in answer or "抱歉" in answer:
            return {
                "answer": "抱歉，我无法回答这个问题。你可以尝试添加更多相关文献到知识库，或者换一个问题。",
                "source_documents": [],
                "retrieval_count": 0,
                "question_type": "knowledge_no_result",
                "confidence": 0.2
            }

        return {
            "answer": f"（注：此回答基于我的内置知识，非知识库内容）\n\n{answer}",
            "source_documents": [],
            "retrieval_count": 0,
            "question_type": "knowledge_internal",
            "confidence": 0.5
        }

    def summarize_paper(self, document_id: str) -> Dict[str, Any]:
        """
        对单篇论文进行结构化精读总结
        Args:
            document_id: 文献ID
        Returns:
            包含总结内容的字典
        """
        # 直接从向量数据库获取该文档的所有块，而不是通过语义检索
        paper_chunks = self.knowledge_manager.get_document_chunks(document_id)

        if not paper_chunks:
            return {
                "summary": "无法找到该论文的内容，请确保已将其添加到知识库。",
                "source_document_id": document_id
            }

        paper_content = "\n\n".join([chunk.content for chunk in paper_chunks])

        if len(paper_content) > self.config["max_context_length"]:
            paper_content = paper_content[:self.config["max_context_length"]] + "..."

        prompt = f"""
以下是论文的内容：

{paper_content}

请按照指定格式对这篇论文进行结构化总结。
"""

        summary = self.llm_client.complete(
            prompt=prompt,
            system_prompt=PAPER_SUMMARY_SYSTEM_PROMPT
        )

        return {
            "summary": summary,
            "source_document_id": document_id,
            "chunks_used": len(paper_chunks)
        }

    def compare_papers(self, document_ids: List[str]) -> Dict[str, Any]:
        """
        对比分析多篇论文
        Args:
            document_ids: 文献ID列表
        Returns:
            包含对比分析的字典
        """
        if len(document_ids) < 2:
            return {
                "comparison": "请至少选择两篇论文进行对比。",
                "source_document_ids": document_ids
            }

        all_papers_content = ""

        for i, doc_id in enumerate(document_ids, 1):
            # 直接从向量数据库获取该文档的所有块
            paper_chunks = self.knowledge_manager.get_document_chunks(doc_id)

            if not paper_chunks:
                continue

            paper_content = "\n\n".join([chunk.content for chunk in paper_chunks])

            if len(paper_content) > 2000:
                paper_content = paper_content[:2000] + "..."

            all_papers_content += f"## 论文{i}\n{paper_content}\n\n"

        if not all_papers_content:
            return {
                "comparison": "无法找到所选论文的内容，请确保已将它们添加到知识库。",
                "source_document_ids": document_ids
            }

        prompt = f"""
以下是需要对比的论文内容：

{all_papers_content}

请按照指定格式对这些论文进行对比分析。
"""

        comparison = self.llm_client.complete(
            prompt=prompt,
            system_prompt=CROSS_PAPER_COMPARISON_SYSTEM_PROMPT
        )

        return {
            "comparison": comparison,
            "source_document_ids": document_ids
        }

    def _process_retrieval_results(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """
        处理检索结果，构建上下文文档列表
        Args:
            results: 检索结果列表
        Returns:
            处理后的上下文文档列表
        """
        context_docs = []

        for result in results:
            context_docs.append({
                "chunk_id": result.chunk.chunk_id,
                "document_id": result.chunk.document_id,
                "file_name": result.chunk.file_name,
                "content": result.chunk.content,
                "metadata": result.chunk.metadata,
                "score": result.score
            })

        total_length = 0
        truncated_docs = []

        for doc in context_docs:
            if total_length + len(doc["content"]) > self.config["max_context_length"]:
                remaining_length = self.config["max_context_length"] - total_length
                if remaining_length > 100:
                    doc["content"] = doc["content"][:remaining_length] + "..."
                    truncated_docs.append(doc)
                break

            truncated_docs.append(doc)
            total_length += len(doc["content"])

        return truncated_docs

    def _answer_with_langgraph(self, query: str, chat_history: List = None) -> Dict[str, Any]:
        """使用 LangGraph 工作流回答问题"""
        if self.langgraph_workflow is None:
            try:
                self.langgraph_workflow = LangGraphRAGWorkflow()
            except Exception as e:
                print(f"LangGraph工作流初始化失败，回退到基础模式: {e}")
                return self.answer_question(query, chat_history, use_langgraph=False)

        try:
            result = self.langgraph_workflow.run(query, chat_history)
            return result
        except Exception as e:
            print(f"LangGraph工作流执行失败，回退到基础模式: {e}")
            return self.answer_question(query, chat_history, use_langgraph=False)
