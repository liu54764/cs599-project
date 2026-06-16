"""基于 LangGraph 的 RAG 工作流编排"""
from typing import List, Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .prompt_templates import build_qa_prompt, ACADEMIC_QA_SYSTEM_PROMPT
from knowledge_base import KnowledgeManager
from .question_classifier import QuestionClassifier
from .llm_client import LLMClient


class RAGState(TypedDict):
    """RAG 工作流状态"""
    query: str
    question_type: str
    chat_history: List[tuple]
    compressed_history: str
    retrieval_results: List[Dict[str, Any]]
    answer: str
    confidence: float
    needs_retry: bool
    retry_count: int


class LangGraphRAGWorkflow:
    """基于 LangGraph 的 RAG 工作流，支持内置 Memory 和历史压缩"""

    def __init__(self):
        self.knowledge_manager = KnowledgeManager()
        self.llm_client = LLMClient()
        self.memory = MemorySaver()
        self.workflow = self._build_workflow()
        self.max_history_length = 10
        self.compression_threshold = 5

    def _build_workflow(self):
        """构建 RAG 工作流图"""
        graph = StateGraph(RAGState)

        graph.add_node("compress_history", self._compress_history)
        graph.add_node("classify", self._classify_question)
        graph.add_node("retrieve", self._retrieve_knowledge)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("evaluate_answer", self._evaluate_answer)
        graph.add_node("summarize_chat", self._summarize_chat)
        graph.add_node("fallback_answer", self._fallback_answer)
        graph.add_node("update_history", self._update_history)

        graph.set_entry_point("compress_history")

        graph.add_edge("compress_history", "classify")

        graph.add_conditional_edges(
            "classify",
            self._decide_next_step,
            {
                "knowledge": "retrieve",
                "chat": "summarize_chat",
                "unknown": "retrieve"
            }
        )

        graph.add_edge("retrieve", "generate_answer")

        graph.add_edge("generate_answer", "evaluate_answer")

        graph.add_conditional_edges(
            "evaluate_answer",
            self._should_retry,
            {
                "retry": "retrieve",
                "fallback": "fallback_answer",
                "finish": "update_history"
            }
        )

        graph.add_edge("fallback_answer", "update_history")

        graph.add_edge("summarize_chat", "update_history")

        graph.add_edge("update_history", END)

        return graph.compile(checkpointer=self.memory)

    def _compress_history(self, state: RAGState) -> RAGState:
        """压缩对话历史，当历史过长时进行总结"""
        chat_history = state["chat_history"]
        compressed_history = ""

        if len(chat_history) >= self.compression_threshold:
            history_to_compress = chat_history[:-2]
            history_text = "\n".join([f"用户：{q}\n助手：{a}" for q, a in history_to_compress])
            
            prompt = f"""
请对以下对话历史进行简洁总结：

{history_text}

总结要求：
1. 提取关键信息和结论
2. 保持对话主题和上下文
3. 不要超过150字
4. 用中文总结
            """
            
            try:
                compressed_history = self.llm_client.complete(
                    prompt,
                    system_prompt="你是一个专业的对话总结助手，请简洁地总结对话内容。"
                )
                compressed_history = compressed_history.strip()
            except Exception as e:
                print(f"历史压缩失败: {str(e)}")
                compressed_history = ""

        return {
            **state,
            "compressed_history": compressed_history,
            "chat_history": chat_history[-self.max_history_length:]
        }

    def _classify_question(self, state: RAGState) -> RAGState:
        """分类问题类型"""
        question_type = QuestionClassifier.classify(state["query"])
        print(f"问题分类: {question_type}")

        return {
            **state,
            "question_type": question_type,
            "needs_retry": False,
            "retry_count": 0
        }

    def _decide_next_step(self, state: RAGState) -> str:
        """根据问题类型决定下一步"""
        return state["question_type"]

    def _retrieve_knowledge(self, state: RAGState) -> RAGState:
        """检索知识库"""
        query = state["query"]

        if state["chat_history"]:
            last_q, _ = state["chat_history"][-1]
            enhanced_query = f"（上文：{last_q}）{query}"
        else:
            enhanced_query = query

        results = self.knowledge_manager.search_knowledge_base(
            enhanced_query,
            top_k=8,
            use_enhanced=True
        )

        retrieval_results = []
        for result in results:
            retrieval_results.append({
                "content": result.chunk.content,
                "file_name": result.chunk.file_name,
                "score": result.score,
                "metadata": result.chunk.metadata
            })

        print(f"检索到 {len(retrieval_results)} 条结果")

        return {
            **state,
            "retrieval_results": retrieval_results,
            "needs_retry": len(retrieval_results) == 0,
            "retry_count": state["retry_count"] + 1
        }

    def _generate_answer(self, state: RAGState) -> RAGState:
        """生成回答"""
        query = state["query"]
        results = state["retrieval_results"]
        chat_history = state["chat_history"]
        compressed_history = state["compressed_history"]

        if not results:
            return {
                **state,
                "answer": "",
                "confidence": 0.0,
                "needs_retry": True
            }

        full_history = chat_history
        if compressed_history:
            full_history = [(f"【对话总结】{compressed_history}", "")] + chat_history[-2:]

        prompt = build_qa_prompt(query, results[:5], full_history)

        answer_parts = []
        for chunk in self.llm_client.stream_complete(
            prompt=prompt,
            system_prompt=ACADEMIC_QA_SYSTEM_PROMPT
        ):
            answer_parts.append(chunk)

        answer = "".join(answer_parts)

        return {
            **state,
            "answer": answer,
            "confidence": self._estimate_confidence(answer, results)
        }

    def _estimate_confidence(self, answer: str, results: List[Dict]) -> float:
        """估算回答置信度"""
        if "未找到相关内容" in answer:
            return 0.2

        avg_score = sum(r["score"] for r in results) / len(results) if results else 0

        source_count = answer.count("文献")

        confidence = min(0.95, avg_score * 0.7 + source_count * 0.05 + 0.2)
        return confidence

    def _evaluate_answer(self, state: RAGState) -> RAGState:
        """评估回答质量"""
        confidence = state["confidence"]
        retry_count = state["retry_count"]

        print(f"回答置信度: {confidence:.2f}, 重试次数: {retry_count}")

        if confidence < 0.3 and retry_count < 2:
            return {**state, "needs_retry": True}
        elif confidence < 0.3:
            return {**state, "needs_retry": False}

        return {**state, "needs_retry": False}

    def _should_retry(self, state: RAGState) -> str:
        """决定是否重试"""
        if state["needs_retry"] and state["retry_count"] < 2:
            return "retry"
        elif state["confidence"] < 0.3:
            return "fallback"
        return "finish"

    def _summarize_chat(self, state: RAGState) -> RAGState:
        """总结闲聊回答"""
        query = state["query"]
        chat_history = state["chat_history"]
        compressed_history = state["compressed_history"]

        history_context = ""
        if compressed_history:
            history_context = f"【对话总结】{compressed_history}\n"

        recent_history = chat_history[-3:]
        for q, a in recent_history:
            history_context += f"用户：{q}\n助手：{a}\n"

        prompt = f"""
{history_context}

用户问题：{query}

请以友好、自然的方式回答用户的问题。
"""

        answer = self.llm_client.complete(
            prompt,
            system_prompt="你是一个友好的聊天助手，擅长进行自然、有趣的对话。"
        )

        return {
            **state,
            "answer": answer,
            "confidence": 0.8,
            "needs_retry": False
        }

    def _fallback_answer(self, state: RAGState) -> RAGState:
        """备用回答（知识库无结果时）"""
        query = state["query"]

        prompt = f"""
用户问了一个问题，但知识库中没有找到相关内容。请使用你自己的知识回答：

用户问题：{query}

如果你也不知道答案，请说："抱歉，我无法回答这个问题。"
"""

        answer = self.llm_client.complete(
            prompt,
            system_prompt="你是一位知识渊博的助手，请诚实回答问题。"
        )

        if "无法回答" in answer or "不知道" in answer:
            answer = "抱歉，我无法回答这个问题。你可以尝试添加更多相关文献到知识库，或者换一个问题。"

        return {
            **state,
            "answer": answer,
            "confidence": 0.5,
            "needs_retry": False
        }

    def _update_history(self, state: RAGState) -> RAGState:
        """更新对话历史"""
        new_history = state["chat_history"] + [(state["query"], state["answer"])]
        new_history = new_history[-self.max_history_length:]

        return {
            **state,
            "chat_history": new_history
        }

    def run(self, query: str, chat_history: List[tuple] = None) -> Dict[str, Any]:
        """运行 RAG 工作流，使用内置 Memory 管理对话状态"""
        config = {"configurable": {"thread_id": "default_session"}}

        try:
            checkpoint = self.workflow.get_state(config)
            existing_history = checkpoint.values.get("chat_history", [])
            
            if chat_history:
                combined_history = existing_history + chat_history
            else:
                combined_history = existing_history
        except Exception:
            combined_history = chat_history if chat_history else []

        initial_state = {
            "query": query,
            "question_type": "",
            "chat_history": combined_history,
            "compressed_history": "",
            "retrieval_results": [],
            "answer": "",
            "confidence": 0.0,
            "needs_retry": False,
            "retry_count": 0
        }

        result = self.workflow.invoke(initial_state, config)

        return {
            "answer": result["answer"],
            "question_type": result["question_type"],
            "retrieval_count": len(result["retrieval_results"]),
            "confidence": result["confidence"],
            "source_documents": [
                {
                    "file_name": r["file_name"],
                    "content": r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"],
                    "score": r["score"]
                }
                for r in result["retrieval_results"]
            ],
            "chat_history": result["chat_history"]
        }

    def stream(self, query: str, chat_history: List[tuple] = None) -> Generator[Dict[str, Any], None, None]:
        """流式运行 RAG 工作流，使用内置 Memory 管理对话状态"""
        config = {"configurable": {"thread_id": "default_session"}}

        try:
            checkpoint = self.workflow.get_state(config)
            existing_history = checkpoint.values.get("chat_history", [])
            
            if chat_history:
                combined_history = existing_history + chat_history
            else:
                combined_history = existing_history
        except Exception:
            combined_history = chat_history if chat_history else []

        initial_state = {
            "query": query,
            "question_type": "",
            "chat_history": combined_history,
            "compressed_history": "",
            "retrieval_results": [],
            "answer": "",
            "confidence": 0.0,
            "needs_retry": False,
            "retry_count": 0
        }

        for event in self.workflow.stream(initial_state, config):
            for node, values in event.items():
                if node == "generate_answer" and "answer" in values:
                    answer = values["answer"]
                    if answer:
                        yield {"type": "answer", "content": answer}
                elif node == "_retrieve_knowledge" and "retrieval_results" in values:
                    results = values["retrieval_results"]
                    if results:
                        yield {"type": "retrieval", "results": results}
        
        final_state = self.workflow.get_state(config).values
        yield {
            "type": "finished",
            "answer": final_state.get("answer", ""),
            "question_type": final_state.get("question_type", ""),
            "retrieval_count": len(final_state.get("retrieval_results", [])),
            "confidence": final_state.get("confidence", 0.0),
            "source_documents": [
                {
                    "file_name": r["file_name"],
                    "content": r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"],
                    "score": r["score"]
                }
                for r in final_state.get("retrieval_results", [])
            ],
            "chat_history": final_state.get("chat_history", [])
        }

    def clear_memory(self):
        """清空对话记忆"""
        try:
            config = {"configurable": {"thread_id": "default_session"}}
            self.workflow.delete_state(config)
            print("对话记忆已清空")
        except Exception as e:
            print(f"清空记忆失败: {e}")
