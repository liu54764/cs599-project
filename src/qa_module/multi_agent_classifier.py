"""多Agent协作的智能问题分类系统

使用LangGraph实现多Agent协作，包含：
- ClassificationAgent: 分类专家，负责判断问题类型
- RetrievalAgent: 检索专家，负责知识库检索
- AnswerGenerator: 回答生成专家，负责生成最终回答
- SupervisorAgent: 协调员，负责协调各Agent工作
"""
from typing import List, Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langchain.tools import tool
import json
import threading

# 导入现有组件
from .question_classifier import QuestionClassifier
from .prompt_templates import build_qa_prompt, ACADEMIC_QA_SYSTEM_PROMPT
from knowledge_base import KnowledgeManager
from .llm_client import LLMClient


# 分类配置常量（提取为配置，便于维护和扩展）
CLASSIFICATION_CONFIG = {
    "technical_keywords": [
        "python", "java", "javascript", "sql", "database",
        "mongodb", "mysql", "redis", "algorithm", "code",
        "编程", "开发", "实现", "架构", "设计", "框架",
        "函数", "方法", "类", "接口", "API", "代码",
        "调试", "错误", "bug", "性能", "优化", "部署"
    ],
    "creative_keywords": [
        "写", "创作", "设计", "故事", "诗歌", "文案",
        "策划", "方案", "创意", "灵感", "生成", "编",
        "描述", "构思", "想象", "文案", "标题", "摘要"
    ],
    "knowledge_keywords": [
        "什么是", "是什么", "定义", "原理", "如何", "区别",
        "为什么", "解释", "说明", "介绍", "含义", "意思",
        "作用", "用途", "特点", "优势", "劣势", "比较"
    ],
    "chat_keywords": [
        "你好", "嗨", "天气", "心情", "谢谢", "再见",
        "你是谁", "你叫什么", "你能", "帮我", "我想",
        "今天", "现在", "最近", "感觉", "觉得", "开心"
    ],
    "priority": ["chat", "technical", "creative", "knowledge"]  # 分类优先级
}


def get_category_name(category: str) -> str:
    """获取分类的中文名称"""
    names = {
        "chat": "闲聊",
        "knowledge": "知识",
        "technical": "技术",
        "creative": "创意"
    }
    return names.get(category, category)


class AgentState(TypedDict):
    """多Agent状态定义"""
    query: str
    chat_history: List[Dict[str, str]]
    classification: Optional[str]  # 'chat' | 'knowledge' | 'technical' | 'creative'
    retrieval_results: Optional[List[Dict[str, Any]]]
    confidence: float
    final_answer: Optional[str]
    agent_info: List[str]  # 记录各Agent的决策过程


@tool
def classify_question(query: str, chat_history: str = "") -> str:
    """
    对用户问题进行智能分类

    Args:
        query: 用户的问题
        chat_history: 可选的对话历史，用于上下文理解

    Returns:
        JSON格式的分类结果，包含:
        - type: 问题类型 ('chat' | 'knowledge' | 'technical' | 'creative')
        - confidence: 置信度 (0-1)
        - reason: 分类理由
        - needs_retrieval: 是否需要检索知识库
    """
    query_lower = query.lower()

    # 统计每个类别的关键词匹配数
    category_matches = {
        "technical": sum(1 for kw in CLASSIFICATION_CONFIG["technical_keywords"] 
                        if kw in query_lower),
        "creative": sum(1 for kw in CLASSIFICATION_CONFIG["creative_keywords"] 
                       if kw in query_lower),
        "knowledge": sum(1 for kw in CLASSIFICATION_CONFIG["knowledge_keywords"] 
                        if kw in query_lower),
        "chat": sum(1 for kw in CLASSIFICATION_CONFIG["chat_keywords"] 
                   if kw in query_lower)
    }

    # 按优先级选择匹配数最多的类别
    max_matches = -1
    classification = "knowledge"  # 默认分类
    reason = "默认分类"

    for category in CLASSIFICATION_CONFIG["priority"]:
        if category_matches[category] > max_matches:
            max_matches = category_matches[category]
            classification = category
            reason = f"检测到 {max_matches} 个{get_category_name(category)}关键词"

    # 计算置信度（基于匹配数量）
    if max_matches == 0:
        confidence = 0.5  # 默认置信度
        reason = "未检测到明确关键词，使用默认分类"
    elif max_matches == 1:
        confidence = 0.65
    elif max_matches == 2:
        confidence = 0.8
    else:
        confidence = min(0.95, 0.8 + max_matches * 0.03)

    # 基于上下文的额外判断
    if chat_history and classification == "chat":
        # 如果有对话历史且分类为闲聊，提高置信度
        confidence = min(0.95, confidence + 0.1)
        reason += "，结合对话历史确认"

    needs_retrieval = classification in ['knowledge', 'technical']

    result = {
        "type": classification,
        "confidence": confidence,
        "reason": reason,
        "needs_retrieval": needs_retrieval,
        "matches": category_matches  # 添加匹配详情便于调试
    }

    return json.dumps(result, ensure_ascii=False)


@tool
def retrieve_knowledge(query: str, top_k: int = 5) -> str:
    """
    从知识库中检索相关内容

    Args:
        query: 用户的问题
        top_k: 返回结果数量（默认5）

    Returns:
        JSON格式的检索结果
    """
    try:
        knowledge_manager = KnowledgeManager()
        results = knowledge_manager.search_knowledge_base(query, top_k=top_k)

        retrieval_results = []
        for result in results:
            retrieval_results.append({
                "content": result.chunk.content,
                "file_name": result.chunk.file_name,
                "score": result.score,
                "document_id": result.chunk.document_id
            })

        return json.dumps({
            "success": True,
            "results": retrieval_results,
            "count": len(retrieval_results)
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "results": [],
            "count": 0
        }, ensure_ascii=False)


@tool
def analyze_context(chat_history: str, query: str) -> str:
    """
    分析对话上下文，判断问题是否需要上下文理解

    Args:
        chat_history: 对话历史
        query: 当前问题

    Returns:
        JSON格式的分析结果
    """
    query_lower = query.lower()

    # 检查是否需要上下文
    context_indicators = ["之前", "刚才", "上次", "之前说", "刚才说",
                          "它", "这个", "那个", "这些", "那些",
                          "我之前", "你之前", "我们之前"]

    needs_context = any(indicator in query_lower for indicator in context_indicators)

    result = {
        "needs_context": needs_context,
        "indicators_found": [ind for ind in context_indicators if ind in query_lower],
        "suggestion": "需要参考对话历史进行理解" if needs_context else "可以独立回答"
    }

    return json.dumps(result, ensure_ascii=False)


class ClassificationExpert:
    """问题分类专家Agent"""

    def __init__(self):
        self.name = "分类专家"

    def __call__(self, state: AgentState) -> AgentState:
        """执行分类任务"""
        query = state["query"]
        chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state.get("chat_history", [])])

        # 调用分类工具
        classification_result = classify_question.invoke({
            "query": query,
            "chat_history": chat_history
        })

        result = json.loads(classification_result)

        state["classification"] = result["type"]
        state["confidence"] = result["confidence"]
        state["agent_info"].append(f"【{self.name}】分类结果: {result['type']} (置信度: {result['confidence']:.2f})")
        state["agent_info"].append(f"【{self.name}】分类理由: {result['reason']}")

        return state


class RetrievalExpert:
    """知识库检索专家Agent"""

    def __init__(self):
        self.name = "检索专家"

    def __call__(self, state: AgentState) -> AgentState:
        """执行检索任务"""
        query = state["query"]

        # 调用检索工具
        retrieval_result = retrieve_knowledge.invoke({"query": query, "top_k": 5})

        result = json.loads(retrieval_result)

        if result["success"]:
            state["retrieval_results"] = result["results"]
            state["agent_info"].append(f"【{self.name}】检索到 {result['count']} 条相关结果")
        else:
            state["retrieval_results"] = []
            state["agent_info"].append(f"【{self.name}】检索失败: {result['error']}")

        return state


class ContextAnalyzer:
    """上下文分析专家Agent"""

    def __init__(self):
        self.name = "上下文分析专家"

    def __call__(self, state: AgentState) -> AgentState:
        """分析上下文需求"""
        query = state["query"]
        chat_history = state.get("chat_history", [])

        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

        # 调用上下文分析工具
        context_result = analyze_context.invoke({
            "chat_history": history_str,
            "query": query
        })

        result = json.loads(context_result)

        state["agent_info"].append(f"【{self.name}】需要上下文: {result['needs_context']}")
        if result["indicators_found"]:
            state["agent_info"].append(f"【{self.name}】检测到的上下文指示词: {', '.join(result['indicators_found'])}")

        return state


class AnswerGenerator:
    """回答生成专家Agent（新增）"""

    def __init__(self):
        self.name = "回答生成专家"

    def __call__(self, state: AgentState) -> AgentState:
        """生成最终回答"""
        query = state["query"]
        retrieval_results = state.get("retrieval_results", [])
        classification = state.get("classification", "unknown")

        if classification == 'chat':
            # 闲聊类问题直接回答
            from .qa_engine import QAEngine
            qa_engine = QAEngine()
            answer = qa_engine._answer_chat(query, state.get("chat_history", []))
        elif not retrieval_results:
            # 无检索结果，使用备用回答
            answer = "知识库中未找到相关内容。"
        else:
            # 有检索结果，结合知识库回答
            context_documents = []
            for result in retrieval_results:
                context_documents.append({
                    "file_name": result["file_name"],
                    "content": result["content"]
                })

            prompt = build_qa_prompt(query, context_documents)
            llm_client = LLMClient()

            answer = llm_client.complete(
                prompt=prompt,
                system_prompt=ACADEMIC_QA_SYSTEM_PROMPT
            )

        state["final_answer"] = answer
        state["agent_info"].append(f"【{self.name}】已生成最终回答")

        return state


class SupervisorAgent:
    """监督协调员Agent"""

    def __init__(self):
        self.name = "协调员"

    def decide_next_step(self, state: AgentState) -> str:
        """决定下一步行动"""
        classification = state.get("classification")
        confidence = state.get("confidence", 0)

        # 如果分类置信度低，需要进一步分析
        if confidence < 0.7:
            return "context_analyzer"

        # 根据分类结果决定
        if classification in ['knowledge', 'technical']:
            # 知识类/技术类问题，需要检索
            return "retrieval_expert"
        elif classification == 'chat':
            # 闲聊类问题，直接回答
            return "answer_generator"
        elif classification == 'creative':
            # 创意类问题，不需要检索
            return "answer_generator"

        return "answer_generator"


def create_multi_agent_classifier():
    """创建多Agent分类系统"""
    # 初始化Agent
    classifier = ClassificationExpert()
    context_analyzer = ContextAnalyzer()
    retrieval_expert = RetrievalExpert()
    answer_generator = AnswerGenerator()
    supervisor = SupervisorAgent()

    # 创建状态图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("classification_expert", classifier)
    workflow.add_node("context_analyzer", context_analyzer)
    workflow.add_node("retrieval_expert", retrieval_expert)
    workflow.add_node("answer_generator", answer_generator)

    # 设置入口点
    workflow.set_entry_point("classification_expert")

    # 添加条件边
    def route_after_classification(state: AgentState) -> str:
        return supervisor.decide_next_step(state)

    workflow.add_conditional_edges(
        "classification_expert",
        route_after_classification,
        {
            "context_analyzer": "context_analyzer",
            "retrieval_expert": "retrieval_expert",
            "answer_generator": "answer_generator"
        }
    )

    def route_after_context(state: AgentState) -> str:
        return supervisor.decide_next_step(state)

    workflow.add_conditional_edges(
        "context_analyzer",
        route_after_context,
        {
            "retrieval_expert": "retrieval_expert",
            "answer_generator": "answer_generator"
        }
    )

    # 检索完成后生成回答
    workflow.add_edge("retrieval_expert", "answer_generator")

    # 回答生成后结束
    workflow.add_edge("answer_generator", END)

    # 编译工作流
    app = workflow.compile()

    return app


class MultiAgentQuestionClassifier:
    """多Agent问题分类器"""

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
        """初始化多Agent系统"""
        self.workflow = create_multi_agent_classifier()

    def classify(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        使用多Agent系统进行问题分类并生成回答

        Args:
            query: 用户问题
            chat_history: 对话历史

        Returns:
            分类结果和回答字典
        """
        # 构建初始状态
        initial_state = {
            "query": query,
            "chat_history": chat_history or [],
            "classification": None,
            "retrieval_results": None,
            "confidence": 0.0,
            "final_answer": None,
            "agent_info": []
        }

        # 执行工作流
        result = self.workflow.invoke(initial_state)

        # 返回结果
        return {
            "query": query,
            "classification": result["classification"],
            "confidence": result["confidence"],
            "needs_retrieval": result["classification"] in ['knowledge', 'technical'],
            "retrieval_results": result["retrieval_results"],
            "final_answer": result["final_answer"],
            "agent_info": result["agent_info"],
            "chat_history": result["chat_history"]
        }

    def should_search_knowledge_base(self, query: str, chat_history: Optional[List] = None) -> bool:
        """判断是否需要检索知识库"""
        result = self.classify(query, chat_history)
        return result["needs_retrieval"]