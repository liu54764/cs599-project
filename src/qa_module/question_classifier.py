from typing import Optional

class QuestionClassifier:
    """问题分类器，判断问题类型"""
    
    # 知识类关键词（学术、技术相关）
    KNOWLEDGE_KEYWORDS = [
        "什么是", "什么叫", "是什么", "定义", "概念", "原理", "方法", "算法",
        "如何", "怎样", "步骤", "流程", "教程", "指南",
        "区别", "对比", "比较", "异同",
        "论文", "研究", "文献", "实验", "数据",
        "原理", "机制", "公式", "模型", "架构",
        "SQL", "数据库", "编程", "代码", "实现",
        "分析", "评估", "评价", "优缺点",
        "基于", "根据", "参考", "引用",
        "MongoDB", "MySQL", "PostgreSQL", "Redis", "Oracle", "SQLite",
        "Python", "Java", "JavaScript", "C++", "Go", "Rust"
    ]
    
    # 闲聊类关键词
    CHAT_KEYWORDS = [
        "你好", "嗨", "哈喽", "Hi", "Hello",
        "最近", "今天", "天气", "吃饭", "睡觉",
        "心情", "开心", "难过", "高兴",
        "谢谢", "不客气", "再见", "拜拜",
        "名字", "身份", "你是谁", "你能做什么",
        "聊聊天", "随便说说", "打发时间",
        "我有", "我是", "我叫", "我的", "我想",
        "你呢", "你觉得", "你认为",
        "之前", "刚才", "上次", "刚才说", "之前说",
        "几个", "多少", "多大", "多久"
    ]
    
    @classmethod
    def classify(cls, query: str) -> str:
        """
        分类问题类型
        Returns:
            'knowledge': 知识类问题，需要检索知识库
            'chat': 闲聊类问题，直接回答
            'unknown': 无法确定类型
        """
        query_lower = query.lower().strip()
        
        # 检查闲聊关键词（优先级更高）
        for keyword in cls.CHAT_KEYWORDS:
            if keyword.lower() in query_lower:
                # 特殊处理：如果同时包含知识类关键词，需要进一步判断
                has_knowledge_keyword = any(k.lower() in query_lower for k in cls.KNOWLEDGE_KEYWORDS)
                if has_knowledge_keyword:
                    # 如果问题涉及"之前说"、"刚才说"等，优先判断为闲聊（需要上下文）
                    context_keywords = ["之前", "刚才", "上次", "之前说", "刚才说"]
                    if any(c in query_lower for c in context_keywords):
                        return 'chat'
                    return 'knowledge'
                return 'chat'
        
        # 检查知识类关键词
        for keyword in cls.KNOWLEDGE_KEYWORDS:
            if keyword.lower() in query_lower:
                return 'knowledge'
        
        # 基于问题长度和复杂度判断
        if len(query) <= 10:
            # 短问题更可能是闲聊
            return 'chat'
        
        # 默认归类为知识类问题
        return 'knowledge'
    
    @classmethod
    def should_search_knowledge_base(cls, query: str) -> bool:
        """判断是否需要检索知识库"""
        classification = cls.classify(query)
        return classification == 'knowledge'
