import os

# LLM模型配置
LLM_CONFIG = {
    # 默认使用DeepSeek V4 Flash（已配置环境变量）
    "provider": "deepseek",  # 已设置为DeepSeek

    # DeepSeek V4 Flash 配置（自动从环境变量读取API Key）
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量获取
        "base_url": "https://api.deepseek.com/v1",
        "model_name": "deepseek-chat",  # DeepSeek V4 Flash 对应的模型名
        "temperature": 0.1,  # 学术问答低温度，保持准确性
        "max_tokens": 4096,
    },

    # 其他模型配置（保留备用）
    "ollama": {
        "base_url": "http://localhost:11434",
        "model_name": "llama3:8b",
        "temperature": 0.1,
        "max_tokens": 2048,
    },

    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-3.5-turbo-0125",
        "temperature": 0.1,
        "max_tokens": 2048,
    }
}

# 问答配置（优化适配DeepSeek）
QA_CONFIG = {
    "retrieval_top_k": 6,  # 增加到6个片段，DeepSeek上下文更大
    "max_context_length": 8000,  # 提升到8000字符，充分利用DeepSeek的长上下文
    "include_citations": True,  # 保留引用来源
    "return_source_documents": True,
    "unknown_answer": "根据提供的文献内容，我无法回答这个问题。请尝试更具体的问题，或者添加更多相关文献到知识库。"
}