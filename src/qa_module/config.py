import os

# LLM模型配置
LLM_CONFIG = {
    # 默认使用DeepSeek（已配置环境变量）
    "provider": "deepseek",

    # DeepSeek 配置（自动从环境变量读取API Key）
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com/v1",
        "model_name": "deepseek-chat",
        "temperature": 0.1,
        "max_tokens": 4096,
    },

    # Ollama 本地模型配置
    "ollama": {
        "base_url": "http://localhost:11434",
        "model_name": "llama3:8b",
        "temperature": 0.1,
        "max_tokens": 2048,
    },

    # OpenAI 配置
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-3.5-turbo-0125",
        "temperature": 0.1,
        "max_tokens": 2048,
    },

    # Anthropic 配置
    "anthropic": {
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "model_name": "claude-3-sonnet-20240229",
        "temperature": 0.1,
        "max_tokens": 4096,
    }
}

# 问答配置
QA_CONFIG = {
    "retrieval_top_k": 6,
    "max_context_length": 8000,
    "include_citations": True,
    "return_source_documents": True,
    "unknown_answer": "根据提供的文献内容，我无法回答这个问题。请尝试更具体的问题，或者添加更多相关文献到知识库。"
}
