from typing import Optional, Generator
import threading
from .config import LLM_CONFIG


class LLMClient:
    """统一的LLM客户端，支持多种模型提供商"""

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
            with LLMClient._lock:
                if not self._initialized:
                    self._initialize()
                    self._initialized = True

    def _initialize(self):
        """初始化指定的LLM客户端"""
        self.provider = LLM_CONFIG["provider"]

        if self.provider == "ollama":
            self._init_ollama()
        elif self.provider == "openai":
            self._init_openai()
        elif self.provider == "deepseek":
            self._init_deepseek()
        elif self.provider == "anthropic":
            self._init_anthropic()
        else:
            raise ValueError(f"不支持的LLM提供商: {self.provider}")

        print(f"✅ LLM客户端初始化完成: {self.provider} - {self.config['model_name']}")

    def _init_ollama(self):
        """初始化Ollama客户端"""
        try:
            import ollama
            self.config = LLM_CONFIG["ollama"]
            self.client = ollama.Client(host=self.config["base_url"])
        except ImportError:
            raise ImportError("请安装ollama库: pip install ollama")

    def _init_openai(self):
        """初始化OpenAI客户端"""
        try:
            from openai import OpenAI
            self.config = LLM_CONFIG["openai"]
            self.client = OpenAI(
                api_key=self.config["api_key"],
                base_url=self.config["base_url"]
            )
        except ImportError:
            raise ImportError("请安装openai库: pip install openai")

    def _init_deepseek(self):
        """初始化DeepSeek客户端（兼容OpenAI格式）"""
        try:
            from openai import OpenAI
            self.config = LLM_CONFIG["deepseek"]

            if not self.config["api_key"]:
                raise ValueError("未找到环境变量 DEEPSEEK_API_KEY，请先配置")

            self.client = OpenAI(
                api_key=self.config["api_key"],
                base_url=self.config["base_url"]
            )
        except ImportError:
            raise ImportError("请安装openai库: pip install openai")

    def _init_anthropic(self):
        """初始化Anthropic客户端"""
        try:
            from anthropic import Anthropic
            self.config = LLM_CONFIG["anthropic"]
            self.client = Anthropic(api_key=self.config["api_key"])
        except ImportError:
            raise ImportError("请安装anthropic库: pip install anthropic")

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """生成文本补全"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if self.provider == "ollama":
            response = self.client.chat(
                model=self.config["model_name"],
                messages=messages,
                options={
                    "temperature": self.config["temperature"],
                    "num_predict": self.config["max_tokens"]
                }
            )
            return response["message"]["content"].strip()
        elif self.provider in ["openai", "deepseek"]:
            response = self.client.chat.completions.create(
                model=self.config["model_name"],
                messages=messages,
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
                stream=False
            )
            return response.choices[0].message.content.strip()
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.config["model_name"],
                system=system_prompt,
                messages=messages,
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"]
            )
            return response.content[0].text.strip()

    def stream_complete(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        """流式生成文本补全"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if self.provider == "ollama":
            response = self.client.chat(
                model=self.config["model_name"],
                messages=messages,
                options={
                    "temperature": self.config["temperature"],
                    "num_predict": self.config["max_tokens"]
                },
                stream=True
            )
            for chunk in response:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
        elif self.provider in ["openai", "deepseek"]:
            response = self.client.chat.completions.create(
                model=self.config["model_name"],
                messages=messages,
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
                stream=True
            )
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.config["model_name"],
                system=system_prompt,
                messages=messages,
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
                stream=True
            )
            for chunk in response:
                if chunk.type == "content_block_delta":
                    content = chunk.delta.text
                    if content:
                        yield content
