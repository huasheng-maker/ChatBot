import os
import openai
from openai import OpenAI
import requests
from typing import Dict, Any, Optional, List

class LLMAPIBase:
    """LLM API的基类，定义统一接口"""
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """生成回复的统一接口"""
        raise NotImplementedError

class DeepSeekAPI(LLMAPIBase):
    """deepseek API封装"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key is required")
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        try:
            client=OpenAI(api_key=self.api_key,base_url="https://api.deepseek.cn/v1")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek API error: {e}")
            return f"Error: {str(e)}"

class HuggingFaceAPI(LLMAPIBase):
    """Hugging Face API封装"""
    
    def __init__(self, api_key: Optional[str] = None, model_id: str = "gpt2"):
        self.api_key = api_key or os.getenv("HF_API_KEY")
        self.model_id = model_id
        if not self.api_key:
            raise ValueError("Hugging Face API key is required")
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens
                }
            }
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            if response.status_code == 200:
                return response.json()[0]["generated_text"][len(prompt):]
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            print(f"Hugging Face API error: {e}")
            return f"Error: {str(e)}"

class LLMAPIFactory:
    """LLM API工厂类，用于创建不同的LLM API实例"""
    
    @staticmethod
    def create_api(provider: str, **kwargs) -> LLMAPIBase:
        if provider.lower() == "deepseek":
            return DeepSeekAPI(**kwargs)
        elif provider.lower() == "huggingface":
            return HuggingFaceAPI(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")    