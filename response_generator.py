from typing import Dict, Any, List, Optional
from llm_api import LLMAPIBase
from prompt_manager import PromptManager

class ResponseGenerator:
    """回复生成器"""
    
    def __init__(self, llm_api: LLMAPIBase, prompt_manager: PromptManager):
        self.llm_api = llm_api
        self.prompt_manager = prompt_manager
        self.context = []  # 对话上下文
    
    def generate_response(self, user_input: str, template_name: str = "default", 
                          temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """生成回复"""
        # 生成提示词
        prompt = self.prompt_manager.generate_prompt(template_name, user_input, self.context)
        
        # 调用LLM API生成回复
        response = self.llm_api.generate(prompt, temperature, max_tokens)
        
        # 更新对话上下文
        self._update_context(user_input, response)
        
        return response
    
    def _update_context(self, user_input: str, response: str) -> None:
        """更新对话上下文"""
        # 添加用户输入和AI回复到上下文
        self.context.append({"role": "user", "content": user_input})
        self.context.append({"role": "assistant", "content": response})
        
        # 控制上下文长度，避免超出LLM的token限制
        # 实际应用中应该根据token计数来截断，这里简化处理
        if len(self.context) > 10:
            self.context = self.context[-10:]
    
    def clear_context(self) -> None:
        """清除对话上下文"""
        self.context = []    