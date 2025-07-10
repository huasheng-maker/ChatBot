from typing import Dict, Any, List, Optional

class PromptTemplate:
    """提示词模板类"""
    
    def __init__(self, template: str, system_prompt: str = ""):
        self.template = template
        self.system_prompt = system_prompt
    
    def format(self, user_input: str, context: Optional[List[Dict[str, str]]] = None) -> str:
        """格式化提示词模板"""
        if context:
            # 将上下文信息整合到提示词中
            context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
            return self.template.format(user_input=user_input, context=context_str, system_prompt=self.system_prompt)
        else:
            return self.template.format(user_input=user_input, system_prompt=self.system_prompt)

class PromptManager:
    """提示词管理器"""
    
    def __init__(self):
        self.templates = {
            "default": PromptTemplate(
                template="{system_prompt}\n\n用户问题: {user_input}",
                system_prompt="你是一个智能助手，回答用户的问题"
            ),
            "code_generation": PromptTemplate(
                template="{system_prompt}\n\n请基于Python语言，实现以下功能，要求代码可运行，并附带注释。\n\n功能描述: {user_input}",
                system_prompt="你是一个专业的Python程序员"
            ),
            "qa": PromptTemplate(
                template="{system_prompt}\n\n上下文信息:\n{context}\n\n用户问题: {user_input}",
                system_prompt="请根据上下文信息，回答用户的问题"
            )
        }
    
    def get_template(self, template_name: str) -> PromptTemplate:
        """获取提示词模板"""
        return self.templates.get(template_name, self.templates["default"])
    
    def register_template(self, template_name: str, template: PromptTemplate) -> None:
        """注册新的提示词模板"""
        self.templates[template_name] = template
    
    def generate_prompt(self, template_name: str, user_input: str, context: Optional[List[Dict[str, str]]] = None) -> str:
        """生成完整提示词"""
        template = self.get_template(template_name)
        return template.format(user_input, context)    