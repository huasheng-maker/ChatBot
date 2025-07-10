from llm_api import LLMAPIFactory
from prompt_manager import PromptManager
from response_generator import ResponseGenerator

# 创建LLM API实例
llm_api = LLMAPIFactory.create_api("deepseek")  # 或者 "huggingface"

# 创建提示词管理器
prompt_manager = PromptManager()

# 创建回复生成器
response_generator = ResponseGenerator(llm_api, prompt_manager)

# 示例：生成默认回复
response = response_generator.generate_response("你好，世界！")
print("默认回复:", response)

# 示例：使用代码生成模板
code_response = response_generator.generate_response(
    "写一个快速排序算法", 
    template_name="code_generation",
    temperature=0.2  # 代码生成需要较低的温度
)
print("\n代码生成回复:", code_response)

# 示例：使用问答模板和上下文
context = [
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习是人工智能的一个分支，它关注计算机系统如何从数据中学习模式并进行预测。"}
]
qa_response = response_generator.generate_response(
    "它有哪些应用？", 
    template_name="qa",
    context=context  # 传递上下文
)
print("\n问答回复:", qa_response)    