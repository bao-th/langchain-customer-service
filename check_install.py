import os
from importlib.metadata import version
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_deepseek import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

print("LangChain 版本：", version("langchain"))
print("LangGraph 版本：", version("langgraph"))

assert os.getenv("DEEPSEEK_API_KEY"), "缺少 DEEPSEEK_API_KEY"
assert os.getenv("DASHSCOPE_API_KEY"), "缺少 DASHSCOPE_API_KEY"

print("依赖导入成功，环境变量已配置。")