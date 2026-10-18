import os

from dotenv import load_dotenv

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

ZHIPU_API_KEY = os.getenv('ZHIPU_API_KEY')
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
