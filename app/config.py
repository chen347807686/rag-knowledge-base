import os
from dotenv import load_dotenv


# 加载.env文件
load_dotenv()


# 阿里云DashScope API Key
DASHSCOPE_API_KEY = os.getenv(
    "DASHSCOPE_API_KEY"
)

# Qwen OpenAI兼容接口地址
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL"
)

# 文档切分参数
CHUNK_SIZE = 500

CHUNK_OVERLAP = 100


# Chroma向量数据库保存路径
VECTOR_DB_PATH = "./vector_store"

# 通义千问LLM模型
LLM_MODEL = "qwen3.7-plus"

# 检索参数
RETRIEVAL_TOP_K = 4