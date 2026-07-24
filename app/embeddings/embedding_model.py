from langchain_community.embeddings import DashScopeEmbeddings

from app.config import DASHSCOPE_API_KEY


def get_embedding_model():

    # 使用通义千问文本向量模型
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key=DASHSCOPE_API_KEY
    )

    return embeddings