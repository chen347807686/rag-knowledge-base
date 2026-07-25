from langchain_chroma import Chroma

from app.config import VECTOR_DB_PATH


def create_vector_store(
        chunks,
        embedding_model
):
    """
    创建向量数据库，将文档chunk写入Chroma并持久化

    参数:
        chunks: 切分后的文档列表
        embedding_model: Embedding模型实例

    返回:
        Chroma向量存储实例
    """
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=VECTOR_DB_PATH
    )

    return vector_store


def load_vector_store(embedding_model):
    """
    加载已持久化的向量数据库

    参数:
        embedding_model: Embedding模型实例

    返回:
        Chroma向量存储实例
    """
    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=VECTOR_DB_PATH
    )

    return vector_store


def retrieve(
        query: str,
        embedding_model,
        top_k: int = 4
):
    """
    相似度检索：根据用户问题从向量库中检索最相关的文档chunk

    参数:
        query: 用户问题
        embedding_model: Embedding模型实例
        top_k: 返回的最相似文档数量

    返回:
        List[Document]: 最相关的文档chunk列表
    """
    # 加载已有的向量数据库
    vector_store = load_vector_store(embedding_model)

    # 相似度检索
    docs = vector_store.similarity_search(
        query,
        k=top_k
    )


    return docs