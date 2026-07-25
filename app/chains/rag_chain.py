from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.config import RETRIEVAL_TOP_K
from app.llm.qwen_model import get_qwen_model
from app.vectorstore.chroma_store import create_vector_store
from app.embeddings.embedding_model import get_embedding_model


def create_rag_chain(vector_store):

    # 创建检索器
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": RETRIEVAL_TOP_K
        }
    )


    # 创建提示词模板
    prompt = ChatPromptTemplate.from_template(
        """
你是一个知识库问答助手。

请根据下面提供的上下文回答问题。
如果上下文没有相关信息，请回答不知道。

上下文:
{context}

问题:
{question}
"""
    )


    # 获取Qwen模型
    llm = get_qwen_model()


    # LCEL链
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )


    return rag_chain