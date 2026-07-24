from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def split_documents(documents):
    """
    文档切分

    参数:
        documents:
            LangChain Document列表

    返回:
        chunks
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(
        documents
    )

    return chunks