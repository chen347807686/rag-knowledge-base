from langchain_chroma import Chroma

from app.config import VECTOR_DB_PATH


def create_vector_store(
        chunks,
        embedding_model
):

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=VECTOR_DB_PATH
    )


    return vector_store