from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str):
    """
    加载PDF文件

    参数:
        file_path: PDF路径

    返回:
        LangChain Document列表
    """

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    return documents