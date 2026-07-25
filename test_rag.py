from app.embeddings.embedding_model import get_embedding_model
from app.vectorstore.chroma_store import create_vector_store
from app.loaders.pdf_loader import load_pdf
from app.loaders.text_splitter import split_documents
from app.chains.rag_chain import create_rag_chain


# PDF路径
pdf_path = "data/test.pdf"


# 1. 加载PDF
documents = load_pdf(pdf_path)


# 2. 文档切分
chunks = split_documents(documents)


print("chunks数量:", len(chunks))


# 3. 创建Embedding模型
embedding_model = get_embedding_model()


# 4. 创建向量数据库
vector_store = create_vector_store(
    chunks,
    embedding_model
)


# 5. 创建RAG Chain
rag_chain = create_rag_chain(
    vector_store
)


# 6. 提问测试
question = "什么是自然语言处理？"


response = rag_chain.invoke(
    question
)


print("\n回答:")
print(response.content)