# 导入PDF加载模块，用于读取PDF文件
from app.loaders.pdf_loader import load_pdf

# 导入文本切分模块，用于将长文本切割成多个小块
from app.loaders.text_splitter import split_documents

# 导入Embedding模型，用于将文本转换成向量
from app.embeddings.embedding_model import get_embedding_model

# 导入Chroma向量数据库创建模块
from app.vectorstore.chroma_store import create_vector_store



# 指定需要处理的PDF文件路径
pdf_path = "data/test.pdf"


# 第一步：加载PDF文档
documents = load_pdf(pdf_path)


# 第二步：对文档进行切分
# 将长文本拆分成多个chunk，方便后续向量检索
chunks = split_documents(documents)


print(
    "chunks数量:",
    len(chunks)
)


# 第三步：初始化Embedding模型
# 将文本转换成向量数据
embedding_model = get_embedding_model()


# 第四步：创建Chroma向量数据库
# 保存文本向量，用于后续RAG检索
vector_store = create_vector_store(
    chunks,
    embedding_model
)


print("向量数据库创建完成")