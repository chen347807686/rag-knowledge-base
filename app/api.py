# ============================================================
# FastAPI Web 服务 —— 让 RAG 系统变成一个可调用的 API
#
# 为什么要这么做：
#   之前的 main.py 每次运行都要重新读 PDF、重新切分、重新 embedding，
#   这是"脚本"思维。作为服务，应该启动时加载一次，保持运行，持续响应。
#
# 整体结构：
#   1. 应用启动时（@app.on_event("startup")）：
#        加载 embedding 模型  →  加载已持久化的 Chroma 向量库  →  创建 RAG Chain
#     这三样东西存在全局变量中，后续所有请求复用它们。
#
#   2. POST /ask 接口：
#        接收用户问题 → RAG 检索 + LLM 生成 → 返回回答和来源
#
#   3. 自动文档（/docs）：
#        FastAPI 自带 Swagger UI，启动后访问 /docs 即可交互式调试
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.embeddings.embedding_model import get_embedding_model
from app.vectorstore.chroma_store import load_vector_store
from app.chains.rag_chain import create_rag_chain


# -----------------------------------------------------------
# 第一步：定义请求体和响应体的数据模型
#
# Pydantic BaseModel：
#   FastAPI 用它们做三件事：
#   - 自动校验请求数据（类型、必填等）
#   - 生成 OpenAPI 文档
#   - 自动序列化/反序列化
# -----------------------------------------------------------

class AskRequest(BaseModel):
    """用户提问的请求体格式"""
    question: str


class SourceItem(BaseModel):
    """回答引用的来源片段"""
    content: str
    source: str


class AskResponse(BaseModel):
    """API 返回的响应体格式"""
    answer: str
    sources: list[SourceItem] = []


# -----------------------------------------------------------
# 第二步：创建 FastAPI 应用实例并初始化全局状态
#
# 为什么用全局变量（chain_ready, rag_chain）：
#   RAG Chain 一旦创建好，就可以被所有并发请求共享。
#   每次请求都重新创建它会非常慢（要重新 embedding）。
# -----------------------------------------------------------

app = FastAPI(
    title="RAG 知识库问答 API",
    description="基于 LangChain + Qwen + Chroma 的 RAG 问答服务",
    version="1.0.0"
)

# 全局变量：用于存储启动时加载的 RAG 组件
chain_ready = False
rag_chain = None


@app.on_event("startup")
async def startup():
    """
    应用启动时的初始化函数
    只执行一次，之后保持 RAG Chain 在内存中

    流程：
      1. 创建 embedding 模型
      2. 加载已有向量库（从磁盘读取）
      3. 基于向量库创建 RAG Chain
    """
    global rag_chain, chain_ready

    try:
        # 加载 embedding 模型（通义千问 text-embedding-v3）
        embedding_model = get_embedding_model()

        # 加载已持久化的 Chroma 向量库
        # 注意这里用的是 load_vector_store（读取已有），不是 create_vector_store（重新创建）
        vector_store = load_vector_store(embedding_model)

        # 创建 RAG Chain（retriever + prompt + LLM）
        rag_chain = create_rag_chain(vector_store)

        chain_ready = True
        print("✓ RAG 系统初始化完成，服务就绪")

    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        chain_ready = False


# -----------------------------------------------------------
# 第三步：定义问答接口
#
# POST /ask：
#   为什么用 POST 而不是 GET？
#   - 问题文本放在请求体里，没有 URL 长度限制
#   - 未来可以加更多参数（top_k、temperature 等）而不改 URL
# -----------------------------------------------------------

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    核心问答接口

    参数：
      request: AskRequest（包含 question 字段）

    返回：
      AskResponse（包含 answer 和 sources 字段）

    错误处理：
      - 如果 RAG 系统未初始化，返回 503
      - 如果调用 LLM 失败，返回 500
    """
    if not chain_ready or rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG 系统未初始化，请检查服务启动日志"
        )

    try:
        # 调用 RAG Chain
        # invoke 返回的是 AIMessage 对象，用 .content 拿文本
        result = rag_chain.invoke(request.question)
        answer = result.content

        # 从 result 中可以提取 source_documents（如果有的话）
        # 当前 rag_chain.py 的 LCEL 链没有传 source_documents 回来
        # 后续可以扩展让 retriever 返回来源
        sources = []

        return AskResponse(answer=answer, sources=sources)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"问答处理失败: {str(e)}"
        )


# -----------------------------------------------------------
# 第四步：健康检查接口（可选，但对部署很有用）
#
# 负载均衡器、Docker 健康检查、Kubernetes 探活等都会用到
# -----------------------------------------------------------

@app.get("/health")
async def health():
    """简单的健康检查接口"""
    return {
        "status": "ok" if chain_ready else "not_ready",
        "version": "1.0.0"
    }
