# RAG 知识库问答系统

基于 LangChain + 通义千问 + Chroma 的 RAG（检索增强生成）问答系统。上传 PDF 文档作为知识库，即可用自然语言提问，大模型基于文档内容给出回答。

---

## 项目结构

`
├── main.py                  # 建库脚本：PDF → 向量数据库
├── test_rag.py              # 完整测试脚本：建库 → 问答
├── app/
│   ├── config.py            # 所有配置集中管理
│   ├── api.py               # FastAPI Web 服务
│   ├── loaders/
│   │   ├── pdf_loader.py    # 读取 PDF
│   │   └── text_splitter.py # 文档切分
│   ├── embeddings/
│   │   └── embedding_model.py  # 文本 → 向量
│   ├── vectorstore/
│   │   └── chroma_store.py  # 向量数据库：存储 + 检索
│   ├── llm/
│   │   └── qwen_model.py    # 大模型接入（通义千问）
│   └── chains/
│       └── rag_chain.py     # RAG 串联链路
├── data/
│   └── test.pdf             # 示例知识库文件
├── vector_store/            # 向量数据库持久化目录（自动生成）
├── requirements.txt
├── .env                     # API Key 等敏感配置
└── README.md
`

---

## 架构流程

### 两条线

**⑴ 建库线**（一次性，处理知识库文档）

`
PDF 文档
  → pdf_loader.py（逐页读取文本）
  → text_splitter.py（切分成 500 字一段）
  → embedding_model.py（每段转成向量）
  → chroma_store.py（向量存入 Chroma，持久化到磁盘）
`

**⑵ 问答线**（反复使用，Web 服务）

`
用户提问
  → 检索器（问题转向量 → 在向量库中找最相似的 4 段）
  → 将检索结果 + 问题拼入 Prompt 模板
  → 通义千问 LLM 基于上下文生成回答
  → 返回给用户
`

### 三个模块的分工

| 模块 | 职责 | 技术选型 |
|------|------|----------|
| 文档加载 | 读取并切分原始文档 | PyPDFLoader + RecursiveCharacterTextSplitter |
| 向量检索 | 将文本转为向量，做相似度搜索 | DashScopeEmbeddings + ChromaDB |
| 大模型 | 基于检索结果生成回答 | 通义千问 qwen3.7-plus（OpenAI 兼容接口） |

---

## 快速开始

### 1. 安装依赖

`ash
pip install -r requirements.txt
`

### 2. 配置 API Key

在 .env 文件中配置阿里云 DashScope 的 API Key（已配好则跳过）：

`
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
`

API Key 可在 [阿里云百炼平台](https://bailian.console.aliyun.com/) 获取。

> 如果需要切换其他兼容 OpenAI 接口的大模型（如 DeepSeek、GLM），只需修改 DASHSCOPE_BASE_URL、DASHSCOPE_API_KEY 和 config.py 中的 LLM_MODEL 即可。

### 3. 建库（将 PDF 存入向量数据库）

`ash
python main.py
`

这会将 data/test.pdf 中的文本切分、embedding 后存入 ector_store/ 目录。日志会输出切分后的文档块数量。

### 4. 启动 Web 服务

`ash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
`

启动后：

| 地址 | 用途 |
|------|------|
| http://localhost:8000/docs | Swagger 交互式 API 文档 |
| http://localhost:8000/ask | 问答接口（POST） |
| http://localhost:8000/health | 健康检查 |

### 5. 测试问答

`ash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是自然语言处理？"}'
`

返回示例：

`json
{
  "answer": "自然语言处理（NLP）是...",
  "sources": []
}
`

---

## API 文档

### POST /ask

**请求体：**

`json
{
  "question": "你的问题"
}
`

**响应体：**

`json
{
  "answer": "大模型生成的回答",
  "sources": []
}
`

### GET /health

`json
{
  "status": "ok",
  "version": "1.0.0"
}
`

---

## 配置说明

所有配置集中在 pp/config.py：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| CHUNK_SIZE | 500 | 每个文档块的最大字数 |
| CHUNK_OVERLAP | 100 | 相邻块的重叠字数，防止割裂上下文 |
| RETRIEVAL_TOP_K | 4 | 每次检索返回的最相似文档块数量 |
| LLM_MODEL | qwen3.7-plus | 通义千问模型版本 |
| VECTOR_DB_PATH | ./vector_store | 向量数据库持久化路径 |

---

## 扩展方向

- **上传接口**：增加 POST /upload 支持上传新 PDF 并自动建库
- **流式输出**：接入 Server-Sent Events（SSE），让大模型逐字返回
- **前端界面**：对接一个简单的对话页面
- **多格式支持**：扩展 Word、Markdown、网页等文档格式
- **增量更新**：只处理新增文档，不重复全量建库
- **会话管理**：支持多轮对话历史
