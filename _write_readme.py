content = """# RAG 知识库问答系统

基于 LangChain + 通义千问 + Chroma 的 RAG（检索增强生成）问答系统。上传 PDF 文档作为知识库，即可用自然语言提问，大模型基于文档内容给出回答。

---

## 项目结构

```
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
```

---

## 架构流程

### 两条线

**（1）建库线**（一次性，处理知识库文档）

```
PDF 文档
  → pdf_loader.py（逐页读取文本）
  → text_splitter.py（切分成 500 字一段）
  → embedding_model.py（每段转成向量）
  → chroma_store.py（向量存入 Chroma，持久化到磁盘）
```

**（2）问答线**（反复使用，Web 服务）

```
用户提问
  → 检索器（问题转向量 → 在向量库中找最相似的 4 段）
  → 将检索结果 + 问题拼入 Prompt 模板
  → 通义千问 LLM 基于上下文生成回答
  → 返回给用户
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 3. 建库

```bash
python main.py
```

### 4. 启动 Web 服务

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

### 5. 测试问答

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "什么是自然语言处理？"}'
```

---

## 配置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| CHUNK_SIZE | 500 | 每个文档块的最大字数 |
| CHUNK_OVERLAP | 100 | 相邻块重叠字数，防止割裂上下文 |
| RETRIEVAL_TOP_K | 4 | 每次检索返回的最相似文档块数 |
| LLM_MODEL | qwen3.7-plus | 通义千问模型版本 |
| VECTOR_DB_PATH | ./vector_store | 向量数据库持久化路径 |
"""
with open("D:/AI/rag_project/README.md", "w", encoding="utf-8") as f:
    f.write(content)
print("done")
