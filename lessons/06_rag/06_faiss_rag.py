"""
第06章 - RAG检索增强生成：FAISS 向量存储 RAG
=============================================
本示例演示如何使用 FAISS 构建本地高速向量检索 RAG 流程，包括：
  1. 基础 FAISS 向量存储创建与 RAG 链构建
  2. FAISS 索引本地保存与加载（生产环境必备）
  3. 多种查询方式（相似度、带分数、MMR）
  4. 两个 FAISS 索引合并（批量分片入库场景）

适用平台：
    ✅ Linux (x86_64 / aarch64)
    ✅ macOS 14+ Apple Silicon (M1/M2/M3)
    ✅ macOS 14+ Intel (macosx_14_0_x86_64)
    ✅ Windows (x86_64)
    ❌ macOS Big Sur (11.x) / Monterey (12.x) / Ventura (13.x) Intel (x86_64)
       — 无 pip wheel，可改用 Milvus：04_milvus_rag.py / 05_milvus_advanced_rag.py

依赖安装：
    uv sync --extra faiss

    macOS Intel 用户若无法安装，可通过 Homebrew 源码编译：
        brew install faiss
    或直接使用 Milvus 替代，功能完全等效：
        uv sync --extra milvus
        uv run python lessons/06_rag/04_milvus_rag.py

运行：
    uv run python lessons/06_rag/06_faiss_rag.py
"""

import os
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    print("错误：请先安装 FAISS 依赖：")
    print("  uv sync --extra faiss")
    print("  # 或：pip install 'faiss-cpu>=1.9.0' 'langchain-community>=0.4.1'")
    print()
    print("⚠️  macOS Intel (Big Sur/Monterey/Ventura, x86_64) 无 faiss-cpu pip wheel。")
    print("   请改用 Milvus：uv sync --extra milvus && uv run python lessons/06_rag/04_milvus_rag.py")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────

# text-embedding-v4 支持动态维度：64/128/256/512/768/1024（默认）/1536/2048
EMBEDDING_DIMENSIONS = 1024

# FAISS 索引本地保存路径（目录，会自动创建）
FAISS_SAVE_DIR = "faiss_index_demo"


# ─────────────────────────────────────────────────────────────
# 初始化工具
# ─────────────────────────────────────────────────────────────

def create_llm() -> ChatOpenAI:
    """创建百炼 API LLM 实例（qwen-plus）。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误：请设置环境变量 DASHSCOPE_API_KEY")
        sys.exit(1)
    return ChatOpenAI(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
    )


def create_embeddings() -> OpenAIEmbeddings:
    """创建百炼嵌入模型（text-embedding-v4，支持动态维度）。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    # dimensions：指定输出向量维度。text-embedding-v4 支持动态维度
    # （64/128/256/512/768/1024/1536/2048），默认 1024。
    #
    # check_embedding_ctx_length=False：关闭 tiktoken 分词。
    # 默认开启时，langchain-openai 会用 tiktoken 将文本转为 token 数组（List[List[int]]）
    # 再发送给 API；但 DashScope 兼容接口只接受字符串输入，收到 token 数组会返回
    # 400 BadRequestError（input.contents is neither str nor list of str）。
    # 关闭后直接发送原始文本字符串，兼容 DashScope。
    return OpenAIEmbeddings(
        model="text-embedding-v4",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
        dimensions=EMBEDDING_DIMENSIONS,
        check_embedding_ctx_length=False,
    )


def format_docs(docs: list[Document]) -> str:
    """将检索文档格式化为上下文字符串（用于 RAG Prompt）。"""
    return "\n\n".join(doc.page_content for doc in docs)


# ─────────────────────────────────────────────────────────────
# 演示文档
# ─────────────────────────────────────────────────────────────

def build_batch_a() -> list[Document]:
    """批次 A：AI 框架与技术类文档。"""
    return [
        Document(
            page_content="LangChain 是用于构建 LLM 应用的框架，提供 Prompt、链、工具与 Agent 等抽象。",
            metadata={"source": "langchain_docs", "batch": "A"},
        ),
        Document(
            page_content="RAG（检索增强生成）的核心是先检索相关文档，再将其拼入提示词进行生成，减少幻觉。",
            metadata={"source": "rag_tutorial", "batch": "A"},
        ),
        Document(
            page_content="向量嵌入（Embedding）将文本映射为高维浮点向量，语义相近的文本向量距离更近。",
            metadata={"source": "embedding_tutorial", "batch": "A"},
        ),
    ]


def build_batch_b() -> list[Document]:
    """批次 B：向量数据库类文档。"""
    return [
        Document(
            page_content="FAISS（Facebook AI Similarity Search）是高性能向量检索库，适合本地离线实验与单机部署。",
            metadata={"source": "faiss_docs", "batch": "B"},
        ),
        Document(
            page_content="Milvus 是开源向量数据库，支持 Lite/Standalone/Distributed 三种部署模式，适合生产化。",
            metadata={"source": "milvus_docs", "batch": "B"},
        ),
        Document(
            page_content="Pinecone 是全托管云向量数据库，无需运维，按用量计费，适合快速上线的团队。",
            metadata={"source": "pinecone_docs", "batch": "B"},
        ),
    ]


# ─────────────────────────────────────────────────────────────
# 演示 1：基础 FAISS RAG 链
# ─────────────────────────────────────────────────────────────

def demo_basic_rag(embeddings: OpenAIEmbeddings, llm: ChatOpenAI) -> FAISS:
    """
    演示：使用 FAISS 创建向量存储并构建 RAG 链
    流程：文档 → 向量化 → FAISS 存储 → 检索 → Prompt → LLM → 答案
    """
    print("\n" + "=" * 60)
    print("演示 1：基础 FAISS RAG 链")
    print("=" * 60)

    documents = build_batch_a() + build_batch_b()

    # 创建 FAISS 向量存储（一次性向量化全部文档）
    vectorstore = FAISS.from_documents(documents, embedding=embeddings)
    print(f"✅ FAISS 向量存储创建完成，共 {len(documents)} 篇文档")

    # 构建检索器（top-k = 2）
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 构建 RAG Prompt
    rag_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是问答助手。请严格依据以下上下文回答问题，如果上下文没有答案，请明确说'不知道'。\n\n上下文：\n{context}",
        ),
        ("human", "{question}"),
    ])

    # LCEL RAG 链：检索 -> 格式化上下文 -> LLM 生成 -> 文本输出
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "FAISS 和 Milvus 分别适合什么场景？",
        "什么是 RAG 技术？",
    ]

    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"\n👤 问：{q}")
        print(f"🤖 答：{answer}")

    return vectorstore


# ─────────────────────────────────────────────────────────────
# 演示 2：保存与加载 FAISS 索引（生产环境必备）
# ─────────────────────────────────────────────────────────────

def demo_save_and_load(vectorstore: FAISS, embeddings: OpenAIEmbeddings) -> FAISS:
    """
    演示：将 FAISS 索引持久化到磁盘，再加载使用

    生产场景：
      - 数据预处理阶段：构建一次 FAISS 索引并 save_local
      - 推理服务阶段：load_local 加载索引，直接查询，无需重新向量化

    allow_dangerous_deserialization=True 说明：
      FAISS 索引使用 pickle 序列化，LangChain 为防止恶意文件反序列化添加了此安全确认。
      只要文件来源可信（你自己生成的），可以安全设为 True。
    """
    print("\n" + "=" * 60)
    print("演示 2：保存与加载 FAISS 索引")
    print("=" * 60)

    save_dir = Path(FAISS_SAVE_DIR)

    # ---- 保存 ----
    vectorstore.save_local(str(save_dir))
    saved_files = list(save_dir.iterdir())
    print(f"✅ FAISS 索引已保存至：{save_dir}/")
    print(f"   生成文件：{[f.name for f in saved_files]}")
    # 通常生成两个文件：
    #   index.faiss — ANN（近似最近邻，Approximate Nearest Neighbor）向量索引
    #   index.pkl   — 文档内容与元数据（pickle 序列化）

    # ---- 加载 ----
    loaded_vectorstore = FAISS.load_local(
        str(save_dir),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,  # 加载你自己生成的可信文件时安全
    )
    print("✅ FAISS 索引加载成功")

    # 验证：加载后仍可正常查询
    query = "向量嵌入是什么？"
    docs = loaded_vectorstore.similarity_search(query, k=2)
    print(f"\n加载验证 —「{query}」：")
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] {doc.page_content[:65]}...")

    return loaded_vectorstore


# ─────────────────────────────────────────────────────────────
# 演示 3：多种查询方式
# ─────────────────────────────────────────────────────────────

def demo_query_methods(vectorstore: FAISS) -> None:
    """
    演示：FAISS 支持的多种查询方式

    ┌──────────────────────────────────┬──────────────────────────────────────────┐
    │ 方法                              │ 说明                                     │
    ├──────────────────────────────────┼──────────────────────────────────────────┤
    │ similarity_search                │ 最基础，返回 Document 列表               │
    │ similarity_search_with_score     │ 返回 (Document, score) 列表，分数为 L2 距离 │
    │ max_marginal_relevance_search    │ MMR 多样性搜索，避免结果重复             │
    │ similarity_search_by_vector      │ 直接用向量查询（用于自定义向量流水线）   │
    └──────────────────────────────────┴──────────────────────────────────────────┘
    """
    print("\n" + "=" * 60)
    print("演示 3：多种查询方式对比")
    print("=" * 60)

    query = "哪些向量数据库适合生产环境使用？"

    # ---- 方式 1：similarity_search（基础）----
    print(f"\n【方式 1】similarity_search（查询：「{query}」，k=2）")
    docs = vectorstore.similarity_search(query, k=2)
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] {doc.page_content[:60]}...")

    # ---- 方式 2：similarity_search_with_score（带 L2 距离）----
    # FAISS 默认使用 L2 距离，分数越小表示越相似
    print(f"\n【方式 2】similarity_search_with_score（L2 距离，越小越相似）")
    docs_scores = vectorstore.similarity_search_with_score(query, k=3)
    for i, (doc, score) in enumerate(docs_scores, 1):
        print(f"  [{i}] L2 距离={score:.4f}  {doc.page_content[:55]}...")

    # ---- 方式 3：MMR 多样性搜索 ----
    print(f"\n【方式 3】max_marginal_relevance_search（MMR，k=3，fetch_k=6）")
    docs_mmr = vectorstore.max_marginal_relevance_search(query, k=3, fetch_k=6)
    for i, doc in enumerate(docs_mmr, 1):
        print(f"  [{i}] {doc.page_content[:60]}...")

    # ---- 方式 4：similarity_search_by_vector（按向量查询）----
    # 应用场景：你已经有现成的向量（如来自上游处理），直接用于检索，避免重复嵌入
    # （此处演示仅做说明，实际向量需要通过嵌入模型生成）
    print(f"\n【方式 4】similarity_search_by_vector（按向量直接查询，仅示意）")
    print("   # embedding_vector = embeddings.embed_query(query)  # 先获取向量")
    print("   # docs = vectorstore.similarity_search_by_vector(embedding_vector, k=2)")


# ─────────────────────────────────────────────────────────────
# 演示 4：合并两个 FAISS 索引
# ─────────────────────────────────────────────────────────────

def demo_merge_indices(embeddings: OpenAIEmbeddings) -> None:
    """
    演示：将两个 FAISS 索引合并为一个
    应用场景：
      - 数据量大，分批向量化后合并
      - 多个独立数据源的文档需要统一检索
      - 增量入库（新批次数据合入主索引）
    注意：FAISS 不支持实时删除单条记录，合并是常用的批量更新手段。
    """
    print("\n" + "=" * 60)
    print("演示 4：合并两个 FAISS 索引")
    print("=" * 60)

    # 分别创建两批文档的 FAISS 索引
    index_a = FAISS.from_documents(build_batch_a(), embedding=embeddings)
    index_b = FAISS.from_documents(build_batch_b(), embedding=embeddings)
    print(f"索引 A：{len(build_batch_a())} 篇文档（AI 框架与技术）")
    print(f"索引 B：{len(build_batch_b())} 篇文档（向量数据库）")

    # 将 index_b 合并进 index_a（index_a 会被原地修改）
    index_a.merge_from(index_b)
    print(f"\n✅ 合并完成")

    # 验证：合并后可以检索到两个索引中的内容
    query = "向量数据库有哪些选择？"
    docs = index_a.similarity_search(query, k=3)
    print(f"\n验证搜索「{query}」（应能检索到来自两批次的文档）：")
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] 批次={doc.metadata.get('batch')}  {doc.page_content[:55]}...")


# ─────────────────────────────────────────────────────────────
# 演示 5：FAISS vs Milvus 对比说明
# ─────────────────────────────────────────────────────────────

def demo_faiss_vs_milvus() -> None:
    """打印 FAISS 与 Milvus 的对比表格，帮助技术选型。"""
    print("\n" + "=" * 60)
    print("演示 5：FAISS vs Milvus 技术选型对比")
    print("=" * 60)
    print("""
┌────────────────┬──────────────────────────────┬────────────────────────────────┐
│ 维度           │ FAISS                         │ Milvus (Lite/Standalone)        │
├────────────────┼──────────────────────────────┼────────────────────────────────┤
│ 部署复杂度     │ ⭐ 最简单，纯 Python 库        │ 略高（Lite 无需服务）           │
│ macOS Intel    │ ❌ 无 pip wheel               │ ✅ 全平台支持                   │
│ 实时增删       │ ❌ 不支持删除单条              │ ✅ 支持增删改                   │
│ 元数据过滤     │ ⚠️  有限支持（需 post-filter） │ ✅ 原生支持，高效                │
│ 持久化         │ ✅ save/load（二进制文件）     │ ✅ 原生持久化（本地 DB 文件）    │
│ 生产服务化     │ ❌ 无内置服务接口              │ ✅ 内置 gRPC/HTTP API           │
│ 横向扩展       │ ❌ 单进程                     │ ✅ Distributed 支持集群         │
│ 适用场景       │ 本地实验、学术研究             │ 开发→生产全流程，推荐           │
└────────────────┴──────────────────────────────┴────────────────────────────────┘
选型建议：
  - 快速验证想法、无需持久化 → FAISS（在支持的平台上）
  - 需要持久化、元数据过滤、生产化 → Milvus（全平台）
  - macOS Intel 用户 → 只能使用 Milvus（04/05 脚本）
    """)


# ─────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────

def main() -> None:
    """运行全部演示。"""
    print("=" * 60)
    print("第06章：FAISS 向量存储 RAG 演示")
    print("=" * 60)
    print("依赖：uv sync --extra faiss")
    print("⚠️  macOS Intel 不支持 faiss-cpu pip wheel，请改用 Milvus。")

    embeddings = create_embeddings()
    llm = create_llm()

    # 演示 1：基础 FAISS RAG 链
    vectorstore = demo_basic_rag(embeddings, llm)

    # 演示 2：保存与加载
    loaded_vs = demo_save_and_load(vectorstore, embeddings)

    # 演示 3：多种查询方式
    demo_query_methods(loaded_vs)

    # 演示 4：合并两个索引
    demo_merge_indices(embeddings)

    # 演示 5：对比说明
    demo_faiss_vs_milvus()

    print("\n" + "=" * 60)
    print("✅ FAISS 全部演示完成！")
    print()
    print("📌 下一步建议：")
    print("  1. 替换 build_batch_a/b() 中的文档，测试你自己的数据")
    print("  2. 参考演示 2，在生产服务中使用 load_local 而非重复向量化")
    print("  3. 数据需要实时增删或元数据过滤时，迁移到 Milvus：04_milvus_rag.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
