"""
第06章 - RAG检索增强生成：Milvus 高级查询与文档管理
====================================================
本示例演示 Milvus 向量存储的高级特性，包括：
  1. 带元数据的文档管理（添加/删除）
  2. 多种查询方式（相似度搜索、带分数、MMR、元数据过滤）
  3. Milvus Lite（本地文件）vs Standalone（远程服务）连接切换
  4. 生产环境最佳实践

依赖安装：
    uv sync --extra milvus

适用平台：
    ✅ macOS Intel (Big Sur / Monterey / Ventura, x86_64)
    ✅ macOS Apple Silicon (M1/M2/M3)
    ✅ Linux (x86_64 / aarch64)
    ✅ Windows

运行：
    uv run python lessons/06_rag/05_milvus_advanced_rag.py
"""

import os
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# ── pkg_resources 兼容垫片 ────────────────────────────────────────────────────
# milvus_lite/__init__.py 仅使用 `from pkg_resources import DistributionNotFound,
# get_distribution`（来自 setuptools）来读取自身版本号。
# uv 默认创建的精简虚拟环境不包含 setuptools，因此注入一个基于 Python 标准库
# importlib.metadata（Python 3.8+ 内置）的最小兼容垫片，让 milvus_lite 正常导入，
# 无需要求用户额外安装 setuptools。
try:
    import pkg_resources  # noqa: F401
except ImportError:
    import types as _types
    import importlib.metadata as _meta
    _pkg = _types.ModuleType("pkg_resources")
    _pkg.DistributionNotFound = _meta.PackageNotFoundError
    class _Distribution:
        __slots__ = ("version",)
        def __init__(self, name: str) -> None:
            self.version = _meta.version(name)
    def _get_distribution(name: str) -> _Distribution:
        try:
            return _Distribution(name)
        except _meta.PackageNotFoundError:
            raise _meta.PackageNotFoundError(name)
    _pkg.get_distribution = _get_distribution
    sys.modules["pkg_resources"] = _pkg
# ─────────────────────────────────────────────────────────────────────────────

try:
    from langchain_milvus import Milvus
except ImportError as e:
    print(f"错误：无法导入 langchain_milvus：{e}")
    print("解决方法：cd ai-agent-test && uv sync --extra milvus")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────

# text-embedding-v4 支持动态维度：64/128/256/512/768/1024（默认）/1536/2048
EMBEDDING_DIMENSIONS = 1024

# Milvus Lite 本地数据库文件路径（无需启动任何服务）
MILVUS_LOCAL_URI = "milvus_advanced_demo.db"

# 远程 Milvus Standalone / Distributed 连接（注释掉表示不使用）
# MILVUS_REMOTE_URI = "http://127.0.0.1:19530"

COLLECTION_NAME = "chapter06_milvus_advanced"


# ─────────────────────────────────────────────────────────────
# 初始化工具
# ─────────────────────────────────────────────────────────────

def create_embeddings() -> OpenAIEmbeddings:
    """创建百炼嵌入模型（DashScope 兼容 OpenAI 接口）。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误：请设置环境变量 DASHSCOPE_API_KEY")
        sys.exit(1)
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


# ─────────────────────────────────────────────────────────────
# 演示文档（带丰富元数据，用于演示元数据过滤）
# ─────────────────────────────────────────────────────────────

def build_documents() -> list[Document]:
    """构建带元数据的演示文档库。"""
    return [
        Document(
            page_content="LangChain 是一个用于构建 LLM 应用的框架，支持 Prompt、链、工具与 Agent。",
            metadata={"source": "langchain_docs", "category": "framework", "year": 2023},
        ),
        Document(
            page_content="Milvus 是开源向量数据库，支持 HNSW、IVF_FLAT 等索引，擅长海量向量高速检索。",
            metadata={"source": "milvus_docs", "category": "database", "year": 2023},
        ),
        Document(
            page_content="RAG（检索增强生成）的核心是先检索后生成：将相关文档拼入提示词以减少模型幻觉。",
            metadata={"source": "rag_tutorial", "category": "technique", "year": 2024},
        ),
        Document(
            page_content="FAISS 是 Facebook 开源的向量检索库，适合本地离线实验，但不支持实时增删数据。",
            metadata={"source": "faiss_docs", "category": "library", "year": 2023},
        ),
        Document(
            page_content="Milvus 支持三种部署模式：Lite（本地文件）、Standalone（单机服务）、Distributed（分布式集群）。",
            metadata={"source": "milvus_docs", "category": "database", "year": 2024},
        ),
        Document(
            page_content="向量嵌入（Embedding）将文本转化为高维浮点向量，语义相近的文本向量距离更近。",
            metadata={"source": "embedding_tutorial", "category": "technique", "year": 2024},
        ),
    ]


# ─────────────────────────────────────────────────────────────
# 演示 1：初始化 Milvus 向量存储
# ─────────────────────────────────────────────────────────────

def demo_init_vectorstore(embeddings: OpenAIEmbeddings) -> Milvus:
    """
    演示：初始化 Milvus 向量存储

    连接模式选择：
      - Milvus Lite：uri="./xxx.db"，单文件，无需服务，开发阶段首选
      - Milvus Standalone：uri="http://127.0.0.1:19530"，需要 Docker 启动服务
      - Milvus Distributed：uri + token，适用于生产集群
    """
    print("\n" + "=" * 60)
    print("演示 1：初始化 Milvus 向量存储（Milvus Lite 模式）")
    print("=" * 60)

    documents = build_documents()

    try:
        # 初始化并存入文档（第一次运行时写入 DB 文件）
        vectorstore = Milvus.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={"uri": MILVUS_LOCAL_URI},
            # drop_old=True  # 取消注释可清空已有数据，每次重建
        )
    except Exception as e:
        err = str(e)
        err_repr = repr(e)
        print("❌ Milvus 初始化失败")
        print(f"   错误类型：{type(e).__name__}")
        if err:
            print(f"   错误信息：{err[:400]}")
        else:
            print(f"   错误详情：{err_repr[:400]}")
        if "pkg_resources" in err or "pkg_resources" in err_repr or "setuptools" in err or "setuptools" in err_repr:
            print("   可能原因：setuptools 未安装或环境不完整。")
            print("   解决方法：cd ai-agent-test && git pull && uv sync --extra milvus")
        elif not err:
            print("   可能原因：pymilvus 与 milvus-lite 版本不兼容，或 gRPC 协议错误。")
            print("   解决方法：cd ai-agent-test && uv sync --extra milvus")
        else:
            print("   解决方法：cd ai-agent-test && uv sync --extra milvus")
        sys.exit(1)

    print(f"✅ 初始化完成：已写入 {len(documents)} 篇文档至 '{MILVUS_LOCAL_URI}'")
    print("   集合名称：", COLLECTION_NAME)
    return vectorstore


# ─────────────────────────────────────────────────────────────
# 演示 2：similarity_search —— 基础相似度搜索
# ─────────────────────────────────────────────────────────────

def demo_similarity_search(vectorstore: Milvus) -> None:
    """
    演示：similarity_search —— 返回最相似的 k 篇文档
    这是最常用的查询方式，直接返回 Document 列表。
    """
    print("\n" + "=" * 60)
    print("演示 2：similarity_search（基础相似度搜索）")
    print("=" * 60)

    query = "Milvus 支持哪些部署模式？"
    # k=3 表示最多返回 3 篇文档
    docs = vectorstore.similarity_search(query, k=3)

    print(f"查询：「{query}」")
    print(f"返回 {len(docs)} 篇文档：")
    for i, doc in enumerate(docs, 1):
        print(f"\n  [{i}] {doc.page_content[:60]}...")
        print(f"      元数据：{doc.metadata}")


# ─────────────────────────────────────────────────────────────
# 演示 3：similarity_search_with_score —— 带相似度分数
# ─────────────────────────────────────────────────────────────

def demo_similarity_search_with_score(vectorstore: Milvus) -> None:
    """
    演示：similarity_search_with_score —— 返回 (Document, score) 元组列表
    score 越小表示向量距离越近，即越相似。
    Milvus 默认使用 IP（内积/点积）或 COSINE 距离，具体取决于索引配置。
    应用场景：当你需要根据分数阈值过滤低质量结果时，使用此方法。
    """
    print("\n" + "=" * 60)
    print("演示 3：similarity_search_with_score（带相似度分数）")
    print("=" * 60)

    query = "如何减少 AI 模型的幻觉问题？"
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=3)

    print(f"查询：「{query}」")
    print(f"返回 {len(docs_and_scores)} 篇文档（含距离分数）：")
    for i, (doc, score) in enumerate(docs_and_scores, 1):
        print(f"\n  [{i}] 距离分数：{score:.4f}（越小越相似）")
        print(f"      内容：{doc.page_content[:60]}...")
        print(f"      元数据：{doc.metadata}")


# ─────────────────────────────────────────────────────────────
# 演示 4：max_marginal_relevance_search（MMR）—— 多样性搜索
# ─────────────────────────────────────────────────────────────

def demo_mmr_search(vectorstore: Milvus) -> None:
    """
    演示：max_marginal_relevance_search（MMR，最大边界相关性）
    MMR 在保证相关性的同时，最大化结果多样性，避免返回大量重复文档。

    参数说明：
      - k: 最终返回的文档数量
      - fetch_k: 预先检索的候选文档数（需 > k），从中选最多样的 k 篇
      - lambda_mult: 多样性权重（0=最多样，1=忽略多样性纯按相似度）
    """
    print("\n" + "=" * 60)
    print("演示 4：max_marginal_relevance_search（MMR 多样性搜索）")
    print("=" * 60)

    query = "向量数据库有哪些选择？"
    docs = vectorstore.max_marginal_relevance_search(
        query,
        k=3,          # 最终返回 3 篇
        fetch_k=6,    # 先取 6 篇候选
        lambda_mult=0.5,  # 平衡相关性与多样性
    )

    print(f"查询：「{query}」（MMR 多样性搜索）")
    print(f"返回 {len(docs)} 篇多样化文档：")
    for i, doc in enumerate(docs, 1):
        print(f"\n  [{i}] {doc.page_content[:65]}...")
        print(f"      来源：{doc.metadata.get('source')}")


# ─────────────────────────────────────────────────────────────
# 演示 5：元数据过滤搜索
# ─────────────────────────────────────────────────────────────

def demo_metadata_filter_search(vectorstore: Milvus) -> None:
    """
    演示：带元数据过滤的相似度搜索
    可以先按元数据字段缩小范围，再做向量相似度搜索，大幅提升精度。
    应用场景：按来源、类别、时间范围等先过滤，再搜索。

    Milvus 过滤语法与 SQL WHERE 子句类似，支持：
      - 相等：category == "database"
      - 不等：year != 2023
      - 范围：year >= 2024
      - IN：category in ["database", "framework"]
      - AND / OR 组合
    """
    print("\n" + "=" * 60)
    print("演示 5：元数据过滤搜索（按类别过滤）")
    print("=" * 60)

    query = "向量检索的技术原理是什么？"

    # 只搜索 category 为 "database" 的文档
    docs = vectorstore.similarity_search(
        query,
        k=3,
        expr='category == "database"',  # Milvus 过滤表达式
    )

    print(f"查询：「{query}」")
    print("过滤条件：category == 'database'")
    print(f"返回 {len(docs)} 篇文档：")
    for i, doc in enumerate(docs, 1):
        print(f"\n  [{i}] {doc.page_content[:65]}...")
        print(f"      类别：{doc.metadata.get('category')}")

    # 组合过滤：year >= 2024 AND category == "technique"
    print("\n--- 组合过滤（year >= 2024 AND category == 'technique'）---")
    docs2 = vectorstore.similarity_search(
        query,
        k=3,
        expr='year >= 2024 and category == "technique"',
    )
    print(f"返回 {len(docs2)} 篇文档：")
    for i, doc in enumerate(docs2, 1):
        print(f"\n  [{i}] {doc.page_content[:65]}...")
        print(f"      年份：{doc.metadata.get('year')}，类别：{doc.metadata.get('category')}")


# ─────────────────────────────────────────────────────────────
# 演示 6：动态添加文档
# ─────────────────────────────────────────────────────────────

def demo_add_documents(vectorstore: Milvus) -> None:
    """
    演示：动态向现有向量存储中添加新文档（实时写入，无需重建索引）
    这是 Milvus 相比 FAISS 的重要优势：支持实时增量入库。
    """
    print("\n" + "=" * 60)
    print("演示 6：动态添加新文档")
    print("=" * 60)

    new_docs = [
        Document(
            page_content="Pinecone 是全托管的云向量数据库，无需运维，按用量计费，适合快速上线。",
            metadata={"source": "pinecone_docs", "category": "database", "year": 2024},
        ),
        Document(
            page_content="向量量化（Product Quantization）可将向量压缩存储，大幅降低内存占用但会损失精度。",
            metadata={"source": "ml_tutorial", "category": "technique", "year": 2024},
        ),
    ]

    # add_documents 会自动为新文档生成向量并写入 Milvus
    ids = vectorstore.add_documents(new_docs)
    print(f"✅ 成功添加 {len(new_docs)} 篇新文档")
    print(f"   生成的文档 ID：{ids}")

    # 验证：检索新加入的文档
    query = "Pinecone 有什么特点？"
    docs = vectorstore.similarity_search(query, k=2)
    print(f"\n验证搜索「{query}」：")
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] {doc.page_content[:65]}...")


# ─────────────────────────────────────────────────────────────
# 演示 7：从现有集合加载（不重新创建）
# ─────────────────────────────────────────────────────────────

def demo_load_existing(embeddings: OpenAIEmbeddings) -> None:
    """
    演示：连接到已有的 Milvus 集合（不重新写入数据）
    生产环境中，数据已经提前入库，查询时只需连接，不需要 from_documents。
    """
    print("\n" + "=" * 60)
    print("演示 7：连接到已有 Milvus 集合（生产环境查询模式）")
    print("=" * 60)

    # 用 Milvus() 构造函数连接已有集合，不重新写入
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"uri": MILVUS_LOCAL_URI},
    )

    query = "什么是向量嵌入？"
    docs = vectorstore.similarity_search(query, k=2)
    print(f"查询：「{query}」（从已有集合加载）")
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] {doc.page_content[:65]}...")

    print("\n✅ 生产环境最佳实践：")
    print("   1. 入库阶段：使用 Milvus.from_documents() 一次性写入")
    print("   2. 查询阶段：使用 Milvus() 构造函数连接已有集合，不重复写入")
    print("   3. Milvus Lite → Standalone 迁移：只需将 uri 改为 'http://host:19530'")


# ─────────────────────────────────────────────────────────────
# 演示 8：三种连接模式对比
# ─────────────────────────────────────────────────────────────

def demo_connection_modes() -> None:
    """
    演示（说明）：Milvus 三种连接模式的代码区别
    注意：此函数只打印说明，不实际连接（远程模式需要服务已启动）。
    """
    print("\n" + "=" * 60)
    print("演示 8：三种连接模式对比（说明）")
    print("=" * 60)

    print("""
┌─────────────────┬───────────────────────────────────────────────┬──────────────────────┐
│ 模式             │ connection_args 示例                          │ 适用场景              │
├─────────────────┼───────────────────────────────────────────────┼──────────────────────┤
│ Milvus Lite      │ {"uri": "./milvus_data.db"}                   │ 本地开发/调试         │
│ （本地文件）     │ 无需启动任何服务，推荐开发阶段使用             │                      │
├─────────────────┼───────────────────────────────────────────────┼──────────────────────┤
│ Standalone       │ {"uri": "http://127.0.0.1:19530"}             │ 单机生产/中等规模    │
│ （Docker 服务）  │ 需要先启动 Docker：                            │                      │
│                  │ docker run -p 19530:19530 milvusdb/milvus      │                      │
├─────────────────┼───────────────────────────────────────────────┼──────────────────────┤
│ Distributed      │ {"uri": "https://milvus-endpoint.cloud",      │ 大规模生产集群        │
│ （分布式/云）    │  "token": "your_token"}                        │                      │
└─────────────────┴───────────────────────────────────────────────┴──────────────────────┘
    """)

    print("代码切换示例（只改 connection_args，其余代码不变）：")
    print("""
  # Milvus Lite（当前脚本使用）
  vectorstore = Milvus(embedding_function=embeddings,
                       connection_args={"uri": "./milvus_data.db"})

  # Standalone（单机 Docker）
  vectorstore = Milvus(embedding_function=embeddings,
                       connection_args={"uri": "http://127.0.0.1:19530"})

  # Zilliz Cloud（Milvus 全托管云服务）
  vectorstore = Milvus(embedding_function=embeddings,
                       connection_args={"uri": "https://your-endpoint.zillizcloud.com",
                                        "token": "your_api_key"})
  """)


# ─────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────

def main() -> None:
    """运行全部演示。"""
    print("=" * 60)
    print("第06章：Milvus 高级 RAG 演示")
    print("=" * 60)
    print("依赖：uv sync --extra milvus")
    print(f"数据库文件：{MILVUS_LOCAL_URI}")

    embeddings = create_embeddings()

    # 清理旧的演示数据（每次重新运行，确保演示一致）
    db_file = Path(MILVUS_LOCAL_URI)
    if db_file.exists():
        db_file.unlink()
        print("\n（已清理上次演示数据，重新写入）")

    # 演示 1：初始化
    vectorstore = demo_init_vectorstore(embeddings)

    # 演示 2：基础相似度搜索
    demo_similarity_search(vectorstore)

    # 演示 3：带分数搜索
    demo_similarity_search_with_score(vectorstore)

    # 演示 4：MMR 多样性搜索
    demo_mmr_search(vectorstore)

    # 演示 5：元数据过滤
    demo_metadata_filter_search(vectorstore)

    # 演示 6：动态添加文档
    demo_add_documents(vectorstore)

    # 演示 7：从已有集合加载
    demo_load_existing(embeddings)

    # 演示 8：连接模式对比说明
    demo_connection_modes()

    print("\n" + "=" * 60)
    print("✅ 全部演示完成！")
    print()
    print("📌 下一步建议：")
    print("  1. 修改 build_documents() 中的数据，测试你自己的文档库")
    print("  2. 尝试调整 expr 过滤条件，体验元数据过滤效果")
    print("  3. 修改 connection_args 切换到 Milvus Standalone 服务")
    print("  4. 参考 04_milvus_rag.py 将本脚本的向量存储接入 RAG 链")
    print("=" * 60)


if __name__ == "__main__":
    main()
