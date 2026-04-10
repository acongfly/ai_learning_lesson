"""
第06章 - RAG检索增强生成：Milvus 向量存储 RAG（Milvus Lite）
==========================================================
本示例演示如何在本地使用 Milvus Lite 构建一个可运行的语义 RAG 流程。

为什么这个示例有价值：
1. FAISS 适合本地单机快速实验，但服务化能力有限。
2. Milvus 支持从本地 Lite 模式平滑迁移到分布式服务，适合后续生产化。
3. 本示例使用本地文件 URI（milvus_demo.db），不需要单独启动 Milvus 服务器。

依赖安装：
    uv sync --extra milvus

说明：
- 本示例会在当前目录创建本地 Milvus Lite 数据文件。
- 如果你在不支持的平台上安装失败，可先学习 01_simple_rag.py。
"""

import os
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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


# text-embedding-v4 支持动态维度：64/128/256/512/768/1024（默认）/1536/2048
EMBEDDING_DIMENSIONS = 1024


def create_llm() -> ChatOpenAI:
    """创建百炼 API ChatOpenAI 实例。"""
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
    """将检索文档格式化为可读上下文字符串。"""
    return "\n\n".join(doc.page_content for doc in docs)


def build_demo_documents() -> list[Document]:
    """准备演示文档。"""
    return [
        Document(page_content="LangChain 是一个用于构建 LLM 应用的框架，支持 Prompt、链、工具与 Agent。"),
        Document(page_content="Milvus 是开源向量数据库，擅长海量向量检索，支持 HNSW、IVF 等索引。"),
        Document(page_content="RAG 的核心是先检索再生成：把相关文档拼接进提示词，减少模型幻觉。"),
        Document(page_content="FAISS 适合本地离线相似检索，Milvus 更适合服务化与分布式检索场景。"),
    ]


def main():
    llm = create_llm()
    embeddings = create_embeddings()
    documents = build_demo_documents()

    # 使用本地 Milvus Lite 文件，确保 demo 在单机可直接运行。
    # 默认不删除已有数据库，避免误删历史数据。
    # 如需重置 demo 数据，请显式设置：
    #   export MILVUS_RESET_DEMO_DB=1
    db_path = Path("milvus_demo.db")
    should_reset = os.getenv("MILVUS_RESET_DEMO_DB", "").strip() == "1"
    if should_reset and db_path.exists():
        print("⚠️ 检测到 MILVUS_RESET_DEMO_DB=1，正在重置本地 Milvus demo 数据文件。")
        db_path.unlink()

    try:
        vectorstore = Milvus.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="chapter06_milvus_demo",
            connection_args={"uri": str(db_path)},
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

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    rag_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是问答助手。请严格依据上下文回答；如果上下文没有答案，请明确说明不知道。\n\n上下文：\n{context}",
        ),
        ("human", "{question}"),
    ])

    # 标准 LCEL RAG 链：检索 -> 拼接 Prompt -> LLM -> 文本输出
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
        "Milvus 和 FAISS 在使用场景上有什么区别？",
        "RAG 的核心流程是什么？",
    ]

    print("=" * 60)
    print("Milvus Lite RAG 示例")
    print("=" * 60)
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"\n👤 问题：{q}")
        print(f"🤖 回答：{answer}")

    print("\n✅ 运行完成：已使用 Milvus Lite 完成向量检索与回答。")


if __name__ == "__main__":
    main()
