"""
第06章 - 生产级 Milvus RAG 系统
================================
本脚本演示将 Milvus RAG 从实验原型推向真实生产环境时需要关注的全部关键模式，
是第07/08脚本的进阶版，直接可作为生产项目的起始模板。

涵盖的生产模式：
  1. 环境与依赖校验     — 启动前检查 API Key，给出清晰错误提示
  2. 配置常量集中管理   — 所有可调参数在顶部统一定义，一改全生效
  3. 三层文档入库       — ingest_documents() 支持切分/嵌入/存储，兼容首次建库与增量更新
  4. 生产 RAG 链        — 带来源引用 [来源: xxx]、MMR 多样性搜索、可选元数据过滤
  5. 集合生命周期管理   — load_or_create_vectorstore()：已有则复用，没有则新建
  6. 三种连接模式       — Lite（本地文件）/ Standalone（Docker）/ Zilliz Cloud（托管）
  7. 批量入库           — ingest_batch() 分批处理，避免大数据量时内存压力
  8. RAG 质量评估       — evaluate_rag() 自动跑测试集，验证关键词出现率

适用平台：
    ✅ macOS Intel (x86_64)
    ✅ macOS Apple Silicon (M1/M2/M3)
    ✅ Linux (x86_64 / aarch64)
    ✅ Windows

依赖安装：
    uv sync --extra milvus

运行：
    uv run python lessons/06_rag/09_milvus_production_rag.py
"""

import os
import sys
from pathlib import Path

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

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =============================================================================
# 1. 配置常量（生产模式：所有可调参数集中在此，不散落在代码各处）
# =============================================================================

# Milvus 连接 URI（当前使用 Lite 本地文件模式；切换模式仅需改此常量）
MILVUS_URI = "milvus_production_rag.db"

# Milvus Collection 名称（相当于关系型数据库的表名）
COLLECTION_NAME = "chapter06_production_rag"

# 文本切分参数
CHUNK_SIZE = 500        # 每个文本块的最大字符数
CHUNK_OVERLAP = 50      # 相邻块的重叠字符数（保持跨块上下文连贯）

# 检索参数
TOP_K = 3               # 每次检索返回的最相关文档块数量
MMR_FETCH_K = 10        # MMR 初始召回数（从中筛选多样性最高的 TOP_K 个）
MMR_LAMBDA = 0.6        # MMR 相关性权重（0=纯多样性，1=纯相关性；0.6 是均衡值）

# 批量入库参数
INGEST_BATCH_SIZE = 50  # 每批处理的最大文档块数（避免内存压力）

# 嵌入模型参数
EMBEDDING_DIMENSIONS = 1024     # text-embedding-v4 支持 64/128/256/512/768/1024/1536/2048
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# LLM 参数
LLM_MODEL = "qwen-plus"
LLM_TEMPERATURE = 0.3           # RAG 场景低温度保证答案忠实于文档


# =============================================================================
# 2. 连接模式说明（生产部署参考）
# =============================================================================
#
# ── 模式一：Milvus Lite（本地文件，开发 / 学习 / 小规模 POC）──────────────────
#   connection_args = {"uri": "milvus_production_rag.db"}
#   无需启动任何服务，数据存入本地 .db 文件，适合单人开发和课程演示。
#
# ── 模式二：Milvus Standalone（Docker，团队共享 / 测试 / 中小规模生产）──────────
#   connection_args = {"uri": "http://127.0.0.1:19530"}
#   需先启动：docker run -d --name milvus -p 19530:19530 -p 9091:9091 \
#              milvusdb/milvus:latest
#   适合千万级向量，团队内共享同一实例。
#
# ── 模式三：Zilliz Cloud（全托管，大规模 / 无运维）──────────────────────────────
#   connection_args = {
#       "uri": "https://your-cluster.zillizcloud.com",
#       "token": "your_api_key_or_token",
#   }
#   访问 https://zilliz.com/cloud 免费试用，亿级向量无需自维护 Milvus 集群。
#
# 切换方式：只需修改 MILVUS_URI（和可选的 token），其余业务代码零改动。
# =============================================================================


# =============================================================================
# 3. 知识文档（生产中通常从文件/数据库加载，此处内嵌演示）
# =============================================================================

KNOWLEDGE_DOCUMENTS: list[Document] = [
    Document(
        page_content="""大语言模型（Large Language Model，LLM）是基于 Transformer 架构、
经过海量文本预训练的神经网络模型。代表产品包括 GPT-4、Claude 3、Qwen 等。
LLM 的核心能力包括：文本生成、摘要、翻译、代码编写、逻辑推理。
训练方式：大规模无监督预训练 + 指令微调（Instruction Tuning）+ RLHF（基于人类反馈的强化学习）。
LLM 的主要局限：知识截止日期、幻觉（Hallucination）、不了解私有数据。""",
        metadata={"source": "llm_overview.txt", "category": "foundation", "year": 2024},
    ),
    Document(
        page_content="""RAG（Retrieval-Augmented Generation，检索增强生成）是解决 LLM 知识局限的主流方案。
核心思路：在生成回答之前，先从外部知识库检索相关文档，将检索结果作为上下文注入 Prompt。
RAG 解决的三个核心问题：
  1. 知识截止问题 → 实时更新向量库即可扩展知识
  2. 私有数据问题 → 将内部文档向量化后存入向量库
  3. 幻觉问题     → 强制 LLM 基于真实文档作答，减少凭空捏造
RAG 系统的两个阶段：离线索引（切分→嵌入→存储）和在线查询（嵌入问题→检索→生成）。""",
        metadata={"source": "rag_fundamentals.txt", "category": "rag", "year": 2024},
    ),
    Document(
        page_content="""Milvus 是业界领先的开源向量数据库，由 Zilliz 团队开发并于 2019 年开源。
核心优势：
  - 全平台支持（包括 macOS Intel x86_64）
  - 支持亿级向量，毫秒级检索
  - 提供丰富的元数据过滤（SQL-like 表达式）
  - 支持多种索引类型：FLAT、IVF_FLAT、HNSW、DiskANN
部署形态：
  - Milvus Lite：本地文件，无需服务，适合开发学习
  - Milvus Standalone：单机 Docker，适合中小规模生产
  - Milvus Distributed：集群部署，支持水平扩展
  - Zilliz Cloud：全托管服务，零运维负担""",
        metadata={"source": "milvus_guide.txt", "category": "database", "year": 2024},
    ),
    Document(
        page_content="""向量嵌入（Embedding）是 RAG 系统的核心基础设施。
工作原理：将文字转换为高维数字向量（如 1024 维），语义相近的文本在向量空间中距离更近。
主流嵌入模型对比：
  - 阿里百炼 text-embedding-v4：1024维，中英双语，生产可用，按量计费（支持 64/128/256/512/768/1024/1536/2048 维）
  - OpenAI text-embedding-3-large：3072维，业界标杆，效果最优
  - BAAI/bge-m3：1024维，开源免费，数据不出本地，中英文效果优秀
选择建议：数据保密要求高 → bge-m3 本地部署；追求效果且数据可上云 → text-embedding-v4。""",
        metadata={"source": "embedding_guide.txt", "category": "embedding", "year": 2024},
    ),
    Document(
        page_content="""MMR（Maximal Marginal Relevance，最大边际相关性）是一种平衡相关性与多样性的检索策略。
问题背景：普通相似度搜索会返回大量冗余结果（例如 5 个段落都在说同一件事）。
MMR 工作原理：
  1. 先召回 fetch_k 个最相关文档（例如 10 个）
  2. 从中迭代选择：每次选一个"与查询相关但与已选文档不相似"的文档
  3. lambda_mult 参数控制权重：0=纯多样性，1=纯相关性，0.6 是实践中的均衡值
使用场景：知识库话题广泛时（如技术文档涵盖多个模块），MMR 能保证检索结果覆盖不同方面。
LangChain 调用：vectorstore.as_retriever(search_type="mmr", search_kwargs={...})""",
        metadata={"source": "mmr_retrieval.txt", "category": "retrieval", "year": 2024},
    ),
    Document(
        page_content="""LangChain 是 2022 年 10 月由 Harrison Chase 创建的开源 LLM 应用框架。
核心组件：
  - LCEL（LangChain Expression Language）：用 | 操作符串联组件，声明式构建 Pipeline
  - Runnable 接口：所有组件的统一接口，支持 invoke / stream / batch / astream
  - LangSmith：追踪、调试、评估 LLM 应用的可观测性平台
LCEL 示例：
  rag_chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt | llm | StrOutputParser()
  )
适用场景：快速构建 RAG、Agent、多步骤 AI 工作流；对外提供标准化接口（LangServe）。""",
        metadata={"source": "langchain_intro.txt", "category": "framework", "year": 2024},
    ),
    Document(
        page_content="""元数据过滤（Metadata Filtering）是生产级向量检索的关键能力。
应用场景：
  - 按文档来源过滤：只搜索"技术手册"而非全库
  - 按时间过滤：只搜索 2024 年以后的文档
  - 按类别过滤：只在"产品文档"类型中检索
Milvus 过滤语法（SQL-like 表达式）：
  expr='category == "database"'              # 精确匹配
  expr='year >= 2023'                         # 数值范围
  expr='category in ["rag", "embedding"]'    # 多值匹配
  expr='year >= 2023 and category == "rag"'  # 组合条件
注意：被过滤的字段必须在建索引时指定为 metadata 字段，Milvus 会自动建立标量索引。""",
        metadata={"source": "metadata_filtering.txt", "category": "retrieval", "year": 2024},
    ),
    Document(
        page_content="""增量入库（Incremental Ingestion）是区分生产系统与演示原型的重要能力。
演示原型：每次运行都重建整个向量库（drop_old=True），简单但效率低。
生产系统：只入库新增/变更的文档，已有数据保持不变。
Milvus 增量入库 API：
  vectorstore.add_documents(new_docs)   # 直接追加，不影响已有数据
  vectorstore.add_texts(texts, metadatas=[...])  # 批量追加文本
对比 FAISS 的优势：FAISS 不支持原生增量更新，每次都需完整重建 + 手动 save()；
Milvus 天然支持增量，数据自动持久化，对生产环境中频繁更新的知识库至关重要。""",
        metadata={"source": "incremental_ingest.txt", "category": "database", "year": 2024},
    ),
    Document(
        page_content="""Transformer 架构是现代大语言模型的基础，由 Google 于 2017 年在论文
"Attention Is All You Need" 中提出。核心机制是自注意力（Self-Attention）机制，
允许模型在处理序列时并行考虑所有位置的依赖关系，突破了 RNN 的顺序计算瓶颈。
关键组件：多头注意力（Multi-Head Attention）、前馈网络（FFN）、位置编码（Positional Encoding）。
GPT 系列采用 Decoder-Only 架构；BERT 系列采用 Encoder-Only；T5 采用 Encoder-Decoder。
Scaling Law：参数量越大、训练数据越多，模型能力呈幂律提升（OpenAI Scaling Laws 论文）。""",
        metadata={"source": "transformer_arch.txt", "category": "foundation", "year": 2023},
    ),
    Document(
        page_content="""向量数据库选型指南（2024）：
  Milvus：开源，全平台，亿级规模，完整元数据过滤，有 Lite/Standalone/Cloud 三种部署
  Chroma：开源，开发友好，不支持 macOS Intel，适合中小规模
  FAISS：Meta 开源，极速本地检索，无服务开销，不支持增量更新，适合离线批处理
  Pinecone：全托管云服务，运维零负担，按用量计费，适合不想维护基础设施的团队
  Weaviate：开源，原生 GraphQL，内置混合检索（向量 + BM25）
选型决策树：
  需要全平台支持（含旧 macOS）→ Milvus
  团队规模小 + 快速原型 → Chroma（或 Milvus Lite）
  单机超高性能 + 离线场景 → FAISS
  生产云端 + 零运维 → Pinecone 或 Zilliz Cloud""",
        metadata={"source": "vectordb_comparison.txt", "category": "database", "year": 2024},
    ),
]


# =============================================================================
# 4. 基础设施：LLM + 嵌入模型
# =============================================================================

def validate_environment() -> str:
    """
    环境校验（生产第一步）：检查必要环境变量是否存在。
    在做任何 API 调用之前先校验，避免在深层调用栈中才报出"无效 key"的错误。
    返回 API Key 字符串，校验失败则直接退出。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("=" * 60)
        print("❌ 环境变量未设置：DASHSCOPE_API_KEY")
        print("=" * 60)
        print("解决方法：")
        print("  export DASHSCOPE_API_KEY='your_api_key_here'")
        print("获取 API Key：https://bailian.console.aliyun.com/")
        sys.exit(1)
    print(f"✅ DASHSCOPE_API_KEY 已设置（前8位：{api_key[:8]}...）")
    return api_key


def create_embeddings(api_key: str) -> OpenAIEmbeddings:
    """创建嵌入模型（百炼 text-embedding-v4，支持动态维度）。"""
    # dimensions：指定输出向量维度。text-embedding-v4 支持动态维度
    # （64/128/256/512/768/1024/1536/2048），默认 1024。
    #
    # check_embedding_ctx_length=False：关闭 tiktoken 分词。
    # 默认开启时，langchain-openai 会用 tiktoken 将文本转为 token 数组（List[List[int]]）
    # 再发送给 API；但 DashScope 兼容接口只接受字符串输入，收到 token 数组会返回
    # 400 BadRequestError（input.contents is neither str nor list of str）。
    # 关闭后直接发送原始文本字符串，兼容 DashScope。
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=EMBEDDING_BASE_URL,
        api_key=api_key,
        dimensions=EMBEDDING_DIMENSIONS,
        check_embedding_ctx_length=False,
    )


def create_llm(api_key: str) -> ChatOpenAI:
    """创建对话模型（百炼 qwen-plus）。"""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        base_url=EMBEDDING_BASE_URL,
        api_key=api_key,
    )


# =============================================================================
# 5. 集合生命周期管理
# =============================================================================

def load_or_create_vectorstore(
    embeddings: OpenAIEmbeddings,
    documents: list[Document] | None = None,
    reset: bool = False,
) -> Milvus:
    """
    生产模式：加载已有集合，或在首次运行时新建。

    与演示脚本每次都 drop_old=True 不同，生产系统应该：
      - 如果集合已存在 → 直接连接，不重建（保留已有数据）
      - 如果集合不存在 → 用初始文档新建
      - 如果明确要重建（reset=True）→ 删旧建新

    参数：
      embeddings: 嵌入模型实例
      documents:  初始文档（仅在新建时使用；已有集合时忽略）
      reset:      强制重建，会清空所有已有数据（谨慎使用）
    """
    db_path = Path(MILVUS_URI)
    collection_exists = db_path.exists() and db_path.stat().st_size > 0

    if reset:
        print("⚠️  reset=True：将删除旧集合并重建...")
        if db_path.exists():
            db_path.unlink()
        collection_exists = False

    if collection_exists:
        print(f"✅ 检测到已有集合（{MILVUS_URI}），直接加载，跳过重建...")
        # 连接已有集合：使用 Milvus() 构造函数（不传 documents）
        try:
            vectorstore = Milvus(
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
                connection_args={"uri": MILVUS_URI},
            )
            print(f"   集合加载成功：{COLLECTION_NAME}")
            return vectorstore
        except Exception as e:
            print(f"⚠️  加载失败（{type(e).__name__}: {e}），尝试重建...")
            db_path.unlink(missing_ok=True)

    # 新建集合：需要初始文档
    if not documents:
        print("❌ 集合不存在且未提供初始文档，无法新建。")
        sys.exit(1)

    print(f"🆕 集合不存在，正在新建：{COLLECTION_NAME}")
    return _build_vectorstore(documents, embeddings, drop_old=True)


def _build_vectorstore(
    documents: list[Document],
    embeddings: OpenAIEmbeddings,
    drop_old: bool = False,
) -> Milvus:
    """内部辅助：切分文档、计算嵌入、写入 Milvus。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"   文档数：{len(documents)} → 切分后块数：{len(chunks)}")
    print("   正在计算嵌入向量（调用百炼 API）...")

    try:
        vectorstore = Milvus.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={"uri": MILVUS_URI},
            drop_old=drop_old,
        )
    except Exception as e:
        print(f"❌ Milvus 操作失败（{type(e).__name__}）：{str(e)[:400]}")
        sys.exit(1)

    print(f"   ✅ 向量存储就绪，共 {len(chunks)} 个文本块。")
    return vectorstore


# =============================================================================
# 6. 三层文档入库
# =============================================================================

def ingest_documents(
    vectorstore: Milvus,
    documents: list[Document],
    drop_old: bool = False,
) -> int:
    """
    将文档列表切分、嵌入并存入 Milvus。

    参数：
      vectorstore: 已初始化的 Milvus 实例
      documents:   待入库文档（Document 列表）
      drop_old:    True=清空重建，False=增量追加（生产环境常用）

    返回：
      本次入库的文档块数量

    生产 vs 演示 区别：
      演示（07脚本）：每次 drop_old=True，始终重建
      生产（本脚本）：默认增量追加，只有明确需要重建时才传 drop_old=True
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        print("⚠️  入库文档切分后为空，跳过。")
        return 0

    try:
        if drop_old:
            # 重建模式：创建新集合覆盖旧数据
            print(f"   重建模式：写入 {len(chunks)} 个块（drop_old=True）...")
            Milvus.from_documents(
                documents=chunks,
                embedding=vectorstore.embedding_func,
                collection_name=COLLECTION_NAME,
                connection_args={"uri": MILVUS_URI},
                drop_old=True,
            )
        else:
            # 增量追加：Milvus 相比 FAISS 的核心优势——原生支持增量更新
            print(f"   增量追加：写入 {len(chunks)} 个块...")
            vectorstore.add_documents(chunks)
    except Exception as e:
        print(f"❌ 入库失败（{type(e).__name__}）：{str(e)[:300]}")
        return 0

    print(f"   ✅ 成功入库 {len(chunks)} 个文本块")
    return len(chunks)


def ingest_batch(
    vectorstore: Milvus,
    documents: list[Document],
    batch_size: int = INGEST_BATCH_SIZE,
) -> int:
    """
    批量入库（生产推荐）：将大量文档分批处理，避免单次请求过大导致内存压力。

    适用场景：
      - 首次建库时有数万篇文档需要入库
      - 定期全量更新知识库
      - 内存受限的环境（如容器化部署）

    参数：
      vectorstore: 已初始化的 Milvus 实例
      documents:   全量文档列表
      batch_size:  每批文档数量（切分前的原始文档数，非块数）

    返回：
      总入库块数
    """
    total_chunks = 0
    total_batches = (len(documents) + batch_size - 1) // batch_size

    print(f"\n📦 批量入库：共 {len(documents)} 篇文档，每批 {batch_size} 篇，共 {total_batches} 批")

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"   处理第 {batch_num}/{total_batches} 批（{len(batch)} 篇文档）...")
        chunks_added = ingest_documents(vectorstore, batch, drop_old=False)
        total_chunks += chunks_added

    print(f"✅ 批量入库完成，共写入 {total_chunks} 个文本块。")
    return total_chunks


# =============================================================================
# 7. 生产 RAG 链
# =============================================================================

def build_production_rag_chain(vectorstore: Milvus, llm: ChatOpenAI, expr: str | None = None):
    """
    构建生产级 RAG 链，包含：
      - 来源引用：每段上下文都标注 [来源: {source}]，让 LLM 的回答可追溯
      - MMR 多样性搜索：避免返回重复冗余的文档块
      - 可选元数据过滤：通过 expr 参数按 category/year 等字段缩小检索范围

    参数：
      vectorstore: 已加载的 Milvus 向量存储
      llm:         对话模型
      expr:        Milvus 元数据过滤表达式（可选），例如 'category == "rag"'

    返回：
      LCEL RAG 链，支持 .invoke(question) 调用
    """
    # MMR 检索器：平衡相关性与多样性
    retriever_kwargs: dict = {
        "k": TOP_K,
        "fetch_k": MMR_FETCH_K,
        "lambda_mult": MMR_LAMBDA,
    }
    if expr:
        retriever_kwargs["expr"] = expr

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=retriever_kwargs,
    )

    # 生产提示模板：强调忠实于文档，引用来源
    rag_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """你是一名专业的 AI 技术顾问，请严格基于以下检索文档回答用户问题。

回答规范：
1. 只使用检索文档中的信息，不添加文档之外的内容
2. 如果文档中没有相关信息，请明确说明"根据现有文档，暂无相关记录"
3. 回答要有条理，适当分点说明
4. 可以在回答末尾注明"参考来源：xxx"

===== 检索文档 =====
{context}
====================""",
        ),
        ("human", "{question}"),
    ])

    def format_docs_with_source(docs: list[Document]) -> str:
        """
        将文档列表格式化为带来源标注的上下文字符串。
        生产环境的关键实践：每段都注明来源，让 LLM 能在回答中引用，
        同时也便于后期做来源可追溯审计。
        """
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            category = doc.metadata.get("category", "")
            label = f"[来源: {source}]" + (f" [{category}]" if category else "")
            parts.append(f"{label}\n{doc.page_content.strip()}")
        return "\n\n".join(parts)

    rag_chain = (
        {
            "context": retriever | format_docs_with_source,
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# =============================================================================
# 8. RAG 质量评估
# =============================================================================

def evaluate_rag(rag_chain, test_cases: list[dict]) -> dict:
    """
    简易 RAG 质量评估（生产上线前的冒烟测试）。

    评估方式：运行一组测试问题，检查答案中是否包含预期关键词。
    这是最轻量的自动化评估，适合 CI/CD 流水线集成。

    参数：
      rag_chain:  已构建的 RAG 链
      test_cases: 测试用例列表，每个用例包含：
                    question:          测试问题
                    expected_keywords: 答案中应出现的关键词列表（满足其中一个即通过）
                    description:       测试说明

    返回：
      评估报告字典，包含 total/passed/failed/pass_rate
    """
    print("\n" + "=" * 60)
    print("🧪 RAG 质量评估")
    print("=" * 60)

    passed = 0
    failed = 0
    results = []

    for i, case in enumerate(test_cases, 1):
        question = case["question"]
        expected = case["expected_keywords"]
        desc = case.get("description", "")

        print(f"\n[{i}/{len(test_cases)}] {desc}")
        print(f"  问题：{question}")

        try:
            answer = rag_chain.invoke(question)
            answer_lower = answer.lower()

            # 检查是否有任意一个关键词出现在答案中
            hit = any(kw.lower() in answer_lower for kw in expected)

            if hit:
                matched = [kw for kw in expected if kw.lower() in answer_lower]
                print(f"  ✅ 通过（命中关键词：{matched}）")
                passed += 1
            else:
                print(f"  ❌ 失败（未命中任何关键词：{expected}）")
                print(f"     答案摘要：{answer[:150]}...")
                failed += 1

            results.append({
                "question": question,
                "passed": hit,
                "answer_preview": answer[:200],
            })

        except Exception as e:
            print(f"  ❌ 异常（{type(e).__name__}）：{str(e)[:200]}")
            failed += 1
            results.append({"question": question, "passed": False, "error": str(e)})

    total = passed + failed
    pass_rate = passed / total if total > 0 else 0.0

    print("\n" + "-" * 60)
    print(f"📊 评估结果：{passed}/{total} 通过（通过率 {pass_rate:.1%}）")
    if pass_rate >= 0.8:
        print("✅ 整体质量良好（≥ 80%）")
    elif pass_rate >= 0.5:
        print("⚠️  质量一般（50%–80%），建议优化检索参数或扩充知识库")
    else:
        print("❌ 质量不达标（< 50%），请检查嵌入模型、切分策略或知识库内容")

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "results": results,
    }


# =============================================================================
# 9. 演示函数
# =============================================================================

def demo_similarity_search(vectorstore: Milvus) -> None:
    """演示基础相似度搜索（与 07_milvus_vector_store_rag.py 等价）。"""
    print("\n=== 基础相似度搜索 ===\n")
    queries = [
        "向量数据库有哪些产品？",
        "怎么让 AI 回答私有数据的问题？",
        "嵌入模型怎么选？",
    ]
    for q in queries:
        results = vectorstore.similarity_search(q, k=2)
        print(f"查询：{q}")
        for j, doc in enumerate(results, 1):
            src = doc.metadata.get("source", "?")
            print(f"  [{j}] 来源: {src} | {doc.page_content[:80]}...")
        print()


def demo_mmr_search(vectorstore: Milvus) -> None:
    """演示 MMR 多样性搜索：相比普通搜索，结果覆盖更多不同方面。"""
    print("\n=== MMR 多样性搜索 ===\n")
    query = "RAG 系统的关键技术"
    print(f"查询：{query}\n")

    # 普通相似度搜索
    plain_results = vectorstore.similarity_search(query, k=3)
    print("普通搜索（可能有冗余）：")
    for j, doc in enumerate(plain_results, 1):
        src = doc.metadata.get("source", "?")
        cat = doc.metadata.get("category", "")
        print(f"  [{j}] {src} [{cat}]")

    # MMR 多样性搜索
    mmr_results = vectorstore.max_marginal_relevance_search(
        query,
        k=TOP_K,
        fetch_k=MMR_FETCH_K,
        lambda_mult=MMR_LAMBDA,
    )
    print("\nMMR 搜索（多样性更高）：")
    for j, doc in enumerate(mmr_results, 1):
        src = doc.metadata.get("source", "?")
        cat = doc.metadata.get("category", "")
        print(f"  [{j}] {src} [{cat}]")


def demo_metadata_filter(vectorstore: Milvus) -> None:
    """演示元数据过滤：按 category 字段限定检索范围。"""
    print("\n=== 元数据过滤搜索 ===\n")
    query = "什么是主流技术？"

    # 不过滤（全库搜索）
    all_results = vectorstore.similarity_search(query, k=3)
    print(f"不过滤（全库）：")
    for doc in all_results:
        print(f"  - [{doc.metadata.get('category')}] {doc.metadata.get('source')}")

    # 只搜索 database 类别
    try:
        db_results = vectorstore.similarity_search(
            query,
            k=3,
            expr='category == "database"',
        )
        print(f"\n只搜索 category='database'：")
        for doc in db_results:
            print(f"  - [{doc.metadata.get('category')}] {doc.metadata.get('source')}")
    except Exception as e:
        print(f"  （元数据过滤示例跳过：{type(e).__name__}: {e}）")


def demo_incremental_ingest(vectorstore: Milvus) -> None:
    """
    演示增量入库：Milvus 相比 FAISS 的核心优势之一。
    生产场景：每天新增文档时，只追加新内容，不重建整个索引。
    """
    print("\n=== 增量入库演示 ===\n")
    new_docs = [
        Document(
            page_content="""生成式 AI（Generative AI）是 2023 年最火爆的技术方向。
代表产品：ChatGPT（OpenAI）、Claude（Anthropic）、Gemini（Google）、通义千问（阿里）。
核心技术突破：RLHF 对齐训练使 LLM 更好地遵循指令，减少有害输出。
商业化落地：代码辅助（GitHub Copilot）、写作助手、AI 搜索引擎等已大规模商用。
2024 年趋势：多模态（文本+图像+视频）、长上下文（100K+ tokens）、本地小模型（<7B）。""",
            metadata={"source": "genai_trends.txt", "category": "trends", "year": 2024},
        ),
    ]
    count = ingest_documents(vectorstore, new_docs, drop_old=False)
    print(f"✅ 增量追加完成，新增 {count} 个块。Milvus 原有数据完整保留。")


def demo_rag_qa(rag_chain) -> None:
    """演示完整 RAG 问答流程。"""
    print("\n=== 完整 RAG 问答 ===\n")
    questions = [
        "什么是 RAG？它解决了 LLM 的哪些问题？",
        "Milvus 有哪些部署模式，分别适合什么场景？",
        "MMR 检索和普通相似度搜索有什么区别？",
    ]
    for q in questions:
        print(f"问：{q}")
        try:
            answer = rag_chain.invoke(q)
            print(f"答：{answer}")
        except Exception as e:
            print(f"（调用失败：{type(e).__name__}: {e}）")
        print("-" * 50)


def demo_filtered_rag(vectorstore: Milvus, llm: ChatOpenAI) -> None:
    """演示带元数据过滤的 RAG：只从 'rag' 类别文档中检索。"""
    print("\n=== 元数据过滤 RAG ===\n")
    try:
        filtered_chain = build_production_rag_chain(
            vectorstore, llm, expr='category == "rag"'
        )
        q = "增量入库和重建索引有什么区别？"
        print(f"问（限定 category='rag'）：{q}")
        answer = filtered_chain.invoke(q)
        print(f"答：{answer}")
    except Exception as e:
        print(f"（过滤 RAG 示例跳过：{type(e).__name__}: {e}）")


# =============================================================================
# 10. 主函数（完整端到端演示）
# =============================================================================

def main() -> None:
    print("=" * 60)
    print("第06章 - 生产级 Milvus RAG 系统")
    print("=" * 60)

    # ── 步骤1：环境校验（生产第一步，失败则立即退出）──────────────────────
    print("\n【步骤1】环境与依赖校验")
    api_key = validate_environment()

    # ── 步骤2：初始化基础设施 ──────────────────────────────────────────────
    print("\n【步骤2】初始化嵌入模型与 LLM")
    embeddings = create_embeddings(api_key)
    llm = create_llm(api_key)
    print(f"  嵌入模型：{EMBEDDING_MODEL}（{EMBEDDING_DIMENSIONS} 维）")
    print(f"  对话模型：{LLM_MODEL}（temperature={LLM_TEMPERATURE}）")

    # ── 步骤3：加载或新建向量集合 ─────────────────────────────────────────
    print("\n【步骤3】集合生命周期管理（load_or_create）")
    vectorstore = load_or_create_vectorstore(
        embeddings=embeddings,
        documents=KNOWLEDGE_DOCUMENTS,  # 首次建库时使用
        reset=True,                     # 演示时强制重建，生产中改为 False
    )

    # ── 步骤4：演示增量入库 ────────────────────────────────────────────────
    print("\n【步骤4】增量入库（Milvus 相比 FAISS 的核心优势）")
    demo_incremental_ingest(vectorstore)

    # ── 步骤5：基础相似度搜索 ─────────────────────────────────────────────
    print("\n【步骤5】基础相似度搜索")
    demo_similarity_search(vectorstore)

    # ── 步骤6：MMR 多样性搜索 ─────────────────────────────────────────────
    print("\n【步骤6】MMR 多样性搜索")
    demo_mmr_search(vectorstore)

    # ── 步骤7：元数据过滤搜索 ─────────────────────────────────────────────
    print("\n【步骤7】元数据过滤搜索")
    demo_metadata_filter(vectorstore)

    # ── 步骤8：完整生产 RAG 链 ────────────────────────────────────────────
    print("\n【步骤8】构建生产 RAG 链（带来源引用 + MMR）")
    rag_chain = build_production_rag_chain(vectorstore, llm)
    demo_rag_qa(rag_chain)

    # ── 步骤9：带元数据过滤的 RAG ─────────────────────────────────────────
    print("\n【步骤9】带元数据过滤的 RAG")
    demo_filtered_rag(vectorstore, llm)

    # ── 步骤10：RAG 质量评估 ──────────────────────────────────────────────
    print("\n【步骤10】RAG 质量自动评估")
    test_cases = [
        {
            "question": "RAG 系统解决了 LLM 的哪三个核心问题？",
            "expected_keywords": ["知识截止", "幻觉", "私有数据", "hallucination"],
            "description": "验证 RAG 核心价值描述",
        },
        {
            "question": "Milvus 有哪些部署形态？",
            "expected_keywords": ["Lite", "Standalone", "Distributed", "Zilliz"],
            "description": "验证 Milvus 部署形态知识",
        },
        {
            "question": "text-embedding-v4 的向量维度是多少？",
            "expected_keywords": ["1024", "1536", "2048", "64", "128", "256", "512", "768"],
            "description": "验证嵌入模型技术细节",
        },
        {
            "question": "MMR 检索的 lambda_mult 参数含义是什么？",
            "expected_keywords": ["多样性", "相关性", "lambda", "0"],
            "description": "验证 MMR 参数知识",
        },
    ]
    eval_report = evaluate_rag(rag_chain, test_cases)

    # ── 完成 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ 生产级 Milvus RAG 系统演示完成！")
    print("=" * 60)
    print(f"  本地数据库文件：{MILVUS_URI}")
    print(f"  RAG 评估通过率：{eval_report['pass_rate']:.1%} "
          f"（{eval_report['passed']}/{eval_report['total']}）")
    print()
    print("  ── 生产部署下一步 ──────────────────────────────")
    print("  切换到 Milvus Standalone（Docker）：")
    print('    将 MILVUS_URI 改为 "http://127.0.0.1:19530"')
    print("  切换到 Zilliz Cloud（零运维托管）：")
    print('    将 MILVUS_URI 改为 "https://your-cluster.zillizcloud.com"')
    print("    并在 connection_args 中添加 token")
    print("  其余业务代码：零改动 ✅")


if __name__ == "__main__":
    main()
