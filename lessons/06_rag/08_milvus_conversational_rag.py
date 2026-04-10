"""
第06章 - RAG检索增强生成：对话式 RAG（Milvus Lite）
====================================================
本示例是 03_conversational_rag.py 的 Milvus 版本。
功能完全对等：多轮对话、问题重构（Contextualize Question）、带历史的 RAG 链，
向量库替换为 Milvus Lite（本地文件，无需启动任何服务）。

核心问题：多轮对话中，用户问题往往依赖上下文。
示例：
  第1轮：北京有哪些景点？
  第2轮：那里的美食是什么？  ← "那里"指代北京，直接检索会失败

解决方案：问题重构（Contextualize Question）
  第2轮改写 → "北京有哪些特色美食？"  ← 独立完整，可正确检索

对比 03_conversational_rag.py 的差异（仅 2 行）：
  - Chroma.from_documents(...) → Milvus.from_documents(..., connection_args={"uri": DB_PATH})
  - ConversationalRAG 类接受 Milvus 类型，其余代码零改动

依赖安装：
    uv sync --extra milvus

适用平台：
    ✅ macOS Intel (x86_64)
    ✅ macOS Apple Silicon (M1/M2/M3)
    ✅ Linux (x86_64 / aarch64)
    ✅ Windows

运行：
    uv run python lessons/06_rag/08_milvus_conversational_rag.py
"""

import os
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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


# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────

# text-embedding-v4 支持动态维度：64/128/256/512/768/1024（默认）/1536/2048
EMBEDDING_DIMENSIONS = 1024
MILVUS_DB_PATH = "milvus_conversational_rag.db"
COLLECTION_NAME = "chapter06_conversational_rag"


# ─────────────────────────────────────────────────────────────
# 初始化：LLM + 嵌入模型
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# 知识库文档（旅游攻略，与 03_conversational_rag.py 相同）
# ─────────────────────────────────────────────────────────────

TRAVEL_DOCS = [
    Document(
        page_content="""
北京旅游攻略：
北京是中国的首都，有着3000多年的建城史和800多年的建都史。
必去景点：故宫（世界最大宫殿建筑群）、长城（建议去慕田峪或八达岭）、
天安门广场、颐和园（皇家园林）、圆明园。
最佳旅游季节：春季（3-5月）和秋季（9-11月），气候宜人。
特色美食：北京烤鸭（全聚德、大董）、炸酱面、豆汁、爆肚、卤煮。
交通：地铁覆盖全市，出租车/滴滴方便，高铁连接全国。
住宿：三里屯、王府井、西单周边酒店多。
""",
        metadata={"destination": "北京", "type": "旅游攻略"},
    ),
    Document(
        page_content="""
上海旅游攻略：
上海是中国最大的城市，国际化大都市，融合中西文化。
必去景点：外滩（夜景绝美）、豫园（传统园林）、城隍庙、
新天地（时尚街区）、迪士尼乐园（中国最大）、南京路步行街。
最佳旅游季节：春季（3-5月）和秋季（9-11月）。
特色美食：小笼包（南翔馒头店）、生煎包、排骨年糕、蟹黄汤包、糟货。
交通：地铁四通八达，共享单车方便，浦东机场有直达市区磁悬浮。
住宿：外滩、人民广场、陆家嘴周边选择多。
""",
        metadata={"destination": "上海", "type": "旅游攻略"},
    ),
    Document(
        page_content="""
成都旅游攻略：
成都是四川省会，以美食、熊猫和悠闲生活著称的"天府之国"。
必去景点：成都大熊猫繁育研究基地（早晨8-9点最佳）、
宽窄巷子（历史文化街区）、锦里古街、武侯祠（三国文化）、
都江堰（世界遗产水利工程）、青城山（道教名山）。
最佳旅游季节：全年皆可，春秋最佳，夏季凉爽舒适。
特色美食：火锅（麻辣鲜香）、夫妻肺片、麻婆豆腐、龙抄手、串串香、冒菜。
交通：地铁建设完善，到市内景点方便。
住宿：春熙路、宽窄巷子附近住宿方便。
""",
        metadata={"destination": "成都", "type": "旅游攻略"},
    ),
    Document(
        page_content="""
西安旅游攻略：
西安是陕西省会，古称长安，十三朝古都，"丝绸之路"起点。
必去景点：兵马俑（世界第八大奇迹）、城墙（可骑车绕城）、
大雁塔、大唐不夜城（唐文化主题街区）、华清宫、骊山。
最佳旅游季节：春秋季节（4-6月、9-11月）。
特色美食：肉夹馍、羊肉泡馍、凉皮、臊子面、葫芦鸡、贾三灌汤包。
交通：高铁连接方便，市内地铁和公交覆盖主要景点。
住宿：钟楼、回民街、大雁塔周边选择丰富。
""",
        metadata={"destination": "西安", "type": "旅游攻略"},
    ),
]


# ─────────────────────────────────────────────────────────────
# 对话式 RAG 类
# ─────────────────────────────────────────────────────────────

class ConversationalRAG:
    """
    对话式 RAG 系统（Milvus 版）。

    核心逻辑：
    ┌─────────────────────────────────────────────────────────┐
    │  用户问题 + 历史对话                                      │
    │       ↓                                                  │
    │  问题重构（contextualize_chain）                          │
    │       ↓  生成独立完整的问题                               │
    │  Milvus 向量检索（retriever.invoke）                      │
    │       ↓  返回相关旅游文档                                 │
    │  RAG 回答（answer_chain）                                 │
    │       ↓  结合历史 + 文档生成自然连贯的回复                │
    │  更新对话历史（chat_history）                             │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self, llm: ChatOpenAI, vectorstore: Milvus) -> None:
        self.llm = llm
        self.vectorstore = vectorstore

        # 检索器：每次返回最相关的 2 个文档块
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # 对话历史（HumanMessage / AIMessage 列表）
        self.chat_history: list = []

        # ── 问题重构提示 ──
        # 将用户的上下文依赖问题改写为独立完整的问题，
        # 使其在没有历史的情况下也能被正确检索。
        self.contextualize_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """你的任务是将用户的最新问题改写为一个独立的、完整的问题。
改写时要结合对话历史，使新问题不依赖上下文也能被理解。

规则：
- 如果用户问题已经是独立完整的，直接返回原问题
- 如果用户问题依赖历史（如"那它呢"、"还有哪些"），结合历史改写
- 只输出改写后的问题，不要解释""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),  # 动态插入历史消息
            ("human", "{question}"),
        ])

        # ── RAG 回答提示 ──
        # 结合检索到的文档和对话历史，生成连贯自然的回答
        self.answer_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """你是一个旅游助手，基于检索到的旅游攻略回答问题。
回答要具体、有用、友好。

===== 相关旅游资料 =====
{context}
========================""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),  # 传入历史，保持对话连贯
            ("human", "{question}"),
        ])

        # 构建链对象
        self._build_chains()

    def _build_chains(self) -> None:
        """构建问题重构链和 RAG 回答链。"""

        def format_docs(docs: list[Document]) -> str:
            """将文档列表格式化为可读字符串。"""
            return "\n\n".join(
                f"[{doc.metadata.get('destination', '')}]\n{doc.page_content.strip()}"
                for doc in docs
            )

        # 问题重构链：contextualize_prompt → LLM → 纯文本输出
        self.contextualize_chain = (
            self.contextualize_prompt | self.llm | StrOutputParser()
        )

        # RAG 回答链：answer_prompt → LLM → 纯文本输出
        self.answer_chain = self.answer_prompt | self.llm | StrOutputParser()

        self.format_docs = format_docs

    def _contextualize_question(self, question: str) -> str:
        """
        根据对话历史重构问题。
        - 无历史时：直接返回原问题（第一轮对话不需要重构）
        - 有历史时：调用 LLM 改写，生成独立完整的问题
        """
        if not self.chat_history:
            return question

        # 将历史传入 LLM，改写为独立问题
        contextualized = self.contextualize_chain.invoke({
            "chat_history": self.chat_history,
            "question": question,
        })
        return contextualized

    def chat(self, user_message: str, verbose: bool = True) -> str:
        """
        处理一轮对话，返回 AI 回答。

        参数：
            user_message：用户输入的问题
            verbose：是否打印调试信息（检索结果、问题改写等）
        """
        if verbose:
            print(f"\n👤 用户：{user_message}")

        # 步骤1：根据历史重构问题（解决代词/指代问题）
        contextualized_q = self._contextualize_question(user_message)
        if verbose and contextualized_q != user_message:
            print(f"  [问题重构] → {contextualized_q}")

        # 步骤2：使用改写后的问题在 Milvus 中检索相关文档
        docs = self.retriever.invoke(contextualized_q)
        if verbose:
            destinations = [d.metadata.get("destination", "") for d in docs]
            print(f"  [检索结果] 相关目的地：{destinations}")

        # 步骤3：格式化检索到的文档
        context = self.format_docs(docs)

        # 步骤4：结合历史 + 检索结果生成回答
        # 注意：传给 answer_chain 的 question 用原始问题，检索用改写问题
        answer = self.answer_chain.invoke({
            "chat_history": self.chat_history,
            "question": user_message,
            "context": context,
        })

        # 步骤5：更新对话历史（用于下一轮的问题重构）
        self.chat_history.append(HumanMessage(content=user_message))
        self.chat_history.append(AIMessage(content=answer))

        if verbose:
            print(f"🤖 AI：{answer}")

        return answer

    def reset(self) -> None:
        """清空对话历史，开始新话题。"""
        self.chat_history = []
        print("\n[对话已重置，开始新的话题]\n")


# ─────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────

def main() -> None:
    llm = create_llm()
    embeddings = create_embeddings()

    # 构建旅游知识库向量存储（Milvus Lite）
    print("正在构建旅游知识库向量存储（Milvus Lite）...")
    db_path = Path(MILVUS_DB_PATH)
    if db_path.exists():
        db_path.unlink()   # 每次运行删除旧数据文件，保证演示数据干净

    try:
        vectorstore = Milvus.from_documents(
            documents=TRAVEL_DOCS,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={"uri": MILVUS_DB_PATH},   # Milvus Lite：本地文件模式
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

    print("向量存储构建完成！\n")

    # 创建对话式 RAG（Milvus 版）
    rag = ConversationalRAG(llm, vectorstore)

    print("=" * 60)
    print("旅游助手对话演示（对话式 RAG，Milvus 版）")
    print("=" * 60)

    # ── 对话轮次1：北京 ──
    # 第1轮是独立问题，无需重构
    rag.chat("北京有哪些必去的景点？")
    # 第2轮："那里"依赖上文（北京），会被改写为"北京的特色美食是什么？"
    rag.chat("那里的特色美食是什么？")
    # 第3轮：依赖上文继续追问
    rag.chat("最好什么季节去？")

    print("\n" + "-" * 60)

    # ── 对话轮次2：切换话题到成都 ──
    rag.chat("成都怎么样？有什么特色？")
    # 追问成都熊猫基地
    rag.chat("熊猫基地什么时候去最好？")
    # 跨话题对比（引用了北京和成都，测试历史感知）
    rag.chat("和北京相比，成都有什么优势？")

    # 重置历史，开始新对话
    rag.reset()

    # ── 对话轮次3：使用代词/指代 ──
    # "它"在下一轮会被正确解析为西安
    rag.chat("我想去有历史文化类的景点，哪个城市好？")
    rag.chat("它的交通方便吗？")   # "它"的指代由 LLM 根据上一轮回答确定（如西安或北京），问题重构后可正确检索

    print("\n✅ 对话式 RAG 示例（Milvus 版）运行完成！")
    print(f"   本地数据库文件：{MILVUS_DB_PATH}")


if __name__ == "__main__":
    main()
