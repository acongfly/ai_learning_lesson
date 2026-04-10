"""
第06章 - RAG检索增强生成：对话式 RAG
========================================
本示例演示多轮对话中的 RAG 系统。
在对话式场景中，需要根据历史对话对问题进行重新表述，
才能准确检索到相关文档。

问题：
"北京今天天气如何？" → 下一轮："那上海呢？"
如果直接搜索"那上海呢"，无法找到相关信息。
需要结合历史改写为："上海今天天气如何？"

学习要点：
1. 多轮对话历史的管理
2. 问题重构（Contextualize Question）
3. 带历史的 RAG 链
4. ChatMessageHistory 的使用

依赖安装（所有主流平台，包括 macOS Intel）：
    uv sync --extra rag
"""

import os
import sys

try:
    from langchain_chroma import Chroma
except ImportError:
    print("错误：请先安装向量存储依赖：")
    print("  uv sync --extra rag")
    print("")
    print("注意：langchain-chroma 通过 chromadb 依赖 onnxruntime，")
    print("      该库不支持 macOS < 14 Intel 系统（Big Sur / Monterey / Ventura，")
    print("      平台标识 macosx_10_16_x86_64）。")
    print("      如需在不支持的平台学习 RAG 基础，请使用 01_simple_rag.py（无向量库依赖）。")
    sys.exit(1)
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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
    """创建嵌入模型（百炼 text-embedding-v4，1024 维）。"""
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


# ============================================================
# 知识库文档
# ============================================================

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
        metadata={"destination": "北京", "type": "旅游攻略"}
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
        metadata={"destination": "上海", "type": "旅游攻略"}
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
        metadata={"destination": "成都", "type": "旅游攻略"}
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
        metadata={"destination": "西安", "type": "旅游攻略"}
    ),
]


class ConversationalRAG:
    """
    对话式 RAG 系统。
    支持多轮对话，通过问题重构实现准确检索。
    """

    def __init__(self, llm: ChatOpenAI, vectorstore: Chroma):
        self.llm = llm
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # 对话历史
        self.chat_history: list = []

        # 问题重构提示：将问题与历史结合，生成独立问题
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
            MessagesPlaceholder(variable_name="chat_history"),  # 插入历史消息
            ("human", "{question}"),
        ])

        # RAG 回答提示
        self.answer_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """你是一个旅游助手，基于检索到的旅游攻略回答问题。
回答要具体、有用、友好。

===== 相关旅游资料 =====
{context}
========================""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),  # 包含历史，保持对话连贯
            ("human", "{question}"),
        ])

        # 构建链
        self._build_chains()

    def _build_chains(self):
        """构建问题重构链和 RAG 回答链。"""

        def format_docs(docs: list[Document]) -> str:
            return "\n\n".join(
                f"[{doc.metadata.get('destination', '')}]\n{doc.page_content.strip()}"
                for doc in docs
            )

        # 问题重构链（当有历史时使用）
        self.contextualize_chain = (
            self.contextualize_prompt | self.llm | StrOutputParser()
        )

        # RAG 回答链
        self.answer_chain = self.answer_prompt | self.llm | StrOutputParser()

        self.format_docs = format_docs

    def _contextualize_question(self, question: str) -> str:
        """
        根据对话历史重构问题。
        如果没有历史，直接返回原问题。
        """
        if not self.chat_history:
            return question

        contextualized = self.contextualize_chain.invoke({
            "chat_history": self.chat_history,
            "question": question,
        })

        return contextualized

    def chat(self, user_message: str, verbose: bool = True) -> str:
        """
        处理一轮对话。
        """
        if verbose:
            print(f"\n👤 用户：{user_message}")

        # 步骤1：重构问题（考虑历史上下文）
        contextualized_q = self._contextualize_question(user_message)

        if verbose and contextualized_q != user_message:
            print(f"  [问题重构] → {contextualized_q}")

        # 步骤2：使用重构后的问题检索文档
        docs = self.retriever.invoke(contextualized_q)

        if verbose:
            destinations = [d.metadata.get("destination", "") for d in docs]
            print(f"  [检索结果] 相关目的地：{destinations}")

        # 步骤3：格式化检索结果
        context = self.format_docs(docs)

        # 步骤4：生成回答（传入历史，保持对话连贯）
        answer = self.answer_chain.invoke({
            "chat_history": self.chat_history,
            "question": user_message,          # 用原始问题，上下文在文档中
            "context": context,
        })

        # 步骤5：更新对话历史
        self.chat_history.append(HumanMessage(content=user_message))
        self.chat_history.append(AIMessage(content=answer))

        if verbose:
            print(f"🤖 AI：{answer}")

        return answer

    def reset(self):
        """重置对话历史。"""
        self.chat_history = []
        print("\n[对话已重置]\n")


def main():
    llm = create_llm()
    embeddings = create_embeddings()

    # 构建向量存储（Chroma，全平台兼容）
    print("正在构建旅游知识库向量存储...")
    vectorstore = Chroma.from_documents(TRAVEL_DOCS, embeddings)
    print("向量存储构建完成！\n")

    # 创建对话式 RAG
    rag = ConversationalRAG(llm, vectorstore)

    print("=" * 60)
    print("旅游助手对话演示（对话式 RAG）")
    print("=" * 60)

    # 对话1：关于北京
    rag.chat("北京有哪些必去的景点？")
    rag.chat("那里的特色美食是什么？")               # 依赖上文（北京）
    rag.chat("最好什么季节去？")                     # 依赖上文（北京）

    print("\n" + "-" * 60)

    # 对话2：切换到成都
    rag.chat("成都怎么样？有什么特色？")
    rag.chat("熊猫基地什么时候去最好？")             # 依赖上文（成都）
    rag.chat("和北京相比，成都有什么优势？")          # 跨越历史的对比

    rag.reset()

    # 对话3：新话题
    rag.chat("我想去有历史文化类的景点，哪个城市好？")
    rag.chat("它的交通方便吗？")                     # 代词"它"需要重构

    print("\n对话式 RAG 示例运行完成！")


if __name__ == "__main__":
    main()
