"""
第06章 - RAG检索增强生成：向量存储 RAG（Chroma）
=================================================
本示例演示使用 Chroma 向量数据库构建真正的语义 RAG 系统。

与简单关键词搜索不同，向量搜索基于语义相似度，
能找到含义相近但用词不同的文档。

RAG 流程：
1. 文档 → 文本嵌入（Embeddings）→ 向量
2. 向量存入 Chroma 索引
3. 查询 → 查询嵌入 → 在 Chroma 中找最近邻 → 相关文档
4. 文档 + 查询 → 构建提示 → LLM → 回答

学习要点：
1. 使用百炼 API 的嵌入模型（text-embedding-v4）
2. 创建 Chroma 向量存储
3. 语义相似度搜索
4. 完整的 RAG 链（使用 LCEL）

依赖安装（所有主流平台均支持，包括 macOS Intel）：
    uv sync --extra rag
"""

import os
import sys

try:
    from langchain_chroma import Chroma
except ImportError:
    print("错误：请先安装向量存储依赖：")
    print("  uv sync --extra rag")
    sys.exit(1)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# text-embedding-v4 模型的嵌入向量维度
# 支持动态维度：64, 128, 256, 512, 768, 1024（默认）, 1536, 2048
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
    """
    创建嵌入模型（使用百炼 text-embedding-v4）。
    嵌入模型将文本转换为高维向量，语义相似的文本向量距离近。
    text-embedding-v4 支持动态维度：64/128/256/512/768/1024/1536/2048。
    """
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
        model="text-embedding-v4",                      # 百炼最新嵌入模型
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
        dimensions=EMBEDDING_DIMENSIONS,
        check_embedding_ctx_length=False,
    )


# ============================================================
# 示例文档（AI/ML 知识库）
# ============================================================

DOCUMENTS = [
    Document(
        page_content="""
机器学习是人工智能的一个子领域，专注于开发能够从数据中学习和改进的算法。
主要类型包括：
- 监督学习：使用带标签的数据训练模型（如图像分类、垃圾邮件检测）
- 无监督学习：从无标签数据中发现模式（如聚类、降维）
- 强化学习：通过奖惩机制训练智能体（如游戏AI、机器人控制）
机器学习的关键步骤：数据收集→特征工程→模型训练→评估→部署
""",
        metadata={"source": "ml_intro.txt", "topic": "机器学习基础"}
    ),
    Document(
        page_content="""
深度学习是机器学习的子集，使用多层神经网络（深度神经网络）进行学习。
核心架构：
- CNN（卷积神经网络）：擅长图像处理，通过卷积层提取特征
- RNN/LSTM：处理序列数据，适合自然语言处理和时间序列
- Transformer：基于自注意力机制，是现代LLM的基础架构
- GAN（生成对抗网络）：生成图像、音频等内容
深度学习需要大量数据和计算资源（GPU），但效果远超传统机器学习。
""",
        metadata={"source": "dl_intro.txt", "topic": "深度学习"}
    ),
    Document(
        page_content="""
大语言模型（LLM）是基于Transformer架构的超大规模语言模型。
代表性模型：GPT系列(OpenAI)、Qwen系列(阿里)、Llama系列(Meta)、Claude(Anthropic)
LLM的训练阶段：
1. 预训练：在海量文本上预测下一个词（自监督学习）
2. 指令微调（SFT）：在指令数据集上进行监督微调
3. RLHF：人类反馈的强化学习，对齐人类偏好
LLM的能力：对话、写作、代码生成、推理、翻译、摘要等
上下文窗口（Context Window）：模型一次能处理的最大文本长度。
""",
        metadata={"source": "llm_intro.txt", "topic": "大语言模型"}
    ),
    Document(
        page_content="""
RAG（Retrieval-Augmented Generation，检索增强生成）是一种将信息检索与文本生成结合的技术。
RAG解决了LLM的两个核心问题：
1. 知识截止：LLM训练数据有时间限制，RAG可提供最新信息
2. 幻觉问题：LLM可能编造信息，RAG让模型基于真实文档回答

RAG的核心组件：
- 文档加载器：读取PDF、网页、数据库等各类文档
- 文本分割器：将长文档切成适合嵌入的小块（chunks）
- 嵌入模型：将文本转为向量（如 text-embedding-v4）
- 向量数据库：存储和检索向量（Chroma、FAISS、Pinecone）
- 生成器：LLM基于检索到的文档生成答案
""",
        metadata={"source": "rag_intro.txt", "topic": "RAG技术"}
    ),
    Document(
        page_content="""
向量数据库是专门用于存储和检索高维向量的数据库系统。
工作原理：将文本/图像等转换为向量，通过相似度（余弦相似度、欧氏距离）搜索最相近的向量。

主流向量数据库对比：
- Chroma：开源、纯Python、易用、支持持久化、适合开发学习
- FAISS（Meta）：高性能、纯内存、适合中小规模、Linux/macOS14+/Windows
- Pinecone：云服务、全托管、高可用、适合生产环境
- Milvus：开源分布式、支持十亿级向量、企业级功能

选择建议：
- 学习/原型：Chroma（全平台兼容，无额外依赖）
- 高性能本地：FAISS（需 Linux/macOS14+/Windows）
- 生产环境：Pinecone 或 Milvus
""",
        metadata={"source": "vectordb.txt", "topic": "向量数据库"}
    ),
    Document(
        page_content="""
LangChain是一个用于开发LLM应用的开源框架，提供了：
- Prompt模板：结构化的提示管理
- 链（Chain）：通过LCEL（|操作符）组合多个组件
- 工具（Tools）：让LLM调用外部API和函数
- 记忆（Memory）：维护对话历史
- Agent：自主决策的智能体
- RAG：检索增强生成的完整工具链

LCEL（LangChain Expression Language）：
使用 | 操作符将组件串联：prompt | llm | parser
支持 invoke（同步）、stream（流式）、batch（批量）调用
""",
        metadata={"source": "langchain_intro.txt", "topic": "LangChain框架"}
    ),
]


def build_vector_store(documents: list[Document], embeddings: OpenAIEmbeddings) -> Chroma:
    """
    从文档列表构建 Chroma 向量存储。

    步骤：
    1. 文本分割（将长文档切成小块）
    2. 嵌入计算（调用 API 将文本转为向量）
    3. 构建 Chroma 索引（内存模式，无需本地文件）
    """
    print("正在构建向量存储...")

    # 步骤1：文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,                                 # 每块最大字符数
        chunk_overlap=50,                               # 相邻块的重叠字符数
        separators=["\n\n", "\n", "。", "，", " ", ""], # 优先按段落分割
    )

    chunks = text_splitter.split_documents(documents)
    print(f"  文档数量：{len(documents)} → 分割后块数：{len(chunks)}")

    # 步骤2 + 3：计算嵌入并构建 Chroma 索引（内存模式）
    print("  正在计算嵌入向量（调用百炼 API）...")
    vectorstore = Chroma.from_documents(chunks, embeddings)

    print(f"  向量存储构建完成！共索引 {len(chunks)} 个文本块")
    return vectorstore


def demo_similarity_search(vectorstore: Chroma):
    """
    演示向量相似度搜索。
    展示语义搜索的强大之处：不需要精确匹配关键词。
    """
    print("\n=== 语义相似度搜索演示 ===\n")

    queries = [
        "神经网络有哪些类型？",                          # 应该找到深度学习文档
        "怎么解决模型知识过时的问题？",                  # 应该找到 RAG 文档
        "向量检索数据库哪个好用？",                      # 应该找到向量数据库文档
    ]

    for query in queries:
        print(f"查询：{query}")
        results = vectorstore.similarity_search(query, k=2)   # 返回最相似的2个块

        for i, doc in enumerate(results, 1):
            topic = doc.metadata.get("topic", "未知")
            preview = doc.page_content.strip()[:100]
            print(f"  [{i}] 来源：{topic}")
            print(f"       内容预览：{preview}...")
        print()


def build_rag_chain(vectorstore: Chroma, llm: ChatOpenAI):
    """
    构建完整的 RAG 链（使用 LCEL）。
    """
    # 创建检索器
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}                          # 每次检索返回3个最相关文档
    )

    # RAG 提示模板
    rag_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """你是一个 AI 技术专家助手。请基于以下检索到的文档内容回答用户问题。

回答要求：
1. 只使用提供的文档内容回答，不要添加文档中没有的信息
2. 如果文档中没有相关信息，请明确说明
3. 回答要简洁、准确、有条理

===== 检索到的文档 =====
{context}
========================""",
        ),
        ("human", "{question}"),
    ])

    def format_docs(docs: list[Document]) -> str:
        """将文档列表格式化为字符串。"""
        parts = []
        for i, doc in enumerate(docs, 1):
            topic = doc.metadata.get("topic", "")
            parts.append(f"[文档{i} - {topic}]\n{doc.page_content.strip()}")
        return "\n\n".join(parts)

    # 构建 RAG 链（LCEL 语法）
    rag_chain = (
        {
            # 检索上下文：将问题发送给检索器，格式化返回的文档
            "context": retriever | format_docs,
            # 传递原始问题
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def demo_rag_qa(rag_chain, questions: list[str]):
    """演示 RAG 问答。"""
    print("\n=== RAG 问答演示 ===\n")

    for question in questions:
        print(f"问：{question}")
        answer = rag_chain.invoke(question)
        print(f"答：{answer}")
        print("-" * 50)


def main():
    llm = create_llm()
    embeddings = create_embeddings()

    # 构建向量存储
    vectorstore = build_vector_store(DOCUMENTS, embeddings)

    # 演示相似度搜索
    demo_similarity_search(vectorstore)

    # 构建 RAG 链
    rag_chain = build_rag_chain(vectorstore, llm)

    # 测试 RAG 问答
    questions = [
        "什么是大语言模型？它是如何训练的？",
        "Chroma 和 Pinecone 有什么区别，分别适合什么场景？",
        "RAG 技术主要解决什么问题？",
        "LangChain 的 LCEL 是什么？",
    ]
    demo_rag_qa(rag_chain, questions)

    print("\n向量存储 RAG 示例运行完成！")


if __name__ == "__main__":
    main()

