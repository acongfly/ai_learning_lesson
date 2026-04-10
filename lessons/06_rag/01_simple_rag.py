"""
第06章 - RAG检索增强生成：简单 RAG（无向量存储）
===================================================
本示例演示 RAG 的基本思想，使用简单的关键词搜索
（不依赖向量数据库），直观展示 RAG 的核心流程。

RAG 流程：
1. 准备知识库（文档集合）
2. 接收用户问题
3. 从知识库中检索相关文档
4. 将文档作为上下文，构建增强的提示
5. 调用 LLM 生成基于上下文的答案

学习要点：
1. RAG 的基本概念和流程
2. 文档的表示和存储
3. 简单的关键词检索
4. 上下文增强的提示构建
5. 对比有无 RAG 的回答质量差异
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


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


# ============================================================
# 知识库（模拟公司内部文档）
# ============================================================

@dataclass
class Document:
    """文档数据结构"""
    title: str          # 文档标题
    content: str        # 文档内容
    category: str       # 分类标签
    keywords: list      # 关键词列表


# 模拟公司内部知识库
KNOWLEDGE_BASE = [
    Document(
        title="公司请假政策",
        category="HR政策",
        keywords=["请假", "年假", "病假", "假期", "休假"],
        content="""
公司请假政策：
1. 年假：工作满1年享有5天年假，满3年7天，满5年10天，最多15天。
2. 病假：凭医院证明可申请病假，每年最多30天带薪病假。
3. 事假：每年有5天带薪事假，超出部分按日薪扣除。
4. 请假流程：提前3天通过OA系统申请，直属上级审批，HR备案。
5. 孕产假：女性员工产假98天，男性陪产假10天。
"""
    ),
    Document(
        title="差旅报销规定",
        category="财务政策",
        keywords=["差旅", "报销", "出差", "交通", "住宿", "餐费"],
        content="""
差旅报销规定：
1. 交通：国内出差优先购买高铁二等座，飞机经济舱需提前审批。
2. 住宿：一线城市上限500元/晚，二线城市350元/晚，三线城市250元/晚。
3. 餐费：100元/天餐费补贴，需提供发票。
4. 报销时限：出差回来后15个工作日内提交报销单据。
5. 超标说明：超出标准需部门经理和财务总监双重审批。
"""
    ),
    Document(
        title="绩效考核制度",
        category="绩效管理",
        keywords=["绩效", "考核", "KPI", "评分", "奖金", "晋升"],
        content="""
绩效考核制度：
1. 考核周期：每季度一次，年底综合评定。
2. 评分标准：S(杰出)、A(优秀)、B(良好)、C(达标)、D(待改进)。
3. 比例限制：S级不超过10%，A级不超过25%，D级不超过5%。
4. 奖金：S级=月薪×4，A级=月薪×3，B级=月薪×2，C级=月薪×1，D级无奖金。
5. 晋升要求：连续2次A级或1次S级可申请晋升，D级连续2次需进入绩效改进计划。
"""
    ),
    Document(
        title="远程办公政策",
        category="工作政策",
        keywords=["远程", "在家", "办公", "工作", "居家", "WFH"],
        content="""
远程办公政策：
1. 资格：入职满6个月、绩效B级及以上的员工可申请远程办公。
2. 频次：每周最多2天远程，具体日期由部门统一协调。
3. 要求：远程期间需保持通讯畅通，参加所有线上会议。
4. 设备：公司提供笔记本电脑，网络费用可报销（上限200元/月）。
5. 安全：需使用VPN连接公司内网，禁止在公共场所处理保密信息。
"""
    ),
    Document(
        title="员工培训体系",
        category="培训发展",
        keywords=["培训", "学习", "技能", "发展", "课程", "证书"],
        content="""
员工培训体系：
1. 入职培训：新员工前2周进行入职培训，涵盖公司文化、业务知识、安全合规。
2. 在线学习：公司订阅了Coursera企业版，员工可免费学习所有课程。
3. 外部培训：每人每年5000元培训预算，需与工作相关，学完需分享总结。
4. 证书奖励：获得认可的专业证书（如AWS、PMP等），公司报销考试费用，并一次性奖励2000元。
5. 导师计划：入职前3个月配备导师，定期1对1指导。
"""
    ),
]


# ============================================================
# 简单关键词检索
# ============================================================

def keyword_search(query: str, documents: list[Document], top_k: int = 3) -> list[Document]:
    """
    基于关键词的简单文档检索。
    计算查询词与文档关键词的匹配分数，返回最相关的文档。

    注意：这是简化版本，真实 RAG 系统使用向量相似度搜索（见下一节）。
    """
    query_words = set(query.lower().split())
    scored_docs = []

    for doc in documents:
        score = 0

        # 检查标题匹配
        for word in query_words:
            if word in doc.title:
                score += 3                                  # 标题匹配权重高

        # 检查关键词匹配
        for kw in doc.keywords:
            if kw in query or query in kw:
                score += 2                                  # 关键词完整匹配

        # 检查内容匹配
        for word in query_words:
            if word in doc.content:
                score += 1                                  # 内容匹配

        if score > 0:
            scored_docs.append((score, doc))

    # 按分数排序，返回 top_k 个
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]


def format_context(documents: list[Document]) -> str:
    """将检索到的文档格式化为上下文字符串。"""
    if not documents:
        return "（未找到相关文档）"

    parts = []
    for i, doc in enumerate(documents, 1):
        parts.append(f"【文档{i}：{doc.title}】\n{doc.content.strip()}")

    return "\n\n".join(parts)


# ============================================================
# RAG 系统
# ============================================================

class SimpleRAG:
    """简单的 RAG 系统（使用关键词搜索）。"""

    def __init__(self, llm: ChatOpenAI, documents: list[Document]):
        self.llm = llm
        self.documents = documents

        # RAG 提示模板
        self.rag_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """你是公司的智能 HR 助手，专门回答员工关于公司政策的问题。

请基于以下检索到的公司文档来回答问题。
如果文档中没有相关信息，请诚实地说"文档中没有找到相关信息"，不要编造内容。

===== 相关文档 =====
{context}
====================""",
            ),
            ("human", "{question}"),
        ])

        self.chain = self.rag_prompt | self.llm | StrOutputParser()

        # 无 RAG 的对照提示（用于对比）
        self.plain_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是公司的智能 HR 助手，回答员工问题。"),
            ("human", "{question}"),
        ])
        self.plain_chain = self.plain_prompt | self.llm | StrOutputParser()

    def answer(self, question: str, verbose: bool = True) -> str:
        """
        使用 RAG 回答问题。
        """
        # 步骤1：检索相关文档
        relevant_docs = keyword_search(question, self.documents, top_k=2)

        if verbose:
            print(f"\n问题：{question}")
            if relevant_docs:
                print(f"检索到 {len(relevant_docs)} 个相关文档：")
                for doc in relevant_docs:
                    print(f"  - {doc.title}")
            else:
                print("未检索到相关文档")

        # 步骤2：格式化上下文
        context = format_context(relevant_docs)

        # 步骤3：生成答案
        answer = self.chain.invoke({"context": context, "question": question})

        return answer

    def compare_with_without_rag(self, question: str):
        """
        对比有无 RAG 的回答质量。
        """
        print(f"\n{'='*60}")
        print(f"问题：{question}")
        print("=" * 60)

        # 无 RAG 回答
        plain_answer = self.plain_chain.invoke({"question": question})
        print(f"\n❌ 无 RAG（LLM 直接回答，可能不准确）：")
        print(plain_answer)

        # 有 RAG 回答
        rag_answer = self.answer(question, verbose=True)
        print(f"\n✅ 有 RAG（基于公司文档）：")
        print(rag_answer)


def main():
    llm = create_llm()
    rag = SimpleRAG(llm, KNOWLEDGE_BASE)

    print("=== 公司 HR 智能问答系统（简单 RAG）===\n")

    # 示例1：对比有无 RAG 的差异
    rag.compare_with_without_rag("我入职3年了，年假有几天？")

    # 示例2：RAG 问答
    print("\n" + "=" * 60)
    questions = [
        "出差住宿的报销标准是多少？",
        "获得AWS认证证书可以报销吗？",
        "绩效考核获得A级有什么奖励？",
        "公司电脑可以带回家远程办公吗？",
    ]

    for q in questions:
        answer = rag.answer(q)
        print(f"\n答案：{answer}\n{'—'*40}")


if __name__ == "__main__":
    main()
