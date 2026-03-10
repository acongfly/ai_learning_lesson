"""
第02章 - Prompt工程：思维链提示（Chain of Thought）
=====================================================
本示例演示思维链（CoT）提示技术：通过引导模型逐步推理，
显著提升复杂问题（数学、逻辑、多步骤推理）的准确性。

学习要点：
1. 基础 CoT：提供逐步推理的示例
2. Zero-Shot CoT：用"让我们一步一步思考"触发推理
3. 自动思维链（Auto-CoT）
4. 对比有无 CoT 的效果差异
"""

import os
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def create_llm() -> ChatOpenAI:
    """创建 LangChain ChatOpenAI 实例。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误：请设置环境变量 DASHSCOPE_API_KEY")
        sys.exit(1)

    return ChatOpenAI(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
    )


def demo_basic_cot(llm: ChatOpenAI):
    """
    演示基础思维链：通过示例教会模型"如何"逐步推理。
    """
    print("=== 1. 基础思维链（Few-Shot CoT）===\n")

    # 带有推理步骤的 CoT 示例
    cot_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """你是一个擅长逻辑推理的AI助手。解题时请按以下格式一步步思考：

示例1：
问题：小明有5个苹果，给了小红2个，小李又给了小明3个，小明现在有几个苹果？
思考过程：
  第一步：小明初始苹果数 = 5
  第二步：给了小红2个，减去：5 - 2 = 3
  第三步：小李给了3个，加上：3 + 3 = 6
答案：小明现在有 6 个苹果。

示例2：
问题：一家商店的商品原价100元，先打九折，再打八折，最终价格是多少？
思考过程：
  第一步：九折后价格 = 100 × 0.9 = 90 元
  第二步：再打八折 = 90 × 0.8 = 72 元
答案：最终价格是 72 元。

现在请用同样的方式回答以下问题：""",
        ),
        ("human", "问题：{question}"),
    ])

    chain = cot_prompt | llm | StrOutputParser()

    # 测试问题
    questions = [
        "一个水桶能装10升水，现在有3个这样的桶，装了总共18升水，还缺多少升才能装满？",
        "火车以每小时120公里的速度行驶，从A城到B城需要2.5小时，A城到B城的距离是多少公里？",
    ]

    for q in questions:
        print(f"问题：{q}")
        result = chain.invoke({"question": q})
        print(result)
        print()


def demo_zero_shot_cot(llm: ChatOpenAI):
    """
    演示 Zero-Shot CoT：不提供示例，只用"让我们一步一步思考"触发推理。
    这是最简单的 CoT 技巧，由 Wei et al. (2022) 提出。
    """
    print("=== 2. Zero-Shot CoT（零样本思维链）===\n")

    question = "有一个池塘，里面有荷花，荷花每天增长一倍。第30天荷花长满了整个池塘，请问第几天荷花占了池塘的一半？"

    # 不使用 CoT（可能直接给出错误答案）
    simple_prompt = ChatPromptTemplate.from_messages([
        ("human", "{question}"),
    ])
    simple_chain = simple_prompt | llm | StrOutputParser()

    # 使用 Zero-Shot CoT（加上"让我们一步一步思考"）
    cot_prompt = ChatPromptTemplate.from_messages([
        ("human", "{question}\n\n让我们一步一步思考："),  # 关键句！触发推理
    ])
    cot_chain = cot_prompt | llm | StrOutputParser()

    print(f"问题：{question}\n")

    print("【不使用CoT的回答】")
    simple_result = simple_chain.invoke({"question": question})
    print(simple_result)

    print("\n【使用Zero-Shot CoT的回答】")
    cot_result = cot_chain.invoke({"question": question})
    print(cot_result)
    print()


def demo_structured_cot(llm: ChatOpenAI):
    """
    演示结构化思维链：要求模型按照固定格式输出推理步骤。
    """
    print("=== 3. 结构化思维链（Structured CoT）===\n")

    structured_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """你是一个专业的分析师。分析问题时请严格按照以下格式：

**问题理解**：用一句话概括问题要求
**已知条件**：列出所有已知信息
**推理步骤**：
  1. [步骤描述]
  2. [步骤描述]
  ...
**验证**：验证答案是否合理
**最终答案**：[答案]""",
        ),
        ("human", "{problem}"),
    ])

    chain = structured_prompt | llm | StrOutputParser()

    problem = """
    一家公司有员工120人。其中技术部占总人数的40%，市场部占25%，其余为行政部。
    技术部的平均月薪是15000元，市场部是12000元，行政部是8000元。
    请计算公司每月的总工资支出是多少？
    """

    print(f"问题：{problem}")
    result = chain.invoke({"problem": problem})
    print(result)


def demo_cot_for_logic(llm: ChatOpenAI):
    """
    演示逻辑推理中的思维链（解决常见的 LLM 逻辑错误）。
    """
    print("=== 4. 逻辑推理思维链 ===\n")

    logic_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "请逐步分析逻辑关系，不要跳跃推理。每一步都要明确写出推理依据。",
        ),
        ("human", "{logic_problem}"),
    ])

    chain = logic_prompt | llm | StrOutputParser()

    problem = """
    有三个人：A、B、C。
    - A说："B是骗子。"
    - B说："C是骗子。"  
    - C说："A和B都是骗子。"
    已知骗子总是说谎，诚实的人总是说真话。
    请问谁是骗子，谁是诚实人？
    """

    print(f"问题：{problem}")
    result = chain.invoke({"logic_problem": problem})
    print(result)


def main():
    llm = create_llm()

    demo_basic_cot(llm)
    demo_zero_shot_cot(llm)
    demo_structured_cot(llm)
    demo_cot_for_logic(llm)

    print("\n所有思维链示例运行完成！")


if __name__ == "__main__":
    main()
