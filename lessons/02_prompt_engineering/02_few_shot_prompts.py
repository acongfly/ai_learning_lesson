"""
第02章 - Prompt工程：Few-Shot 提示
=====================================
本示例演示 Few-Shot 提示技术：通过提供少量示例，
引导 LLM 按照期望的格式和风格生成输出。

学习要点：
1. FewShotPromptTemplate 的使用
2. 设计好的 Few-Shot 示例
3. 示例选择器（动态选择最相关的示例）
4. 在 ChatPromptTemplate 中嵌入 Few-Shot 示例
"""

import os
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
)
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


def demo_basic_few_shot(llm: ChatOpenAI):
    """
    演示基础的 FewShotPromptTemplate 使用。
    通过提供情感分析示例，引导模型按照固定格式输出。
    """
    print("=== 1. 基础 FewShotPromptTemplate：情感分析 ===\n")

    # 定义示例列表（每个示例是一个字典）
    examples = [
        {
            "text": "这个产品太棒了，我非常满意！",
            "sentiment": "正面",
            "confidence": "高"
        },
        {
            "text": "送货太慢了，等了一周才到。",
            "sentiment": "负面",
            "confidence": "高"
        },
        {
            "text": "产品质量还可以，价格稍贵。",
            "sentiment": "中性",
            "confidence": "中"
        },
        {
            "text": "包装精美，但功能一般。",
            "sentiment": "中性",
            "confidence": "中"
        },
    ]

    # 单个示例的格式模板
    example_prompt = PromptTemplate(
        input_variables=["text", "sentiment", "confidence"],
        template="文本：{text}\n情感：{sentiment}\n置信度：{confidence}",
    )

    # 创建 FewShotPromptTemplate
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,                              # 示例列表
        example_prompt=example_prompt,                 # 示例格式模板
        prefix="请分析以下文本的情感，并按照示例格式输出：\n",   # 前缀说明
        suffix="\n文本：{input_text}\n情感：",          # 后缀（包含要分析的变量）
        input_variables=["input_text"],                # 实际需要填入的变量
        example_separator="\n\n",                      # 示例之间的分隔符
    )

    # 测试
    test_texts = [
        "客服态度很好，但产品有些问题。",
        "完全超出预期，强烈推荐！",
    ]

    for text in test_texts:
        formatted = few_shot_prompt.format(input_text=text)
        # 直接调用 LLM（不使用链）
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=formatted)])
        print(f"文本：{text}")
        print(f"分析：{response.content.strip()}")
        print()


def demo_few_shot_chat(llm: ChatOpenAI):
    """
    演示在 ChatPromptTemplate 中使用 FewShotChatMessagePromptTemplate。
    这是更现代的方式，直接使用对话消息格式。
    """
    print("=== 2. FewShotChatMessagePromptTemplate：格式转换 ===\n")

    # 示例：将非正式表达转换为正式商务语言
    examples = [
        {
            "input": "能来一下吗？",
            "output": "敬请问您是否方便来此一叙？",
        },
        {
            "input": "这个方案不行，换个。",
            "output": "恕我直言，此方案存在一定局限性，建议重新审视并制定新方案。",
        },
        {
            "input": "你们服务太差了！",
            "output": "贵公司的服务品质有待进一步提升，希望能得到妥善处理。",
        },
    ]

    # 单个示例的格式
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    # FewShotChatMessagePromptTemplate
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
    )

    # 完整的对话模板（包含 Few-Shot 示例）
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个商务语言助手，将非正式语言转换为正式商务语言。"),
        few_shot_prompt,                                # 嵌入 Few-Shot 示例
        ("human", "{input}"),                          # 实际用户输入
    ])

    chain = final_prompt | llm | StrOutputParser()

    # 测试
    test_inputs = [
        "这个价格太贵了吧？",
        "什么时候能给我答复？",
    ]

    for text in test_inputs:
        result = chain.invoke({"input": text})
        print(f"原文：{text}")
        print(f"商务语言：{result}")
        print()


def demo_few_shot_structured_output(llm: ChatOpenAI):
    """
    演示使用 Few-Shot 引导 LLM 输出结构化格式（JSON）。
    """
    print("=== 3. Few-Shot 引导结构化输出 ===\n")

    # 示例：从文本中提取实体信息，输出 JSON 格式
    examples = [
        {
            "text": "张伟，男，35岁，担任北京科技公司的软件工程师。",
            "output": '{"name": "张伟", "gender": "男", "age": 35, "city": "北京", "job": "软件工程师"}'
        },
        {
            "text": "李华是一名28岁的上海金融分析师。",
            "output": '{"name": "李华", "gender": "未知", "age": 28, "city": "上海", "job": "金融分析师"}'
        },
    ]

    # 构建包含示例的系统提示
    system_content = "从文本中提取人物信息，以JSON格式输出，包含字段：name, gender, age, city, job。\n\n"
    system_content += "示例：\n"
    for ex in examples:
        system_content += f"输入：{ex['text']}\n"
        system_content += f"输出：{ex['output']}\n\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_content),
        ("human", "输入：{text}\n输出："),
    ])

    chain = prompt | llm | StrOutputParser()

    test_texts = [
        "王芳，女，42岁，在深圳担任产品经理。",
        "陈明是广州一家医院的30岁外科医生。",
    ]

    for text in test_texts:
        result = chain.invoke({"text": text})
        print(f"输入：{text}")
        print(f"提取结果：{result}")
        print()


def main():
    llm = create_llm()

    demo_basic_few_shot(llm)
    demo_few_shot_chat(llm)
    demo_few_shot_structured_output(llm)

    print("所有 Few-Shot 示例运行完成！")


if __name__ == "__main__":
    main()
