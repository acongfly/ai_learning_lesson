"""
第02章 - Prompt工程：Prompt 模板
==================================
本示例演示 LangChain 的提示词模板系统，包括：
- PromptTemplate（简单文本模板）
- ChatPromptTemplate（多角色对话模板）
- 模板变量替换和链式调用

学习要点：
1. 如何创建和使用提示词模板
2. 模板中的变量占位符
3. 将模板与 LLM 组合成链
4. 多角色对话消息的构建
"""

import os
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_openai import ChatOpenAI


def create_llm() -> ChatOpenAI:
    """创建 LangChain ChatOpenAI 实例（连接百炼API）。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误：请设置环境变量 DASHSCOPE_API_KEY")
        sys.exit(1)

    return ChatOpenAI(
        model="qwen-plus",                              # 使用 qwen-plus 模型
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
    )


def demo_prompt_template(llm: ChatOpenAI):
    """
    演示基础 PromptTemplate 的使用。
    PromptTemplate 适用于简单的文本提示，使用 {variable} 定义变量。
    """
    print("=== 1. PromptTemplate 基础模板 ===\n")

    # 创建简单文本模板
    template = PromptTemplate(
        input_variables=["topic", "length"],            # 声明模板中的变量
        template="请用{length}字以内介绍{topic}的基本概念和主要应用。",
    )

    # 方式1：使用 format() 生成提示文本
    prompt_text = template.format(topic="区块链", length="200")
    print("格式化后的提示：")
    print(prompt_text)

    # 方式2：直接与 LLM 组合（LCEL 链式调用）
    chain = template | llm | StrOutputParser()          # prompt | llm | 输出解析器
    result = chain.invoke({"topic": "区块链", "length": "100"})
    print(f"\nLLM 回答：\n{result}\n")


def demo_chat_prompt_template(llm: ChatOpenAI):
    """
    演示 ChatPromptTemplate 的使用。
    ChatPromptTemplate 支持多角色消息，更适合对话场景。
    """
    print("=== 2. ChatPromptTemplate 对话模板 ===\n")

    # 方式1：from_messages 创建（最常用）
    # 元组格式：("角色", "消息模板")
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位专业的{role}，专注于{domain}领域。请用专业且易懂的语言回答问题。"),
        ("human", "{question}"),
    ])

    # 格式化模板，查看生成的消息列表
    messages = chat_prompt.format_messages(
        role="数据科学家",
        domain="机器学习",
        question="什么是过拟合？如何避免？",
    )

    print("格式化后的消息列表：")
    for msg in messages:
        print(f"  [{msg.__class__.__name__}] {msg.content[:50]}...")

    # 与 LLM 组合
    chain = chat_prompt | llm | StrOutputParser()
    result = chain.invoke({
        "role": "数据科学家",
        "domain": "机器学习",
        "question": "什么是过拟合？如何避免？",
    })
    print(f"\n专家回答：\n{result}\n")


def demo_multi_message_template(llm: ChatOpenAI):
    """
    演示包含多条消息的模板（如包含示例对话的模板）。
    """
    print("=== 3. 多消息模板（含示例对话）===\n")

    # 创建包含对话示例的模板
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个将中文翻译成英文的助手。只需给出翻译结果，不需要解释。"),
        ("human", "你好"),                              # 示例：用户输入
        ("ai", "Hello"),                                # 示例：AI 回答
        ("human", "谢谢"),
        ("ai", "Thank you"),
        ("human", "{text}"),                            # 实际用户输入（变量）
    ])

    chain = chat_prompt | llm | StrOutputParser()

    # 测试翻译
    test_texts = ["今天天气真好", "我爱学习人工智能", "祝你生日快乐"]
    for text in test_texts:
        result = chain.invoke({"text": text})
        print(f"中文：{text} → 英文：{result}")

    print()


def demo_partial_template(llm: ChatOpenAI):
    """
    演示 partial 方法：预先填入部分变量，剩余变量后续填入。
    """
    print("=== 4. 部分填充模板（partial）===\n")

    # 创建包含多个变量的模板
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一位{subject}老师，面向{level}学生授课。"),
        ("human", "{question}"),
    ])

    # 预先填入 subject 和 level，创建"专用"模板
    python_beginner_template = template.partial(
        subject="Python编程",
        level="初学者",
    )

    # 调用时只需提供 question
    chain = python_beginner_template | llm | StrOutputParser()
    result = chain.invoke({"question": "什么是变量？"})
    print("Python初学者模板的回答：")
    print(result)


def main():
    llm = create_llm()

    demo_prompt_template(llm)
    demo_chat_prompt_template(llm)
    demo_multi_message_template(llm)
    demo_partial_template(llm)

    print("\n所有 Prompt 模板示例运行完成！")


if __name__ == "__main__":
    main()
