"""
第03章 - Chain链式调用：简单链
================================
本示例演示 LangChain LCEL（LangChain Expression Language）的核心：
使用 | 操作符将组件串联成链。

学习要点：
1. LCEL 的基本语法（| 操作符）
2. prompt | llm | parser 的基础链
3. 链的 invoke、stream、batch 调用方式
4. RunnableLambda 自定义步骤
"""

import os
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
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


def demo_basic_chain(llm: ChatOpenAI):
    """
    演示最基础的链：prompt | llm | parser
    这是 LCEL 的核心模式。
    """
    print("=== 1. 基础链：prompt | llm | parser ===\n")

    # 步骤1：创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个简洁的AI助手，每次回答不超过100字。"),
        ("human", "{question}"),
    ])

    # 步骤2：输出解析器（提取字符串内容）
    parser = StrOutputParser()

    # 步骤3：使用 | 操作符组合成链
    # 数据流：question → prompt → messages → llm → AIMessage → parser → string
    chain = prompt | llm | parser

    # 调用链
    result = chain.invoke({"question": "什么是Python的GIL？"})
    print(f"问：什么是Python的GIL？")
    print(f"答：{result}\n")


def demo_chain_operations(llm: ChatOpenAI):
    """
    演示链的三种调用方式：invoke、stream、batch
    """
    print("=== 2. 链的调用方式 ===\n")

    prompt = ChatPromptTemplate.from_messages([
        ("human", "用一句话解释{concept}。"),
    ])
    chain = prompt | llm | StrOutputParser()

    # 方式1：invoke - 同步调用，返回单个结果
    print("【invoke - 同步调用】")
    result = chain.invoke({"concept": "面向对象编程"})
    print(f"结果：{result}\n")

    # 方式2：stream - 流式输出
    print("【stream - 流式输出】")
    print("输出：", end="", flush=True)
    for chunk in chain.stream({"concept": "函数式编程"}):
        print(chunk, end="", flush=True)
    print("\n")

    # 方式3：batch - 批量处理多个输入
    print("【batch - 批量处理】")
    inputs = [
        {"concept": "递归"},
        {"concept": "闭包"},
        {"concept": "装饰器"},
    ]
    results = chain.batch(inputs)                       # 一次处理多个输入
    for inp, res in zip(inputs, results):
        print(f"  {inp['concept']}：{res}")
    print()


def demo_chain_with_lambda(llm: ChatOpenAI):
    """
    演示 RunnableLambda：将自定义 Python 函数集成到链中。
    """
    print("=== 3. 使用 RunnableLambda 添加自定义步骤 ===\n")

    # 自定义函数：对 LLM 输出进行后处理
    def format_output(text: str) -> str:
        """为输出添加格式装饰。"""
        lines = text.strip().split("\n")
        formatted = ["📌 " + line if line.strip() else line for line in lines]
        return "\n".join(formatted)

    def count_words(text: str) -> dict:
        """统计输出的词数和字符数。"""
        return {
            "content": text,
            "word_count": len(text.split()),
            "char_count": len(text),
        }

    prompt = ChatPromptTemplate.from_messages([
        ("human", "列出{topic}的三个要点，每点一行。"),
    ])

    # 在链中添加后处理步骤
    chain = (
        prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(format_output)             # 添加格式装饰
    )

    result = chain.invoke({"topic": "Python最佳实践"})
    print(f"格式化后的输出：\n{result}\n")

    # 链末尾统计词数
    stats_chain = (
        prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(count_words)               # 添加统计步骤
    )

    stats = stats_chain.invoke({"topic": "机器学习的应用"})
    print(f"内容：{stats['content']}")
    print(f"词数：{stats['word_count']}，字符数：{stats['char_count']}\n")


def demo_passthrough(llm: ChatOpenAI):
    """
    演示 RunnablePassthrough：在链中传递原始输入，同时进行其他处理。
    """
    print("=== 4. RunnablePassthrough 传递原始输入 ===\n")

    prompt = ChatPromptTemplate.from_messages([
        ("human", "将以下文本翻译成英文：{text}"),
    ])

    # 构建一个同时返回原文和译文的链
    # RunnableParallel 并行执行多个分支，结果合并为 dict
    chain = RunnableParallel(
        original=RunnablePassthrough(),            # 传递原始输入 {"text": "..."}
        translated=prompt | llm | StrOutputParser(),  # 翻译结果
    )

    result = chain.invoke({"text": "人工智能正在改变世界。"})
    print(f"原文：{result['original']['text']}")
    print(f"译文：{result['translated']}\n")


def main():
    llm = create_llm()

    demo_basic_chain(llm)
    demo_chain_operations(llm)
    demo_chain_with_lambda(llm)
    demo_passthrough(llm)

    print("所有简单链示例运行完成！")


if __name__ == "__main__":
    main()
