"""
第01章 - 基础LLM调用：思考模型
=================================
本示例演示如何使用思考模型（Thinking Model），
该模型会展示推理过程（reasoning_content），再给出最终答案（content）。

学习要点：
1. 启用 enable_thinking=True 参数
2. 区分 reasoning_content（思考过程）和 content（最终答案）
3. 思考模型的适用场景（复杂推理、数学、逻辑题）
4. 流式处理思考内容

注意：思考模型需要使用支持该功能的模型，如 qwen3 系列。
"""

import os
import sys

from openai import OpenAI


def create_client() -> OpenAI:
    """创建百炼 API 客户端。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误：请设置环境变量 DASHSCOPE_API_KEY")
        sys.exit(1)

    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def thinking_chat_streaming(client: OpenAI, question: str, model: str = "qwen3-235b-a22b"):
    """
    使用思考模型进行流式对话，同时展示思考过程和最终答案。

    参数:
        client: OpenAI 客户端
        question: 用户问题
        model: 使用的模型名称（需支持思考功能）
    """
    print(f"问题：{question}")
    print("=" * 60)

    # 创建流式请求，开启思考模式
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        stream=True,
        extra_body={
            "enable_thinking": True,                    # 开启思考模式
        },
    )

    thinking_content = ""                               # 累积思考过程
    final_content = ""                                  # 累积最终答案
    is_thinking = False                                 # 当前是否在输出思考过程
    is_answering = False                                # 当前是否在输出最终答案

    for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        # 检查是否有思考内容（reasoning_content）
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            if not is_thinking:
                print("\n💭 思考过程：")
                print("-" * 40)
                is_thinking = True
            print(reasoning, end="", flush=True)        # 实时输出思考过程
            thinking_content += reasoning

        # 检查是否有最终答案（content）
        if delta.content:
            if is_thinking and not is_answering:
                print("\n" + "-" * 40)
                print("\n✅ 最终答案：")
                print("-" * 40)
                is_answering = True
                is_thinking = False
            elif not is_answering:
                print("\n✅ 最终答案：")
                print("-" * 40)
                is_answering = True
            print(delta.content, end="", flush=True)    # 实时输出最终答案
            final_content += delta.content

    print("\n" + "=" * 60)

    # 返回思考内容和最终答案
    return thinking_content, final_content


def thinking_chat_non_streaming(client: OpenAI, question: str, model: str = "qwen3-235b-a22b"):
    """
    使用思考模型进行非流式对话（一次性返回完整结果）。

    注意：非流式模式下，thinking 内容在 message.reasoning_content 中。
    """
    print(f"问题：{question}")
    print("=" * 60)

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        stream=False,                                   # 非流式
        extra_body={
            "enable_thinking": True,
        },
    )

    message = completion.choices[0].message

    # 非流式模式下，思考内容在 reasoning_content 字段中
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        print("\n💭 思考过程：")
        print("-" * 40)
        print(reasoning)
        print("-" * 40)

    print("\n✅ 最终答案：")
    print("-" * 40)
    print(message.content)
    print("=" * 60)

    return reasoning, message.content


def demonstrate_thinking_advantage(client: OpenAI):
    """
    演示思考模型在复杂推理上的优势。
    对比普通模型和思考模型处理同一问题的差异。
    """
    question = "9.11 和 9.8 哪个更大？"

    print("\n=== 对比：普通模型 vs 思考模型 ===\n")

    # 普通模型（不开启思考）
    print("【普通模型（qwen-plus）的回答】")
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": question}],
    )
    print(f"回答：{completion.choices[0].message.content}\n")

    # 思考模型
    print("【思考模型（qwen3-235b-a22b）的回答】")
    thinking_chat_streaming(client, question)


def main():
    client = create_client()

    print("=== 示例1：数学推理（思考模型） ===\n")
    thinking_chat_streaming(
        client,
        "一个班有30名学生，其中60%是女生，女生中有1/3喜欢数学，"
        "男生中有1/2喜欢数学，请问全班有多少人喜欢数学？"
    )

    print("\n\n=== 示例2：逻辑推理 ===\n")
    thinking_chat_streaming(
        client,
        "有三个盒子，分别标有'苹果'、'橙子'、'苹果和橙子'。"
        "但所有标签都贴错了。你只能从一个盒子里取出一个水果，"
        "如何通过最少的操作确定所有盒子里装的是什么？"
    )

    print("\n\n=== 示例3：普通模型 vs 思考模型对比 ===")
    demonstrate_thinking_advantage(client)


if __name__ == "__main__":
    main()
