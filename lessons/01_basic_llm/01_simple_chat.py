"""
第01章 - 基础LLM调用：简单对话
=================================
本示例演示如何使用 OpenAI SDK 调用阿里云百炼 API，
实现最简单的非流式对话。

学习要点：
1. 配置 OpenAI 客户端连接百炼 API
2. 构造对话消息列表
3. 发起非流式调用
4. 解析并打印响应
"""

import os  # 用于读取环境变量
import sys

from openai import OpenAI  # OpenAI SDK（兼容百炼API）


def create_client() -> OpenAI:
    """
    创建并返回 OpenAI 客户端。
    百炼 API 完全兼容 OpenAI SDK，只需修改 base_url 即可。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误：请设置环境变量 DASHSCOPE_API_KEY")
        print("获取 API Key：https://bailian.console.aliyun.com/")
        sys.exit(1)

    return OpenAI(
        api_key=api_key,                                                    # 从环境变量读取，不要硬编码
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",      # 百炼兼容端点
    )


def simple_chat(client: OpenAI, user_message: str) -> str:
    """
    执行简单的非流式对话。

    参数:
        client: OpenAI 客户端
        user_message: 用户输入的消息

    返回:
        模型的回答字符串
    """
    # 构造消息列表
    # messages 是一个列表，包含对话历史
    messages = [
        {
            "role": "system",                           # 系统消息：定义 AI 的角色和行为
            "content": "你是一个友好、专业的 AI 助手，擅长用简洁清晰的语言回答问题。",
        },
        {
            "role": "user",                             # 用户消息：用户的问题
            "content": user_message,
        },
    ]

    # 调用 API 获取回答
    # 非流式调用会等待模型生成完所有内容后一次性返回
    completion = client.chat.completions.create(
        model="qwen-plus",                              # 使用 qwen-plus 模型（均衡性能）
        messages=messages,                              # 传入消息列表
    )

    # 从响应中提取文本内容
    # completion.choices 是候选回答列表，通常取第一个 [0]
    # .message.content 是回答的文本内容
    return completion.choices[0].message.content


def multi_turn_chat(client: OpenAI):
    """
    演示多轮对话：通过保存历史消息实现上下文记忆。
    """
    print("\n=== 多轮对话示例 ===")
    print("（输入 'quit' 退出）\n")

    # 初始化对话历史
    messages = [
        {
            "role": "system",
            "content": "你是一个有帮助的 AI 助手。",
        }
    ]

    while True:
        user_input = input("你：").strip()
        if user_input.lower() == "quit":
            print("对话结束。")
            break
        if not user_input:
            continue

        # 将用户消息加入历史
        messages.append({"role": "user", "content": user_input})

        # 调用 API（传入完整对话历史）
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
        )

        # 提取回答
        assistant_reply = completion.choices[0].message.content

        # 将 AI 回答也加入历史（用于下一轮的上下文）
        messages.append({"role": "assistant", "content": assistant_reply})

        print(f"\nAI：{assistant_reply}\n")


def show_usage_info(client: OpenAI):
    """
    演示如何获取 Token 使用量信息（用于计费统计）。
    """
    print("\n=== Token 使用量示例 ===")

    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "用一句话介绍Python。"}],
    )

    # 获取 Token 使用量
    usage = completion.usage
    print(f"输入 Token 数：{usage.prompt_tokens}")
    print(f"输出 Token 数：{usage.completion_tokens}")
    print(f"总 Token 数：{usage.total_tokens}")
    print(f"\n回答：{completion.choices[0].message.content}")


def main():
    # 创建客户端
    client = create_client()

    print("=== 示例1：简单问答 ===")
    answer = simple_chat(client, "什么是人工智能？请用三句话简单介绍。")
    print(f"问题：什么是人工智能？\n答案：{answer}")

    print("\n=== 示例2：不同话题问答 ===")
    questions = [
        "Python 和 Java 有什么区别？",
        "解释一下什么是机器学习。",
    ]
    for q in questions:
        print(f"\n问：{q}")
        answer = simple_chat(client, q)
        print(f"答：{answer}")

    # 显示 Token 使用量
    show_usage_info(client)


if __name__ == "__main__":
    main()
