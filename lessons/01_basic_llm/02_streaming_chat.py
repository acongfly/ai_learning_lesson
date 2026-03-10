"""
第01章 - 基础LLM调用：流式对话
=================================
本示例演示如何使用流式（Streaming）方式调用百炼 API，
实现"打字机"效果的实时输出。

学习要点：
1. 开启 stream=True 进行流式调用
2. 逐块处理响应流（chunk）
3. 处理 stream_options 获取用量统计
4. 流式输出的优势：更低延迟、更好的用户体验
"""

import os
import sys
import time

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


def streaming_chat(client: OpenAI, user_message: str) -> str:
    """
    流式对话：边生成边输出，实现打字机效果。

    参数:
        client: OpenAI 客户端
        user_message: 用户消息

    返回:
        完整的回答文本
    """
    print("AI：", end="", flush=True)

    # 创建流式请求
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": user_message},
        ],
        stream=True,                                    # 开启流式输出
        stream_options={"include_usage": True},         # 包含 Token 用量统计
    )

    full_response = ""                                  # 收集完整响应
    usage_info = None                                   # Token 用量

    # 迭代处理每个数据块
    for chunk in stream:
        # 最后一个 chunk 可能只包含用量信息（没有 choices）
        if chunk.usage:
            usage_info = chunk.usage

        # 检查是否有内容块
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta                  # 增量内容

        # 提取文本内容（delta.content 可能为 None）
        if delta.content:
            print(delta.content, end="", flush=True)    # 实时打印，不换行
            full_response += delta.content              # 累积完整响应

    print()                                             # 输出结束后换行

    # 打印用量统计（如果有）
    if usage_info:
        print(f"[Token用量 - 输入:{usage_info.prompt_tokens} "
              f"输出:{usage_info.completion_tokens} "
              f"合计:{usage_info.total_tokens}]")

    return full_response


def streaming_with_timing(client: OpenAI, user_message: str):
    """
    带时间统计的流式对话，展示首 Token 延迟和总耗时。
    """
    print(f"\n问：{user_message}")
    print("答：", end="", flush=True)

    start_time = time.time()                            # 记录开始时间
    first_token_time = None                             # 首个 Token 的时间
    full_response = ""

    stream = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": user_message}],
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        if delta.content:
            if first_token_time is None:
                first_token_time = time.time()          # 记录第一个 Token 的时间
            print(delta.content, end="", flush=True)
            full_response += delta.content

    end_time = time.time()                              # 记录结束时间

    print(f"\n\n[性能统计]")
    print(f"  首 Token 延迟：{(first_token_time - start_time):.2f} 秒")
    print(f"  总耗时：{(end_time - start_time):.2f} 秒")
    print(f"  输出字符数：{len(full_response)}")


def streaming_long_content(client: OpenAI):
    """
    流式输出的优势：长内容生成时，用户不需要等待全部完成。
    """
    print("\n=== 长内容流式输出示例 ===")
    print("（流式输出让用户在生成过程中就能看到内容）\n")

    streaming_chat(
        client,
        "请写一篇300字左右的关于人工智能发展历史的简短文章。"
    )


def compare_stream_vs_non_stream(client: OpenAI):
    """
    对比流式和非流式调用的体验差异。
    """
    question = "列出5种常见的编程语言及其主要用途。"

    print("\n=== 非流式调用（等待完整响应）===")
    start = time.time()
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": question}],
        stream=False,                                   # 非流式
    )
    elapsed = time.time() - start
    print(completion.choices[0].message.content)
    print(f"[非流式总耗时：{elapsed:.2f} 秒（全部等待后才显示）]")

    print("\n=== 流式调用（实时输出）===")
    streaming_with_timing(client, question)


def main():
    client = create_client()

    print("=== 示例1：基础流式对话 ===")
    streaming_chat(client, "用简单的语言解释什么是深度学习？")

    print("\n=== 示例2：长内容流式输出 ===")
    streaming_long_content(client)

    print("\n=== 示例3：流式 vs 非流式对比 ===")
    compare_stream_vs_non_stream(client)


if __name__ == "__main__":
    main()
