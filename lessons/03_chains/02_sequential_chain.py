"""
第03章 - Chain链式调用：顺序链
================================
本示例演示如何将多个链顺序连接，
实现"上一个链的输出作为下一个链的输入"的管道处理。

学习要点：
1. 多链顺序连接
2. 在链之间传递中间结果
3. 使用字典组合多个链的输出
4. 实际用例：翻译 → 摘要 → 评分
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


def demo_two_step_chain(llm: ChatOpenAI):
    """
    演示两步链：先生成文章，再对文章进行摘要。

    核心演示：
        combined_chain = write_chain | RunnableLambda(wrap_as_article_dict) | summary_chain
    通过 LCEL 管道操作符将两条独立的链串联成一条完整链，
    上一步的字符串输出经 RunnableLambda 包装成字典后流入下一步。
    """
    print("=== 1. 两步链：生成文章 → 生成摘要 ===\n")

    # 链1：根据主题生成文章（输出：字符串）
    write_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位专业作家，写文章时注重逻辑和细节。"),
        ("human", "写一篇关于{topic}的200字短文。"),
    ])
    write_chain = write_prompt | llm | StrOutputParser()

    # 链2：对文章进行摘要（输入：{"article": "..."} 字典，输出：字符串）
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个擅长总结的助手，用50字以内概括文章核心观点。"),
        ("human", "请摘要以下文章：\n\n{article}"),
    ])
    summary_chain = summary_prompt | llm | StrOutputParser()

    def wrap_as_article_dict(article: str) -> dict:
        """将 write_chain 输出的字符串包装为 summary_chain 所需的字典格式。"""
        return {"article": article}

    # 将两个链串联成完整的两步链
    # 关键：write_chain 输出字符串 → RunnableLambda 转为字典 → summary_chain 接收字典
    combined_chain = (
        write_chain                                     # 步骤1：生成文章（输出：字符串）
        | RunnableLambda(wrap_as_article_dict)          # 中间转换：str → {"article": str}
        | summary_chain                                 # 步骤2：生成摘要（输出：字符串）
    )

    topic = "量子计算的未来"
    print(f"主题：{topic}\n")

    # 【方式一】使用组合链一次调用（LCEL 推荐用法，内部自动完成两步）
    print("--- 方式一：组合链一次调用 ---")
    combined_summary = combined_chain.invoke({"topic": topic})
    print(f"最终摘要（通过 combined_chain）：\n{combined_summary}")

    # 【方式二】分步调用，便于调试查看中间结果
    print("\n--- 方式二：分步调用（便于调试） ---")
    article = write_chain.invoke({"topic": topic})
    print(f"生成的文章：\n{article}")
    summary = summary_chain.invoke({"article": article})
    print(f"\n文章摘要：\n{summary}")


def demo_pipeline_chain(llm: ChatOpenAI):
    """
    演示多步管道链（顺序链）：
    英文文本 → 翻译为中文 → 并行生成标题 & 情感分析

    核心演示：
        pipeline_chain = translate_chain | RunnableLambda(split_to_branches) | RunnableParallel(title=title_chain, sentiment=sentiment_chain)
    三条链通过 LCEL 连接成单一管道，第一步的输出自动流入后续步骤。
    """
    print("\n=== 2. 多步管道链：翻译 → 生成标题 & 情感分析 ===\n")

    english_text = """
    Artificial intelligence is transforming industries across the globe.
    From healthcare to finance, AI systems are making complex decisions
    that were once reserved for human experts. However, this rapid advancement
    also raises important questions about privacy, bias, and the future of work.
    """

    # 步骤1：翻译（输入：{"english": "..."}, 输出：中文字符串）
    translate_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业翻译，将英文翻译成自然流畅的中文。"),
        ("human", "翻译以下英文文本：\n{english}"),
    ])

    # 步骤2：生成标题（输入：{"chinese_text": "..."}, 输出：标题字符串）
    title_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位编辑，根据文章内容生成吸引人的标题（不超过20字）。"),
        ("human", "为以下文章生成一个标题：\n{chinese_text}"),
    ])

    # 步骤3：情感分析（输入：{"text": "..."}, 输出：情感分析字符串）
    sentiment_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是情感分析专家，判断文本的情感倾向（正面/负面/中性）并说明原因。"),
        ("human", "分析以下文本的情感：\n{text}"),
    ])

    # 构建各步骤链
    translate_chain = translate_prompt | llm | StrOutputParser()
    title_chain = title_prompt | llm | StrOutputParser()
    sentiment_chain = sentiment_prompt | llm | StrOutputParser()

    def split_to_branches(chinese: str) -> dict:
        """将翻译结果分发给后续两个分支所需的字典。"""
        return {"chinese_text": chinese, "text": chinese}

    # 构建完整管道链（LCEL 推荐写法）：
    # translate_chain  →  RunnableLambda(split)  →  RunnableParallel(title, sentiment)
    pipeline_chain = (
        translate_chain                          # 步骤1：翻译（输出：中文字符串）
        | RunnableLambda(split_to_branches)      # 中间转换：str → {"chinese_text":..., "text":...}
        | RunnableParallel(                      # 步骤2&3 并行执行（共享同一输入字典）
            title=title_chain,                   #   使用 {chinese_text} → 生成标题
            sentiment=sentiment_chain,           #   使用 {text}         → 情感分析
        )
    )

    print("原始英文：")
    print(english_text.strip())
    print()

    # 【方式一】使用完整管道链一次调用（LCEL 推荐用法）
    print("--- 方式一：管道链一次调用 ---")
    result = pipeline_chain.invoke({"english": english_text})
    print(f"生成标题：{result['title']}\n")
    print(f"情感分析：\n{result['sentiment']}")

    # 【方式二】分步调用，便于调试查看中间结果
    print("\n--- 方式二：分步调用（便于调试） ---")
    chinese = translate_chain.invoke({"english": english_text})
    print(f"步骤1 - 翻译结果：\n{chinese}\n")

    title = title_chain.invoke({"chinese_text": chinese})
    print(f"步骤2 - 生成标题：{title}\n")

    sentiment = sentiment_chain.invoke({"text": chinese})
    print(f"步骤3 - 情感分析：\n{sentiment}")


def demo_parallel_chains(llm: ChatOpenAI):
    """
    演示并行链：同时执行多个处理任务，合并结果。
    """
    print("\n=== 3. 并行链：同时生成多种内容 ===\n")

    topic = "人工智能"

    # 定义三个不同的处理链（针对同一输入并行执行）
    pros_prompt = ChatPromptTemplate.from_messages([
        ("human", "列出{topic}的3个主要优势（简洁，每点一行）"),
    ])

    cons_prompt = ChatPromptTemplate.from_messages([
        ("human", "列出{topic}的3个主要挑战（简洁，每点一行）"),
    ])

    summary_prompt = ChatPromptTemplate.from_messages([
        ("human", "用两句话总结{topic}"),
    ])

    # 使用 RunnableParallel 并行执行多个链
    # RunnableParallel 是 LCEL 的正确并行写法，dict 字面量没有 .invoke()
    parallel_chain = RunnableParallel(
        topic=RunnablePassthrough() | RunnableLambda(lambda x: x["topic"]),
        advantages=pros_prompt | llm | StrOutputParser(),
        challenges=cons_prompt | llm | StrOutputParser(),
        summary=summary_prompt | llm | StrOutputParser(),
    )

    result = parallel_chain.invoke({"topic": topic})

    print(f"主题：{result['topic']}\n")
    print(f"✅ 主要优势：\n{result['advantages']}\n")
    print(f"⚠️  主要挑战：\n{result['challenges']}\n")
    print(f"📝 总结：\n{result['summary']}")


def main():
    llm = create_llm()

    demo_two_step_chain(llm)
    demo_pipeline_chain(llm)
    demo_parallel_chains(llm)

    print("\n\n所有顺序链示例运行完成！")


if __name__ == "__main__":
    main()
