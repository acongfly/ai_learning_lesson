"""
第03章 - Chain链式调用：输出解析器
=====================================
本示例演示 LangChain 的各种输出解析器，
将 LLM 的非结构化文本输出转换为结构化数据。

学习要点：
1. StrOutputParser - 字符串解析
2. JsonOutputParser - JSON 解析（带 Pydantic 验证）
3. PydanticOutputParser - 严格 Pydantic 模型解析
4. CommaSeparatedListOutputParser - 列表解析
5. 解析失败的处理
"""

import json
import os
import sys
import warnings
from typing import List, Optional

from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
    JsonOutputParser,
    StrOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


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


def demo_str_parser(llm: ChatOpenAI):
    """
    演示 StrOutputParser：最简单的解析器，提取 AIMessage.content 字符串。
    """
    print("=== 1. StrOutputParser - 字符串解析 ===\n")

    prompt = ChatPromptTemplate.from_messages([
        ("human", "用一句话介绍{topic}。"),
    ])

    # StrOutputParser 将 AIMessage 转为纯字符串
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"topic": "Python"})
    print(f"类型：{type(result).__name__}")  # str
    print(f"内容：{result}\n")


def demo_json_parser(llm: ChatOpenAI):
    """
    演示 JsonOutputParser：解析 LLM 输出的 JSON 数据。
    可以结合 Pydantic 模型进行类型验证。
    """
    print("=== 2. JsonOutputParser - JSON 解析 ===\n")

    # 定义 Pydantic 数据模型
    class BookInfo(BaseModel):
        title: str = Field(description="书名")
        author: str = Field(description="作者")
        year: int = Field(description="出版年份")
        genre: str = Field(description="书籍类型（如：小说、技术书、传记）")
        description: str = Field(description="100字以内的简介")
        rating: float = Field(description="评分（1-10分）")

    # 创建 JSON 解析器（绑定 Pydantic 模型）
    parser = JsonOutputParser(pydantic_object=BookInfo)

    # 获取格式指令（告诉 LLM 应该输出什么格式）
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是一个书评专家。请按照指定格式输出书籍信息。\n\n{format_instructions}",
        ),
        ("human", "请提供《{book_name}》的信息。"),
    ])

    chain = prompt | llm | parser

    # 测试
    book_names = ["三体", "Python编程：从入门到实践"]
    for book in book_names:
        print(f"查询书籍：《{book}》")
        try:
            result = chain.invoke({
                "book_name": book,
                "format_instructions": format_instructions,
            })
            print(f"类型：{type(result).__name__}")
            print(f"书名：{result.get('title', 'N/A')}")
            print(f"作者：{result.get('author', 'N/A')}")
            print(f"年份：{result.get('year', 'N/A')}")
            print(f"书籍类型：{result.get('genre', 'N/A')}")
            print(f"简介：{result.get('description', 'N/A')}")
            print(f"评分：{result.get('rating', 'N/A')}/10")
        except Exception as e:
            print(f"解析失败：{e}")
        print()


def demo_list_parser(llm: ChatOpenAI):
    """
    演示 CommaSeparatedListOutputParser：将逗号分隔的文本解析为列表。
    """
    print("=== 3. CommaSeparatedListOutputParser - 列表解析 ===\n")

    parser = CommaSeparatedListOutputParser()
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "请按照以下格式输出：{format_instructions}",
        ),
        ("human", "列出{category}领域的5个重要概念。"),
    ])

    chain = prompt | llm | parser

    categories = ["机器学习", "网络安全"]
    for category in categories:
        print(f"领域：{category}")
        result = chain.invoke({
            "category": category,
            "format_instructions": format_instructions,
        })
        print(f"类型：{type(result).__name__}")  # list
        for i, item in enumerate(result, 1):
            print(f"  {i}. {item.strip()}")
        print()


def demo_structured_output(llm: ChatOpenAI):
    """
    演示使用 Pydantic 模型的结构化输出（更现代的方式）。
    使用 llm.with_structured_output() 方法。
    """
    print("=== 4. with_structured_output - 结构化输出（推荐方式）===\n")

    # 定义结构化输出的 Pydantic 模型
    class ProductReview(BaseModel):
        """产品评测结果"""
        product_name: str = Field(description="产品名称")
        pros: List[str] = Field(description="优点列表（3-5条）")
        cons: List[str] = Field(description="缺点列表（2-3条）")
        overall_score: float = Field(description="综合评分，1-10分")
        recommendation: str = Field(description="购买建议：推荐/不推荐/视情况而定")
        summary: str = Field(description="一句话总结")

    # 使用 with_structured_output 绑定 Pydantic 模型
    structured_llm = llm.with_structured_output(ProductReview)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位专业的产品评测师，请对产品进行客观评测。"),
        ("human", "请评测{product}这款产品。"),
    ])

    chain = prompt | structured_llm

    product = "MacBook Pro M3"
    print(f"评测产品：{product}")

    try:
        # with_structured_output 在 Pydantic v2 + langchain-core 0.3+ 下会触发：
        #   PydanticSerializationUnexpectedValue(Expected `none`, field_name='parsed')
        # 这是 LangChain 内部回调数据模型的已知类型标注问题（parsed 字段注解为 None，
        # 但实际接收到的是 Pydantic 模型实例），代码逻辑本身正确，此处局部屏蔽该噪声。
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=r".*Pydantic serializer warnings.*PydanticSerializationUnexpectedValue.*field_name='parsed'.*",
            )
            result = chain.invoke({"product": product})
        print(f"\n产品名称：{result.product_name}")
        print(f"综合评分：{result.overall_score}/10")
        print(f"购买建议：{result.recommendation}")
        print(f"\n优点：")
        for pro in result.pros:
            print(f"  ✅ {pro}")
        print(f"\n缺点：")
        for con in result.cons:
            print(f"  ⚠️  {con}")
        print(f"\n总结：{result.summary}")
    except Exception as e:
        print(f"结构化输出失败：{e}")


def main():
    llm = create_llm()

    demo_str_parser(llm)
    demo_json_parser(llm)
    demo_list_parser(llm)
    demo_structured_output(llm)

    print("\n所有输出解析器示例运行完成！")


if __name__ == "__main__":
    main()
