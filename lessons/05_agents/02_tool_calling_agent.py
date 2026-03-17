"""
第05章 - Agent智能体：手动工具调用 Agent
==========================================
本示例演示如何手动实现 Agent 的工具调用循环，
深入理解 Agent 的内部工作机制。

Agent 循环：
1. 将用户消息发送给 LLM
2. LLM 分析后决定调用哪些工具（返回 tool_calls）
3. 执行工具，获取结果
4. 将工具结果作为 ToolMessage 加入消息历史
5. 再次调用 LLM，让它基于工具结果继续推理
6. 重复 2-5，直到 LLM 不再调用工具（给出最终答案）

学习要点：
1. LLM 的 tool_calls 属性
2. ToolMessage 的构造和使用
3. Agent 循环的实现
4. 最大迭代次数限制（防止无限循环）
5. 错误处理
"""

import os
import sys
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
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


# ============================================================
# 工具定义
# ============================================================

@tool
def search_knowledge_base(query: str) -> str:
    """
    在知识库中搜索相关信息。
    模拟知识库查询，实际项目中会连接真实的向量数据库或搜索引擎。
    """
    # 模拟知识库内容
    knowledge_base = {
        "python": "Python是一种高级编程语言，以简洁易读著称。1991年由Guido van Rossum创建。",
        "机器学习": "机器学习是AI的子领域，通过数据训练模型，让计算机能从经验中学习。",
        "langchain": "LangChain是一个AI应用开发框架，提供链、工具、Agent等组件，简化LLM应用开发。",
        "langgraph": "LangGraph是基于图状态机的AI工作流框架，支持循环、条件分支和多Agent协作。",
        "向量数据库": "向量数据库专门存储和检索高维向量，是RAG系统的核心组件，代表产品有Pinecone、FAISS、Chroma。",
        "transformer": "Transformer是一种基于注意力机制的神经网络架构，是现代LLM的基础，由Google在2017年提出。",
    }

    # 简单关键词匹配
    results = []
    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key.lower() in query_lower or query_lower in key.lower():
            results.append(f"【{key}】{value}")

    if results:
        return "\n".join(results)
    return f"知识库中未找到关于'{query}'的信息"


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式，支持基本四则运算和幂运算。
    示例：'10 * 5 + 3'、'2 ** 8'、'100 / 4'
    """
    try:
        allowed = set("0123456789+-*/.() \t")
        if not all(c in allowed for c in expression):
            return f"表达式包含不支持的字符"
        result = eval(expression, {"__builtins__": {}})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


@tool
def get_recommendation(topic: str, count: int = 3) -> str:
    """
    获取指定主题的学习资源推荐。
    参数 count 指定推荐数量（1-5之间）。
    """
    recommendations = {
        "python": [
            "《Python编程：从入门到实践》- 适合初学者",
            "Python官方文档：https://docs.python.org",
            "Codecademy Python课程：在线互动学习",
            "《流畅的Python》- 适合中级开发者",
            "Python Cookbook - 实用技巧集",
        ],
        "机器学习": [
            "《机器学习》周志华 - 西瓜书，经典教材",
            "Coursera机器学习课程 - Andrew Ng",
            "《Python机器学习》- 实践导向",
            "Kaggle竞赛平台 - 实战练习",
            "《深度学习》花书 - 理论深度",
        ],
        "langchain": [
            "LangChain官方文档：https://python.langchain.com",
            "LangChain GitHub：代码示例丰富",
            "YouTube: LangChain教程系列",
            "《构建LLM应用》- 实战指南",
        ],
    }

    # 查找最匹配的主题
    for key, recs in recommendations.items():
        if key in topic.lower() or topic.lower() in key:
            count = min(count, len(recs))
            return "\n".join(f"{i+1}. {r}" for i, r in enumerate(recs[:count]))

    return f"暂无关于'{topic}'的推荐资源"


# ============================================================
# 手动 Agent 循环实现
# ============================================================

class ManualAgent:
    """
    手动实现的工具调用 Agent。
    演示 Agent 循环的底层工作机制。
    """

    def __init__(self, llm: ChatOpenAI, tools: list, max_iterations: int = 10):
        """
        初始化 Agent。

        参数:
            llm: 语言模型
            tools: 工具列表
            max_iterations: 最大迭代次数（防止无限循环）
        """
        self.llm_with_tools = llm.bind_tools(tools)        # 绑定工具到 LLM
        self.tool_map = {t.name: t for t in tools}          # 工具名称 → 工具函数
        self.max_iterations = max_iterations

    def run(self, user_message: str, system_prompt: str = None, verbose: bool = True) -> str:
        """
        运行 Agent，执行完整的工具调用循环。

        参数:
            user_message: 用户输入
            system_prompt: 系统提示（可选）
            verbose: 是否打印详细过程

        返回:
            最终答案字符串
        """
        # 初始化消息历史
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=user_message))

        if verbose:
            print(f"🎯 任务：{user_message}\n")

        iteration = 0

        # ============================================================
        # Agent 核心循环
        # ============================================================
        while iteration < self.max_iterations:
            iteration += 1

            if verbose:
                print(f"--- 第 {iteration} 轮推理 ---")

            # 步骤1：调用 LLM
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)                      # 将 AI 响应加入历史

            # 步骤2：检查是否有工具调用
            if not response.tool_calls:
                # 没有工具调用 → LLM 给出了最终答案
                if verbose:
                    print(f"✅ AI给出最终答案（无需更多工具调用）")
                return response.content

            # 步骤3：执行所有工具调用
            if verbose:
                print(f"🔧 LLM 决定调用 {len(response.tool_calls)} 个工具：")

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                if verbose:
                    print(f"  → 工具：{tool_name}")
                    print(f"    参数：{tool_args}")

                # 执行工具
                if tool_name not in self.tool_map:
                    tool_result = f"错误：未找到工具 '{tool_name}'"
                else:
                    try:
                        tool_result = self.tool_map[tool_name].invoke(tool_args)
                    except Exception as e:
                        tool_result = f"工具执行错误：{str(e)}"

                if verbose:
                    result_preview = str(tool_result)[:150]
                    print(f"    结果：{result_preview}{'...' if len(str(tool_result)) > 150 else ''}")

                # 步骤4：将工具结果作为 ToolMessage 加入历史
                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_id,               # 关联到对应的工具调用
                    )
                )

            if verbose:
                print()

        # 超过最大迭代次数
        if verbose:
            print(f"⚠️  达到最大迭代次数（{self.max_iterations}），强制结束")

        return "已达到最大迭代次数，请尝试更简单的问题。"


def demo_agent_loop(llm: ChatOpenAI):
    """演示手动 Agent 循环。"""
    tools = [search_knowledge_base, calculate, get_recommendation]

    agent = ManualAgent(
        llm=llm,
        tools=tools,
        max_iterations=10,
    )

    system_prompt = """你是一个知识渊博的AI助手，可以搜索知识库、进行计算和提供学习建议。
回答时要全面、准确，充分利用可用的工具。"""

    print("=" * 60)
    print("测试1：查询知识并推荐学习资源")
    print("=" * 60)
    answer = agent.run(
        "什么是LangChain？有哪些学习资源推荐？",
        system_prompt=system_prompt,
    )
    print(f"\n最终答案：\n{answer}")

    print("\n\n" + "=" * 60)
    print("测试2：综合任务（查询 + 计算）")
    print("=" * 60)
    answer = agent.run(
        "Python有多少年历史了（2024年算，Python 1991年创建）？另外，LangGraph是什么？",
        system_prompt=system_prompt,
    )
    print(f"\n最终答案：\n{answer}")


def main():
    llm = create_llm()
    demo_agent_loop(llm)
    print("\n\n手动工具调用 Agent 示例运行完成！")


if __name__ == "__main__":
    main()
