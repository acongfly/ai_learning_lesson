"""
第05章 - Agent智能体：ReAct Agent
====================================
本示例演示使用 create_agent（LangGraph V1.0 起从 langchain.agents 导入，
原名 create_react_agent）快速创建 ReAct Agent，
Agent 能够自主使用工具，循环推理直到完成任务。

ReAct = Reasoning（推理）+ Acting（行动）
Agent 流程：思考 → 选择工具 → 执行 → 观察结果 → 继续思考...

学习要点：
1. create_agent（原 create_react_agent）的使用方式
2. 自定义工具的创建
3. 运行 Agent 并查看推理过程
4. 解析 Agent 的消息历史
"""

import math
import os
import sys
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
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
# 自定义工具
# ============================================================

class CalculatorInput(BaseModel):
    expression: str = Field(description="数学表达式，如 '2 + 3'、'sqrt(16)'、'2 ** 10'")


@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """
    计算数学表达式。
    支持基本运算（+、-、*、/）、幂运算（**）、
    以及常见数学函数：sqrt（平方根）、abs（绝对值）、round（四舍五入）。
    """
    try:
        # 允许使用 math 模块中的安全函数
        safe_globals = {
            "__builtins__": {},
            "sqrt": math.sqrt,
            "abs": abs,
            "round": round,
            "pow": pow,
            "pi": math.pi,
            "e": math.e,
            "log": math.log,
            "log10": math.log10,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
        }
        result = eval(expression, safe_globals)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


class StringOpsInput(BaseModel):
    text: str = Field(description="要操作的文本")
    operation: str = Field(
        description="操作类型：'upper'（大写）、'lower'（小写）、'reverse'（反转）、"
                    "'length'（长度）、'count_words'（统计词数）、'title'（标题格式）"
    )


@tool(args_schema=StringOpsInput)
def string_operations(text: str, operation: str) -> str:
    """
    对字符串执行各种操作：转换大小写、反转、统计长度等。
    """
    ops = {
        "upper": lambda t: t.upper(),
        "lower": lambda t: t.lower(),
        "reverse": lambda t: t[::-1],
        "length": lambda t: str(len(t)),
        "count_words": lambda t: str(len(t.split())),
        "title": lambda t: t.title(),
        "strip": lambda t: t.strip(),
    }

    if operation not in ops:
        return f"不支持的操作：{operation}。支持的操作：{', '.join(ops.keys())}"

    result = ops[operation](text)
    return f"操作 '{operation}' 的结果：{result}"


@tool
def get_current_datetime() -> str:
    """
    获取当前的日期和时间。
    返回格式：年-月-日 时:分:秒，以及星期几。
    """
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekdays[now.weekday()]
    return f"当前时间：{now.strftime('%Y年%m月%d日 %H:%M:%S')}，{weekday}"


class UnitConverterInput(BaseModel):
    value: float = Field(description="要转换的数值")
    from_unit: str = Field(description="原始单位（km/m/cm/kg/g/lb/c/f）")
    to_unit: str = Field(description="目标单位（km/m/cm/kg/g/lb/c/f）")


@tool(args_schema=UnitConverterInput)
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    单位换算工具，支持长度（km/m/cm）、重量（kg/g/lb）、温度（c/f）的转换。
    """
    conversions = {
        ("km", "m"): lambda x: x * 1000,
        ("m", "km"): lambda x: x / 1000,
        ("m", "cm"): lambda x: x * 100,
        ("cm", "m"): lambda x: x / 100,
        ("kg", "g"): lambda x: x * 1000,
        ("g", "kg"): lambda x: x / 1000,
        ("kg", "lb"): lambda x: x * 2.20462,
        ("lb", "kg"): lambda x: x / 2.20462,
        ("c", "f"): lambda x: x * 9 / 5 + 32,
        ("f", "c"): lambda x: (x - 32) * 5 / 9,
    }

    key = (from_unit.lower(), to_unit.lower())
    if key not in conversions:
        return f"不支持从 {from_unit} 到 {to_unit} 的转换"

    result = conversions[key](value)
    return f"{value} {from_unit} = {result:.4f} {to_unit}"


# ============================================================
# Agent 演示
# ============================================================

def show_agent_steps(result: dict):
    """
    展示 Agent 的推理步骤（消息历史）。
    """
    messages = result.get("messages", [])
    print(f"\n=== Agent 执行过程（共 {len(messages)} 条消息）===")

    for i, msg in enumerate(messages):
        msg_type = msg.__class__.__name__

        if msg_type == "HumanMessage":
            print(f"\n[{i+1}] 👤 用户：{msg.content}")

        elif msg_type == "AIMessage":
            if msg.tool_calls:
                # AI 决定调用工具
                for tc in msg.tool_calls:
                    print(f"\n[{i+1}] 🤖 AI（调用工具）：{tc['name']}({tc['args']})")
            else:
                # AI 给出最终答案
                print(f"\n[{i+1}] 🤖 AI（最终回答）：{msg.content}")

        elif msg_type == "ToolMessage":
            print(f"\n[{i+1}] 🔧 工具结果：{msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")


def demo_simple_agent(agent):
    """演示简单单工具问题。"""
    print("\n" + "=" * 60)
    print("示例1：简单数学计算")
    print("=" * 60)

    result = agent.invoke({
        "messages": [HumanMessage(content="计算 sqrt(144) + 2**8 的结果")]
    })

    show_agent_steps(result)
    print(f"\n✅ 最终答案：{result['messages'][-1].content}")


def demo_multi_step_agent(agent):
    """演示需要多步推理的问题。"""
    print("\n" + "=" * 60)
    print("示例2：多步骤推理")
    print("=" * 60)

    result = agent.invoke({
        "messages": [HumanMessage(
            content="现在几点了？另外帮我算一下：如果我从现在开始跑步2.5小时，"
                    "每小时跑8公里，我能跑多少公里？最后把结果转换成米。"
        )]
    })

    show_agent_steps(result)
    print(f"\n✅ 最终答案：{result['messages'][-1].content}")


def demo_string_agent(agent):
    """演示字符串操作。"""
    print("\n" + "=" * 60)
    print("示例3：字符串处理")
    print("=" * 60)

    result = agent.invoke({
        "messages": [HumanMessage(
            content="请把 'Hello World from LangGraph' 转换成大写，"
                    "然后统计它有多少个单词。"
        )]
    })

    show_agent_steps(result)
    print(f"\n✅ 最终答案：{result['messages'][-1].content}")


def main():
    llm = create_llm()

    # 工具列表
    tools = [calculator, string_operations, get_current_datetime, unit_converter]

    print("创建 ReAct Agent...")
    print(f"可用工具：{[t.name for t in tools]}")

    # 使用 create_agent 一行代码创建 Agent
    agent = create_agent(
        llm,
        tools=tools,
    )

    # 运行演示
    demo_simple_agent(agent)
    demo_multi_step_agent(agent)
    demo_string_agent(agent)

    print("\n\n所有 ReAct Agent 示例运行完成！")


if __name__ == "__main__":
    main()
