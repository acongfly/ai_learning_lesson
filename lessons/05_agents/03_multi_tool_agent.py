"""
第05章 - Agent智能体：多工具 Agent
=====================================
本示例演示一个拥有多种不同类型工具的 Agent，
能够处理数学、字符串、时间、单位转换等多种任务。

Agent 的强大之处在于它能自主决定使用哪些工具，
以及按什么顺序使用，无需人工预先规划。

学习要点：
1. 创建多个不同类型的工具
2. Agent 自主选择工具解决复杂问题
3. 观察 Agent 的工具选择策略
4. 处理需要多工具协作的复杂查询
"""

import math
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

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
# 数学工具组
# ============================================================

@tool
def basic_calculator(expression: str) -> str:
    """
    基础计算器：支持 +、-、*、/、** 运算和括号。
    示例：'2 + 3 * 4'、'(10 + 5) / 3'、'2 ** 10'
    """
    try:
        safe_globals = {"__builtins__": {}, "abs": abs, "round": round, "pow": pow}
        result = eval(expression, safe_globals)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


@tool
def statistics_calculator(numbers: str) -> str:
    """
    统计计算器：计算一组数字的平均值、最大值、最小值、总和。
    数字用逗号分隔，例如：'1, 2, 3, 4, 5' 或 '10, 20, 30'
    """
    try:
        nums = [float(n.strip()) for n in numbers.split(",")]
        if not nums:
            return "错误：数字列表为空"
        return (
            f"数据：{nums}\n"
            f"总和：{sum(nums)}\n"
            f"平均值：{sum(nums)/len(nums):.4f}\n"
            f"最大值：{max(nums)}\n"
            f"最小值：{min(nums)}\n"
            f"数量：{len(nums)}"
        )
    except ValueError as e:
        return f"数字解析错误：{str(e)}"


@tool
def geometry_calculator(shape: str, dimensions: str) -> str:
    """
    几何计算器：计算常见几何图形的面积和周长。
    shape: 图形类型（circle/square/rectangle/triangle）
    dimensions: 尺寸参数（用逗号分隔）
      - circle: 半径
      - square: 边长
      - rectangle: 长,宽
      - triangle: 底,高
    """
    try:
        dims = [float(d.strip()) for d in dimensions.split(",")]
        shape = shape.lower()

        if shape == "circle":
            r = dims[0]
            return f"圆形 半径={r}：\n  面积 = π×r² = {math.pi * r**2:.4f}\n  周长 = 2πr = {2*math.pi*r:.4f}"
        elif shape == "square":
            s = dims[0]
            return f"正方形 边长={s}：\n  面积 = {s**2:.4f}\n  周长 = {4*s:.4f}"
        elif shape == "rectangle":
            l, w = dims[0], dims[1]
            return f"长方形 长={l} 宽={w}：\n  面积 = {l*w:.4f}\n  周长 = {2*(l+w):.4f}"
        elif shape == "triangle":
            base, height = dims[0], dims[1]
            return f"三角形 底={base} 高={height}：\n  面积 = {0.5*base*height:.4f}"
        else:
            return f"不支持的图形：{shape}（支持：circle, square, rectangle, triangle）"
    except Exception as e:
        return f"几何计算错误：{str(e)}"


# ============================================================
# 字符串工具组
# ============================================================

@tool
def text_analyzer(text: str) -> str:
    """
    文本分析器：统计文本的字符数、词数、句子数等信息。
    """
    char_count = len(text)
    word_count = len(text.split())
    sentence_count = text.count("。") + text.count("！") + text.count("？") + \
                     text.count(".") + text.count("!") + text.count("?")

    # 中文字符数
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")

    return (
        f"文本分析结果：\n"
        f"  总字符数：{char_count}\n"
        f"  中文字符数：{chinese_chars}\n"
        f"  词数（空格分隔）：{word_count}\n"
        f"  句子数（估计）：{max(1, sentence_count)}\n"
        f"  平均词长：{char_count/max(1,word_count):.1f}字符/词"
    )


@tool
def text_transformer(text: str, operation: str) -> str:
    """
    文本转换工具。
    operation 支持：
    - 'upper': 转大写
    - 'lower': 转小写
    - 'reverse': 反转字符串
    - 'title': 标题格式（每词首字母大写）
    - 'remove_spaces': 删除所有空格
    """
    ops = {
        "upper": str.upper,
        "lower": str.lower,
        "reverse": lambda t: t[::-1],
        "title": str.title,
        "remove_spaces": lambda t: t.replace(" ", ""),
    }

    if operation not in ops:
        return f"不支持的操作：{operation}"

    return ops[operation](text)


# ============================================================
# 时间工具组
# ============================================================

@tool
def get_current_time() -> str:
    """获取当前日期和时间，包括星期几。"""
    now = datetime.now()
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return (
        f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"星期：{weekdays[now.weekday()]}\n"
        f"当前年份：{now.year}\n"
        f"今天是第 {now.timetuple().tm_yday} 天"
    )


class DateCalcInput(BaseModel):
    start_date: str = Field(description="起始日期（格式：YYYY-MM-DD，如 2024-01-01）")
    end_date: str = Field(description="结束日期（格式：YYYY-MM-DD，如 2024-12-31）")


@tool(args_schema=DateCalcInput)
def date_calculator(start_date: str, end_date: str) -> str:
    """
    计算两个日期之间的天数差。
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        delta = end - start
        days = delta.days

        return (
            f"从 {start_date} 到 {end_date}：\n"
            f"  相差 {abs(days)} 天\n"
            f"  约 {abs(days)//7} 周 {abs(days)%7} 天\n"
            f"  约 {abs(days)/30:.1f} 个月"
        )
    except ValueError as e:
        return f"日期格式错误：{str(e)}"


# ============================================================
# 演示
# ============================================================

def run_agent_task(agent, task: str):
    """运行 Agent 任务并展示结果。"""
    print(f"\n{'='*60}")
    print(f"任务：{task}")
    print("=" * 60)

    result = agent.invoke({
        "messages": [HumanMessage(content=task)]
    })

    # 展示推理过程
    messages = result["messages"]
    for msg in messages:
        msg_type = msg.__class__.__name__
        if msg_type == "AIMessage":
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  🔧 调用工具：{tc['name']}({tc['args']})")
            else:
                print(f"\n✅ 最终答案：{msg.content}")
        elif msg_type == "ToolMessage":
            result_preview = msg.content[:100]
            print(f"     结果：{result_preview}{'...' if len(msg.content) > 100 else ''}")


def main():
    llm = create_llm()

    # 所有工具
    tools = [
        # 数学工具
        basic_calculator,
        statistics_calculator,
        geometry_calculator,
        # 字符串工具
        text_analyzer,
        text_transformer,
        # 时间工具
        get_current_time,
        date_calculator,
    ]

    print(f"多工具 Agent 初始化完成")
    print(f"可用工具（{len(tools)}个）：{[t.name for t in tools]}\n")

    # 创建 Agent
    agent = create_agent(llm, tools=tools)

    # 测试任务1：纯数学任务
    run_agent_task(agent, "计算一个半径为5的圆的面积，结果保留两位小数。")

    # 测试任务2：字符串处理
    run_agent_task(agent, "分析文本 'Hello World from LangGraph Multi-Tool Agent' 的统计信息，然后把它转换成大写。")

    # 测试任务3：复合任务（需要多个工具）
    run_agent_task(
        agent,
        "今天是几号？请计算从2024年1月1日到今天共经过了多少天？"
        "然后用这个天数除以7，告诉我大约经过了几周。"
    )

    # 测试任务4：统计计算
    run_agent_task(
        agent,
        "计算以下数字的平均值和总和：85, 92, 78, 96, 88, 75, 91"
    )

    print("\n\n所有多工具 Agent 示例运行完成！")


if __name__ == "__main__":
    main()
