"""
第04章 - Tools工具调用：基础工具
==================================
本示例演示如何使用 @tool 装饰器创建 LangChain 工具，
以及如何将工具绑定到 LLM 并手动执行工具调用。

学习要点：
1. @tool 装饰器的基本用法
2. 使用 Pydantic 定义工具参数
3. llm.bind_tools() 绑定工具到模型
4. 解析 tool_calls 并执行工具
5. 完整的工具调用循环
"""

import json
import os
import sys
from typing import Optional

from langchain_core.messages import HumanMessage, ToolMessage
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
# 定义工具
# ============================================================

class CalculatorInput(BaseModel):
    """计算器工具的参数模型"""
    expression: str = Field(description="要计算的数学表达式，如 '2 + 3 * 4' 或 '(10 + 5) / 3'")


@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """
    计算数学表达式的值。
    支持基本运算：加(+)、减(-)、乘(*)、除(/)、幂(**）、括号()。
    不支持三角函数等高级函数。
    """
    try:
        # 安全地评估数学表达式（只允许数字和运算符）
        # 注意：生产环境中应使用更安全的表达式解析库
        allowed_chars = set("0123456789+-*/().**% \t\n")
        if not all(c in allowed_chars for c in expression):
            return f"错误：表达式包含不允许的字符"

        result = eval(expression, {"__builtins__": {}})        # 限制内置函数
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "错误：除数不能为零"
    except Exception as e:
        return f"计算错误：{str(e)}"


class WeatherInput(BaseModel):
    """天气查询工具的参数模型"""
    city: str = Field(description="城市名称，如'北京'、'上海'、'广州'")
    unit: str = Field(default="celsius", description="温度单位：celsius（摄氏度）或 fahrenheit（华氏度）")


@tool(args_schema=WeatherInput)
def get_weather(city: str, unit: str = "celsius") -> str:
    """
    获取指定城市的当前天气信息。
    返回天气状况、温度和湿度。

    注意：这是一个模拟工具，实际项目中应调用真实天气API。
    """
    # 模拟天气数据
    weather_data = {
        "北京": {"condition": "晴天", "temp_celsius": 22, "humidity": 45},
        "上海": {"condition": "多云", "temp_celsius": 26, "humidity": 70},
        "广州": {"condition": "小雨", "temp_celsius": 28, "humidity": 85},
        "成都": {"condition": "阴天", "temp_celsius": 20, "humidity": 75},
        "深圳": {"condition": "晴天", "temp_celsius": 29, "humidity": 65},
    }

    # 查找城市天气（模糊匹配）
    data = None
    for key, val in weather_data.items():
        if key in city or city in key:
            data = val
            city = key
            break

    if not data:
        return f"抱歉，暂无{city}的天气数据"

    temp = data["temp_celsius"]
    if unit == "fahrenheit":
        temp = temp * 9 / 5 + 32                              # 转换为华氏度
        unit_str = "°F"
    else:
        unit_str = "°C"

    return (
        f"{city}当前天气：{data['condition']}，"
        f"温度 {temp}{unit_str}，"
        f"湿度 {data['humidity']}%"
    )


class UnitConvertInput(BaseModel):
    """单位转换工具的参数模型"""
    value: float = Field(description="要转换的数值")
    from_unit: str = Field(description="原始单位（如：km, m, kg, lb, c, f）")
    to_unit: str = Field(description="目标单位（如：km, m, kg, lb, c, f）")


@tool(args_schema=UnitConvertInput)
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    进行单位换算，支持长度、重量、温度的转换。
    支持的单位：km(千米), m(米), cm(厘米), kg(千克), g(克), lb(磅), c(摄氏), f(华氏)
    """
    conversions = {
        # 长度（以米为基准）
        ("km", "m"): lambda x: x * 1000,
        ("m", "km"): lambda x: x / 1000,
        ("m", "cm"): lambda x: x * 100,
        ("cm", "m"): lambda x: x / 100,
        ("km", "cm"): lambda x: x * 100000,
        # 重量（以千克为基准）
        ("kg", "g"): lambda x: x * 1000,
        ("g", "kg"): lambda x: x / 1000,
        ("kg", "lb"): lambda x: x * 2.20462,
        ("lb", "kg"): lambda x: x / 2.20462,
        # 温度
        ("c", "f"): lambda x: x * 9 / 5 + 32,
        ("f", "c"): lambda x: (x - 32) * 5 / 9,
    }

    key = (from_unit.lower(), to_unit.lower())
    if key not in conversions:
        return f"不支持从 {from_unit} 到 {to_unit} 的转换"

    result = conversions[key](value)
    return f"{value} {from_unit} = {result:.4f} {to_unit}"


# ============================================================
# 工具调用流程演示
# ============================================================

def demo_tool_inspection():
    """
    演示如何查看工具的属性（名称、描述、参数 Schema）。
    LLM 正是通过这些信息来决定调用哪个工具。
    """
    print("=== 1. 工具属性检查 ===\n")

    tools = [calculator, get_weather, unit_converter]
    for t in tools:
        print(f"工具名称：{t.name}")
        print(f"工具描述：{t.description}")
        print(f"参数Schema：{json.dumps(t.args_schema.model_json_schema(), ensure_ascii=False, indent=2)}")
        print()


def demo_manual_tool_call():
    """
    演示直接调用工具（不经过 LLM）。
    """
    print("=== 2. 直接调用工具 ===\n")

    # 直接调用工具函数
    result1 = calculator.invoke({"expression": "123 * 456 + 789"})
    print(f"计算结果：{result1}")

    result2 = get_weather.invoke({"city": "北京", "unit": "celsius"})
    print(f"天气信息：{result2}")

    result3 = unit_converter.invoke({"value": 5.5, "from_unit": "km", "to_unit": "m"})
    print(f"单位转换：{result3}")
    print()


def demo_llm_with_tools(llm: ChatOpenAI):
    """
    演示将工具绑定到 LLM，让 LLM 自动决定调用哪个工具。
    """
    print("=== 3. 绑定工具到 LLM ===\n")

    tools = [calculator, get_weather, unit_converter]

    # 将工具绑定到 LLM
    llm_with_tools = llm.bind_tools(tools)

    # 构建工具映射（用于后续执行）
    tool_map = {t.name: t for t in tools}

    questions = [
        "北京今天天气怎么样？",
        "计算 (123 + 456) * 2 等于多少？",
        "5千克等于多少磅？",
    ]

    for question in questions:
        print(f"用户问题：{question}")

        # 调用 LLM（LLM 会返回工具调用指令，而不是直接回答）
        messages = [HumanMessage(content=question)]
        response = llm_with_tools.invoke(messages)

        if response.tool_calls:
            # LLM 决定调用工具
            print(f"LLM 决定调用工具：{[tc['name'] for tc in response.tool_calls]}")

            # 执行工具调用
            tool_messages = []
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_func = tool_map[tool_name]

                # 执行工具
                tool_result = tool_func.invoke(tool_args)
                print(f"  工具 [{tool_name}] 执行结果：{tool_result}")

                # 创建 ToolMessage（将结果返回给 LLM）
                tool_messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],    # 关联到对应的工具调用
                    )
                )

            # 将工具结果返回给 LLM，获取最终回答
            all_messages = messages + [response] + tool_messages
            final_response = llm_with_tools.invoke(all_messages)
            print(f"最终回答：{final_response.content}")
        else:
            # LLM 直接回答（不需要工具）
            print(f"LLM 直接回答：{response.content}")

        print()


def main():
    llm = create_llm()

    demo_tool_inspection()
    demo_manual_tool_call()
    demo_llm_with_tools(llm)

    print("所有工具调用示例运行完成！")


if __name__ == "__main__":
    main()
