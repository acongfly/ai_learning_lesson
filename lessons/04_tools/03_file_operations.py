"""
第04章 - Tools工具调用：文件操作工具
======================================
本示例演示创建文件操作类工具：读取、写入、追加、列出文件等，
并让 LLM 通过工具调用来完成文件相关任务。

学习要点：
1. 文件读取工具（带安全检查）
2. 文件写入工具（带路径验证）
3. 文件列表工具
4. 文件信息工具
5. LLM 综合使用多个文件工具
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
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
# 安全工作目录（文件操作限制在此目录内）
# ============================================================

# 获取安全的工作目录（/tmp/ai_workspace 或系统临时目录）
# 使用 .resolve() 解析符号链接（修复 macOS 上 /tmp -> /private/tmp 导致的路径比较失败问题）
SAFE_WORKSPACE = Path(os.environ.get("AI_WORKSPACE", "/tmp/ai_file_workspace")).resolve()
SAFE_WORKSPACE.mkdir(parents=True, exist_ok=True)

# 允许的文件类型
ALLOWED_WRITE_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".log", ".yaml", ".yml"}


def resolve_safe_path(filename: str) -> tuple[Path, str]:
    """
    解析文件路径，确保它在安全工作目录内。
    返回 (绝对路径, 错误信息) 如果有错误则路径为None。
    """
    # 防止目录遍历攻击（如 ../../etc/passwd）
    safe_path = (SAFE_WORKSPACE / filename).resolve()

    # 确保路径在工作目录内
    try:
        safe_path.relative_to(SAFE_WORKSPACE)
    except ValueError:
        return None, f"⛔ 路径不在允许范围内：{filename}"

    return safe_path, ""


# ============================================================
# 工具定义
# ============================================================

class ReadFileInput(BaseModel):
    filename: str = Field(description=f"要读取的文件名（相对于工作目录）")


@tool(args_schema=ReadFileInput)
def read_file(filename: str) -> str:
    """
    读取工作目录中的文本文件内容。
    工作目录是一个安全的沙箱目录，不能访问系统文件。
    """
    file_path, error = resolve_safe_path(filename)
    if error:
        return error

    if not file_path.exists():
        return f"文件不存在：{filename}\n（工作目录：{SAFE_WORKSPACE}）"

    if not file_path.is_file():
        return f"路径不是文件：{filename}"

    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        return f"文件：{filename}（{len(lines)}行，{len(content)}字符）\n\n{content}"
    except Exception as e:
        return f"读取失败：{str(e)}"


class WriteFileInput(BaseModel):
    filename: str = Field(description="要写入的文件名（如 'output.txt'、'data.json'）")
    content: str = Field(description="要写入的文本内容")
    mode: str = Field(
        default="overwrite",
        description="写入模式：'overwrite'（覆盖，默认）或 'append'（追加）"
    )


@tool(args_schema=WriteFileInput)
def write_file(filename: str, content: str, mode: str = "overwrite") -> str:
    """
    将文本内容写入工作目录中的文件。
    支持覆盖（overwrite）和追加（append）两种模式。
    只能在安全工作目录内创建文件，不能写入系统文件。
    """
    file_path, error = resolve_safe_path(filename)
    if error:
        return error

    # 检查文件类型
    if file_path.suffix.lower() not in ALLOWED_WRITE_EXTENSIONS:
        return f"⛔ 不允许写入该类型的文件：{file_path.suffix}（允许：{', '.join(ALLOWED_WRITE_EXTENSIONS)}）"

    # 确保父目录存在
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        write_mode = "a" if mode == "append" else "w"
        with open(file_path, write_mode, encoding="utf-8") as f:
            f.write(content)

        file_size = file_path.stat().st_size
        action = "追加" if mode == "append" else "写入"
        return f"✅ 成功{action}文件：{filename}（文件大小：{file_size}字节）"
    except Exception as e:
        return f"写入失败：{str(e)}"


class ListFilesInput(BaseModel):
    pattern: str = Field(
        default="*",
        description="文件名匹配模式（如 '*.txt' 匹配所有txt文件，'*' 匹配所有文件）"
    )


@tool(args_schema=ListFilesInput)
def list_workspace_files(pattern: str = "*") -> str:
    """
    列出工作目录中的所有文件。
    可以使用通配符筛选文件（如 *.txt、*.json）。
    """
    try:
        files = list(SAFE_WORKSPACE.glob(pattern))

        if not files:
            return f"工作目录中没有匹配 '{pattern}' 的文件\n（工作目录：{SAFE_WORKSPACE}）"

        file_info = []
        for f in sorted(files):
            if f.is_file():
                size = f.stat().st_size
                modified = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                file_info.append(f"  📄 {f.name} ({size}字节, 修改于 {modified})")

        return (
            f"工作目录：{SAFE_WORKSPACE}\n"
            f"找到 {len(file_info)} 个文件：\n"
            + "\n".join(file_info)
        )
    except Exception as e:
        return f"列出文件失败：{str(e)}"


class DeleteFileInput(BaseModel):
    filename: str = Field(description="要删除的文件名")


@tool(args_schema=DeleteFileInput)
def delete_file(filename: str) -> str:
    """
    删除工作目录中的指定文件。
    只能删除工作目录内的文件，不能删除系统文件。
    """
    file_path, error = resolve_safe_path(filename)
    if error:
        return error

    if not file_path.exists():
        return f"文件不存在：{filename}"

    if not file_path.is_file():
        return f"路径不是文件，不能删除：{filename}"

    try:
        file_path.unlink()
        return f"✅ 已成功删除文件：{filename}"
    except Exception as e:
        return f"删除失败：{str(e)}"


# ============================================================
# 演示
# ============================================================

def demo_direct_file_operations():
    """直接调用文件工具演示。"""
    print(f"=== 1. 直接文件操作演示 ===")
    print(f"工作目录：{SAFE_WORKSPACE}\n")

    # 写入文件
    result = write_file.invoke({
        "filename": "hello.txt",
        "content": "你好，AI学习者！\n欢迎来到文件操作工具的演示。\n",
        "mode": "overwrite"
    })
    print(f"写入操作：{result}")

    # 追加内容
    result = write_file.invoke({
        "filename": "hello.txt",
        "content": f"\n--- 追加时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n",
        "mode": "append"
    })
    print(f"追加操作：{result}")

    # 读取文件
    result = read_file.invoke({"filename": "hello.txt"})
    print(f"\n读取结果：\n{result}")

    # 创建 JSON 文件
    data = json.dumps({"name": "AI学习", "level": "初级", "chapters": 8}, ensure_ascii=False, indent=2)
    write_file.invoke({"filename": "data.json", "content": data})

    # 列出所有文件
    result = list_workspace_files.invoke({"pattern": "*"})
    print(f"\n工作目录文件列表：\n{result}\n")


def demo_llm_file_agent(llm: ChatOpenAI):
    """演示 LLM 使用文件工具完成任务。"""
    print("=== 2. LLM 文件操作 Agent ===\n")

    tools = [read_file, write_file, list_workspace_files, delete_file]
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    def run_with_tools(user_message: str):
        """执行工具调用循环。"""
        print(f"用户：{user_message}")
        messages = [HumanMessage(content=user_message)]

        # 最多执行 10 轮，防止无限循环
        max_iterations = 10
        for _ in range(max_iterations):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                print(f"AI：{response.content}\n")
                break

            # 执行工具调用
            tool_messages = []
            for tc in response.tool_calls:
                tool_name = tc["name"]
                print(f"  → 调用工具：{tool_name}({tc['args']})")
                # 防御性检查：LLM 有时会幻觉出不存在的工具名
                if tool_name not in tool_map:
                    result = f"未知工具：{tool_name}，可用工具：{list(tool_map.keys())}"
                else:
                    result = tool_map[tool_name].invoke(tc["args"])
                result_str = str(result)
                print(f"    结果：{result_str[:100]}{'...' if len(result_str) > 100 else ''}")
                tool_messages.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))

            messages.extend(tool_messages)
        else:
            print("⚠️ 达到最大迭代次数，停止执行\n")

    # 测试任务
    run_with_tools("帮我创建一个名为 'notes.txt' 的文件，内容是：'今日学习：LangChain工具调用'")
    run_with_tools("列出当前工作目录中的所有文件")
    run_with_tools("读取 hello.txt 文件的内容并告诉我里面写了什么")


def cleanup():
    """清理演示创建的文件。"""
    for f in SAFE_WORKSPACE.glob("*"):
        if f.is_file():
            f.unlink()
    print(f"\n已清理工作目录：{SAFE_WORKSPACE}")


def main():
    llm = create_llm()

    demo_direct_file_operations()
    demo_llm_file_agent(llm)

    # 可选：清理演示文件
    # cleanup()

    print("\n所有文件操作工具示例运行完成！")
    print(f"演示文件保存在：{SAFE_WORKSPACE}")


if __name__ == "__main__":
    main()
