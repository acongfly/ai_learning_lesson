"""
第04章 - Tools工具调用：本地命令执行工具
==========================================
本示例演示如何创建执行本地系统命令的工具。

⚠️  安全警告：
执行系统命令存在严重的安全风险！本示例包含多层安全检查：
1. 命令白名单（只允许安全的只读命令）
2. 危险命令黑名单
3. 超时限制
4. 沙箱化执行

在生产环境中请谨慎使用，建议在容器或沙箱环境中运行。

学习要点：
1. 创建执行系统命令的工具
2. 多层安全检查机制
3. 命令超时处理
4. 目录遍历工具
5. 文件内容读取工具
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

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
# 安全配置
# ============================================================

# ⚠️  安全说明：
# 本示例采用白名单 + 黑名单双重检查。
# 白名单是主要防线：只有在 ALLOWED_COMMANDS 中的命令才能执行。
# 黑名单作为额外保护，拦截关键危险词。
#
# 重要提示：在生产环境中，命令执行工具应在以下环境中运行：
# - Docker 容器（资源隔离）
# - 专用沙箱（如 firejail）
# - 只读文件系统
# 不应该在宿主机上直接执行用户提供的命令。

# 允许执行的命令白名单（只读、无破坏性操作）
ALLOWED_COMMANDS = {
    "ls", "dir", "pwd", "echo", "cat", "head", "tail",
    "find", "grep", "wc", "date", "whoami", "hostname",
    "uname", "df", "du", "ps", "env", "python", "python3",
}

# 明确禁止的危险命令黑名单
DANGEROUS_COMMANDS = {
    "rm", "rmdir", "del", "format", "dd", "mkfs",
    "shutdown", "reboot", "halt", "poweroff",
    "kill", "pkill", "killall",
    "chmod", "chown", "sudo", "su",
    "curl", "wget", "nc", "netcat",
    "ssh", "scp", "rsync",
    ">", ">>",                                          # 重定向写入
}

# 命令执行超时（秒）
COMMAND_TIMEOUT = 10


def is_safe_command(command: str) -> tuple[bool, str]:
    """
    检查命令是否安全。
    返回 (是否安全, 拒绝原因)
    """
    if not command.strip():
        return False, "命令为空"

    # 获取命令的第一个词（命令名）
    parts = command.strip().split()
    cmd_name = parts[0].lower()

    # 去掉路径前缀（如 /bin/ls → ls）
    if "/" in cmd_name:
        cmd_name = cmd_name.split("/")[-1]

    # 检查黑名单
    for dangerous in DANGEROUS_COMMANDS:
        if dangerous in command.lower():
            return False, f"命令包含危险操作：{dangerous}"

    # 检查白名单
    if cmd_name not in ALLOWED_COMMANDS:
        return False, f"命令 '{cmd_name}' 不在允许列表中"

    # 检查管道和重定向中的危险命令
    if "|" in command:
        for pipe_cmd in command.split("|")[1:]:
            pipe_name = pipe_cmd.strip().split()[0].lower() if pipe_cmd.strip() else ""
            if pipe_name and pipe_name not in ALLOWED_COMMANDS:
                return False, f"管道命令 '{pipe_name}' 不在允许列表中"

    return True, ""


# ============================================================
# 工具定义
# ============================================================

class RunCommandInput(BaseModel):
    command: str = Field(
        description="要执行的系统命令（仅支持安全命令：ls, pwd, echo, date, whoami, df 等）"
    )


@tool(args_schema=RunCommandInput)
def run_safe_command(command: str) -> str:
    """
    执行安全的只读系统命令。
    只允许以下命令：ls, pwd, echo, date, whoami, hostname, uname, df 等。
    不允许修改文件、执行危险操作或网络操作。
    """
    # 安全检查
    safe, reason = is_safe_command(command)
    if not safe:
        return f"⛔ 命令被拒绝：{reason}"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,                    # 超时保护
            cwd=os.path.expanduser("~"),               # 在用户主目录执行
        )

        output = result.stdout.strip()
        error = result.stderr.strip()

        if result.returncode != 0:
            return f"命令执行失败（退出码 {result.returncode}）：{error}"

        return output if output else "命令执行成功（无输出）"

    except subprocess.TimeoutExpired:
        return f"命令超时（超过 {COMMAND_TIMEOUT} 秒）"
    except Exception as e:
        return f"命令执行错误：{str(e)}"


class ListDirectoryInput(BaseModel):
    path: str = Field(
        default=".",
        description="要列出的目录路径（默认为当前目录）"
    )
    show_hidden: bool = Field(
        default=False,
        description="是否显示隐藏文件（以.开头的文件）"
    )


@tool(args_schema=ListDirectoryInput)
def list_directory(path: str = ".", show_hidden: bool = False) -> str:
    """
    列出指定目录的文件和子目录。
    显示文件名、类型（文件/目录）和大小。
    """
    try:
        target_path = Path(path).resolve()

        # 安全检查：防止遍历到系统关键目录
        forbidden_paths = ["/etc/shadow", "/etc/passwd", "/root", "/sys", "/proc"]
        path_str = str(target_path)
        for forbidden in forbidden_paths:
            if path_str.startswith(forbidden):
                return f"⛔ 禁止访问该目录：{path}"

        if not target_path.exists():
            return f"目录不存在：{path}"

        if not target_path.is_dir():
            return f"路径不是目录：{path}"

        items = []
        for item in sorted(target_path.iterdir()):
            # 跳过隐藏文件（除非明确要求显示）
            if item.name.startswith(".") and not show_hidden:
                continue

            if item.is_dir():
                items.append(f"📁 {item.name}/")
            elif item.is_file():
                size = item.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size/1024/1024:.1f}MB"
                items.append(f"📄 {item.name} ({size_str})")

        if not items:
            return f"目录为空：{path}"

        return f"目录：{target_path}\n" + "\n".join(items)

    except PermissionError:
        return f"权限不足，无法访问：{path}"
    except Exception as e:
        return f"错误：{str(e)}"


class ReadFileInput(BaseModel):
    file_path: str = Field(description="要读取的文件路径")
    max_lines: int = Field(
        default=50,
        description="最大读取行数（默认50行，防止读取过大文件）"
    )


@tool(args_schema=ReadFileInput)
def read_text_file(file_path: str, max_lines: int = 50) -> str:
    """
    读取文本文件的内容（限制最大行数）。
    支持 .txt、.py、.md、.json、.yaml、.csv 等文本格式。
    不支持读取二进制文件或系统敏感文件。
    """
    try:
        target_path = Path(file_path).resolve()

        # 安全检查：只允许读取特定类型的文件
        allowed_extensions = {
            ".txt", ".py", ".md", ".json", ".yaml", ".yml",
            ".csv", ".log", ".cfg", ".ini", ".toml", ".rst"
        }
        if target_path.suffix.lower() not in allowed_extensions:
            return f"⛔ 不支持读取该类型的文件：{target_path.suffix}"

        # 安全检查：禁止读取敏感文件
        sensitive_patterns = [".env", ".secret", "password", "credential", "private_key"]
        for pattern in sensitive_patterns:
            if pattern in target_path.name.lower():
                return f"⛔ 禁止读取敏感文件"

        if not target_path.exists():
            return f"文件不存在：{file_path}"

        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)
        if total_lines > max_lines:
            content = "".join(lines[:max_lines])
            return f"文件：{target_path.name}（共{total_lines}行，显示前{max_lines}行）\n\n{content}\n\n[...文件已截断...]"
        else:
            return f"文件：{target_path.name}（共{total_lines}行）\n\n{''.join(lines)}"

    except PermissionError:
        return f"权限不足，无法读取：{file_path}"
    except Exception as e:
        return f"读取文件失败：{str(e)}"


# ============================================================
# 演示
# ============================================================

def demo_security_check():
    """演示安全检查机制。"""
    print("=== 1. 命令安全检查演示 ===\n")

    test_commands = [
        "ls -la",                       # 允许
        "pwd",                          # 允许
        "date",                         # 允许
        "rm -rf /",                     # 拒绝：危险命令
        "cat /etc/passwd",              # 拒绝（需要按白名单看）
        "curl https://evil.com",        # 拒绝：网络命令
        "shutdown -h now",              # 拒绝：危险操作
        "ls | grep .py",                # 允许
    ]

    for cmd in test_commands:
        safe, reason = is_safe_command(cmd)
        status = "✅ 允许" if safe else f"⛔ 拒绝（{reason}）"
        print(f"  命令：{cmd!r:40s} → {status}")
    print()


def demo_tools_directly():
    """直接调用工具演示。"""
    print("=== 2. 直接调用本地工具 ===\n")

    # 列出当前目录
    result = list_directory.invoke({"path": ".", "show_hidden": False})
    print("当前目录内容：")
    print(result)
    print()

    # 执行安全命令
    result = run_safe_command.invoke({"command": "date"})
    print(f"当前日期时间：{result}")

    result = run_safe_command.invoke({"command": "echo Hello from AI Agent!"})
    print(f"echo 命令结果：{result}")
    print()


def demo_llm_with_local_tools(llm: ChatOpenAI):
    """演示 LLM 使用本地工具回答问题。"""
    print("=== 3. LLM 使用本地工具 ===\n")

    tools = [run_safe_command, list_directory, read_text_file]
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    question = "当前目录有哪些文件？"
    print(f"用户问题：{question}\n")

    messages = [HumanMessage(content=question)]
    response = llm_with_tools.invoke(messages)

    # 处理工具调用
    if response.tool_calls:
        tool_messages = []
        for tc in response.tool_calls:
            result = tool_map[tc["name"]].invoke(tc["args"])
            print(f"工具 [{tc['name']}] 结果：\n{result}\n")
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        # 获取最终回答
        final = llm_with_tools.invoke(messages + [response] + tool_messages)
        print(f"LLM 最终回答：{final.content}")
    else:
        print(f"LLM 直接回答：{response.content}")


def main():
    llm = create_llm()

    demo_security_check()
    demo_tools_directly()
    demo_llm_with_local_tools(llm)

    print("\n所有本地命令工具示例运行完成！")


if __name__ == "__main__":
    main()
