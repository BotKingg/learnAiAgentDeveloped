"""
实验 10（可选联网）：用真实模型替换前面的固定脚本模型。

在仓库根目录运行：
    python -m examples.real_agent

运行前必须安全配置：
- OPENAI_API_KEY：由安全环境变量注入，不写进代码、Git 或聊天；
- LLM_MODEL：账号实际可用且支持工具调用的模型 ID。

这会产生供应商费用。本仓库测试使用脚本模型验证框架工具循环，
并没有替你验证真实供应商的可用性、实际效果或调用费用。

先理解 tool_loop.py，再看本文件：工具协议相同，
但“何时调用工具”的决策由真实模型完成，不再是固定返回的脚本。
"""

# 只检查环境变量是否存在，不打印密钥内容。
import os

# 新项目的标准 Agent 入口位于 langchain.agents。
# create_agent 在内部基于 LangGraph 构建模型—工具循环。
from langchain.agents import create_agent

# 模型适配器负责把 LangChain 消息与供应商 API 请求对应起来。
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

# 复用已经解释过、带范围限制的乘法工具。
# 导入 tool_loop 不会执行其 __main__ 演示，所以不会意外跑一次脚本图。
from examples.tool_loop import multiply


def build_agent(model, checkpointer):
    """把模型和检查点器作为依赖传入，既方便真实运行，也方便离线测试。

    本函数只负责构建，不写死 ChatOpenAI。
    测试会传入脚本模型，因此可以检查工具循环而不付费请求模型服务。
    """

    return create_agent(
        # tools 列出执行层允许的工具，不是模型生成任意代码的授权。
        model=model, tools=[multiply],

        # 当前参数名为 system_prompt，不照搬旧 create_react_agent 的 prompt。
        # 提示词描述期望行为，但不能保证模型每次都一定按期望调用工具；
        # 真正的参数边界由工具和业务执行层检查。
        system_prompt="你是中文计算助手。计算乘法时使用 multiply 工具。不要编造工具结果。",

        # 同一实例与相同 thread_id 才能延续本进程内的会话检查点。
        checkpointer=checkpointer,
    )


if __name__ == "__main__":
    # 任一配置缺失都明确停止，不编造默认模型名，也不把密钥写入源文件。
    # .get 只用于检查是否有非空值，错误提示不包含变量值。
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("LLM_MODEL"):
        raise SystemExit("请先安全配置 OPENAI_API_KEY 和支持工具调用的 LLM_MODEL。")

    # model 从环境中读取，避免把某个可能退役的模型 ID 当成永远可用。
    # timeout 限制模型客户端请求的等待；不是整张图的总执行时间预算。
    # max_retries=1 是客户端对适用错误的额外重试设置；
    # 一个 Agent 任务仍可能包含多次模型调用，费用不能只按一条请求估算。
    model = ChatOpenAI(model=os.environ["LLM_MODEL"], timeout=30, max_retries=1)

    agent = build_agent(model, InMemorySaver())

    # 模型通常先输出 tool_calls，工具执行后再调用模型生成最后回复。
    # 但实际模型有随机性、供应商差异，不能仅凭提示词就认定工具已被使用。
    result = agent.invoke(
        {"messages": [("human", "请使用工具计算 6 × 7，并用中文回答。") ]},
        config={"configurable": {"thread_id": "real-agent-demo"}, "recursion_limit": 10},
    )

    # 查看完整协议轨迹，而不是只打印最后一句“42”：
    # 找到工具请求，再找到对应 tool_call_id 的 ToolMessage。
    for message in result["messages"]:
        print(message.type, repr(message.content), getattr(message, "tool_calls", ""))

# 练习：先独立调用 multiply.invoke({"a": 6, "b": 7}) 验证工具，再跑 Agent。
# 结果正确与确实使用工具，是两个分别需要检查的事实。
