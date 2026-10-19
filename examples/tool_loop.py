"""
实验 3：逐条看懂模型请求工具、工具执行和最终回复之间的协议。

运行：
    python -m examples.tool_loop

完整路径：
    START → model → tools → model → END

这里的 model 是固定脚本：先请求 multiply(6, 7)，收到 ToolMessage 后结束。
它不理解任意问题，也不访问真实模型。这样能把工具协议稳定地展示出来。
预期消息顺序：
    HumanMessage → AIMessage(tool_calls) → ToolMessage("42") → AIMessage

重点看 tool_calls 中的 id 与 ToolMessage.tool_call_id 如何对应。
"""

# AIMessage 表示助手消息，既可能是最终文本，也可能携带工具请求。
# ToolMessage 表示工具执行结果，用 tool_call_id 关联先前的请求。
from langchain_core.messages import AIMessage, ToolMessage

# @tool 将函数包装成工具；函数签名生成参数 schema，文档字符串提供描述。
# 因此下面 multiply 的短文档字符串会成为工具元数据，保留其原有内容。
from langchain_core.tools import tool

from langgraph.graph import END, START, MessagesState, StateGraph

# ToolNode 负责读取并执行请求；tools_condition 负责检查是否还有工具请求。
# 分工不同：绑定或声明一个工具，并不等于已经执行了它。
from langgraph.prebuilt import ToolNode, tools_condition


@tool
def multiply(a: int, b: int) -> int:
    """将两个绝对值不超过一百万的整数相乘。"""

    # a、b 的类型声明进入工具的参数 schema；实际包装层会解析和校验输入。
    # 不应把类型提示本身误当成所有业务限制已经自动完成。
    # 绝对值上限是我们亲自写出的业务规则，负数也按绝对值检查。
    if abs(a) > 1_000_000 or abs(b) > 1_000_000:
        raise ValueError("参数超出教学示例允许范围")

    # 只执行这一种确定的运算，不接收任意 Python 表达式，也不用 eval。
    return a * b


def scripted_model(state: MessagesState) -> dict:
    """模拟模型的两种输出：请求一次工具，或者根据工具结果给出最终回复。"""

    # 每次进入模型节点，都查看当前最后一条消息。
    # 第一次是用户消息；经过 tools 节点后，最后一条会变成 ToolMessage。
    last = state["messages"][-1]

    # isinstance 判断最后一条是否已经是工具结果。
    # 走到这里时，不再发起新工具调用，所以稍后的条件路由会选择 END。
    if isinstance(last, ToolMessage):
        return {"messages": [AIMessage(content=f"工具返回：{last.content}")]}

    # 第一次进入时，构造一条“请求执行工具”的助手消息。
    # content 为空并不代表什么都没有发生：信息在 tool_calls 字段中。
    # 字段逐个看：
    # - name：需要执行的工具名，与 @tool 包装后的 multiply 名称对应；
    # - args：工具输入字典，键名必须与参数 schema 对应；
    # - id：本次调用请求的关联标识，工具结果会用同一个值回应；
    # - type：说明这是已经解析好的 tool_call 结构。
    # 固定 id 只用于这个不持久化、单次调用的演示；真实调用应使用唯一 ID。
    return {"messages": [AIMessage(content="", tool_calls=[{
        "name": "multiply", "args": {"a": 6, "b": 7},
        "id": "call-demo-1", "type": "tool_call",
    }])]}


def build_graph():
    """搭建工具循环；决定何时终止的是条件路由，不是固定循环次数。"""

    builder = StateGraph(MessagesState)

    # 同一个 model 节点会执行两次，但两次读取的最后一条消息不同。
    builder.add_node("model", scripted_model)

    # 这里只注册执行层真正允许的工具。模型无法凭空执行任意 Python 函数。
    # ToolNode 会执行请求，并产生关联到原请求 ID 的 ToolMessage。
    builder.add_node("tools", ToolNode([multiply]))

    # 首次先运行 model，让它决定是否需要工具。
    builder.add_edge(START, "model")

    # tools_condition 检查最后一条 AIMessage 是否包含 tool_calls：
    # 有则返回 "tools"，没有则返回 END。
    # 它默认使用节点名 "tools"；若改节点名，需要相应配置路由映射。
    builder.add_conditional_edges("model", tools_condition)

    # 工具结果不直接作为最终回复，而是回到 model，让其读取结果再决定下一步。
    builder.add_edge("tools", "model")
    return builder.compile()


if __name__ == "__main__":
    # 新建图没有接入 checkpointer，每次 invoke 是独立的本次输入流程。
    result = build_graph().invoke(
        {"messages": [("human", "演示 6 × 7 的工具调用过程")]},

        # recursion_limit 限制图的执行步数，用于阻止失控循环。
        # 它不是模型 token、调用费用或精确工具次数的预算。
        config={"recursion_limit": 10},
    )

    # 打印完整消息轨迹，不能只看最后一句“42”就推断工具一定执行过。
    for message in result["messages"]:
        # repr 能让空字符串显示成 ''，便于观察工具请求的 content 是空的。
        # getattr(..., "") 在消息没有 tool_calls 属性时返回空字符串，
        # 避免在 HumanMessage / ToolMessage 上直接读该属性而出错。
        print(message.type, repr(message.content), getattr(message, "tool_calls", ""))

# 练习：先把 recursion_limit 改成 1，理解为何会提前超限。
# 再把脚本请求的 a 改成超出范围的值，观察工具失败如何呈现。
