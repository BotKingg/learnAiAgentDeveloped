"""运行：python -m examples.tool_loop。模型节点是固定脚本，不会理解任意问题。"""
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


@tool
def multiply(a: int, b: int) -> int:
    """将两个绝对值不超过一百万的整数相乘。"""
    if abs(a) > 1_000_000 or abs(b) > 1_000_000:
        raise ValueError("参数超出教学示例允许范围")
    return a * b


def scripted_model(state: MessagesState) -> dict:
    last = state["messages"][-1]
    if isinstance(last, ToolMessage):
        return {"messages": [AIMessage(content=f"工具返回：{last.content}")]}
    return {"messages": [AIMessage(content="", tool_calls=[{
        "name": "multiply", "args": {"a": 6, "b": 7},
        "id": "call-demo-1", "type": "tool_call",
    }])]}


def build_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("model", scripted_model)
    builder.add_node("tools", ToolNode([multiply]))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")
    return builder.compile()


if __name__ == "__main__":
    result = build_graph().invoke(
        {"messages": [("human", "演示 6 × 7 的工具调用过程")]},
        config={"recursion_limit": 10},
    )
    for message in result["messages"]:
        print(message.type, repr(message.content), getattr(message, "tool_calls", ""))
