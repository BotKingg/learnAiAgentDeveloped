"""运行：python -m examples.message_memory；这是回显器，不是 LLM。"""
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.checkpoint.memory import InMemorySaver


def build_graph(checkpointer):
    def echo(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content=f"已收到：{state['messages'][-1].content}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("echo", echo)
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)
    return builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    app = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "chat-a"}}
    app.invoke({"messages": [("human", "我叫小明")]}, config)
    result = app.invoke({"messages": [("human", "第二轮")]}, config)
    print([(m.type, m.content) for m in result["messages"]])
    print("消息数：", len(result["messages"]))
