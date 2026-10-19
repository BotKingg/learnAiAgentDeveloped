"""运行：python -m examples.streaming。稳定的 StreamPart v2 格式。"""
from typing import TypedDict
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    value: int


def work(state: State) -> dict:
    writer = get_stream_writer()
    writer({"stage": "started"})
    return {"value": state["value"] + 1}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("work", work)
    builder.add_edge(START, "work")
    builder.add_edge("work", END)
    return builder.compile()


if __name__ == "__main__":
    for part in build_graph().stream(
        {"value": 0}, stream_mode=["updates", "custom"], version="v2",
    ):
        print(part["type"], part["data"])
