"""运行：python -m examples.minimal"""
from typing import TypedDict
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    name: str
    greeting: str


def greet(state: State) -> dict:
    name = state["name"].strip()
    if not name:
        raise ValueError("name 不能为空")
    return {"greeting": f"你好，{name}！"}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("greet", greet)
    builder.add_edge(START, "greet")
    builder.add_edge("greet", END)
    return builder.compile()


if __name__ == "__main__":
    print(build_graph().invoke({"name": " 小明 "})["greeting"])
