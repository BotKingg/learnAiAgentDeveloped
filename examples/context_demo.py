"""运行：python -m examples.context_demo。context 是本次调用注入的依赖。"""
from dataclasses import dataclass
from typing import TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime


@dataclass(frozen=True)
class Context:
    user_id: str


class State(TypedDict):
    name: str
    greeting: str


def greet(state: State, runtime: Runtime[Context]) -> dict:
    return {"greeting": f"{runtime.context.user_id}：你好，{state['name']}"}


def build_graph():
    builder = StateGraph(State, context_schema=Context)
    builder.add_node("greet", greet)
    builder.add_edge(START, "greet")
    builder.add_edge("greet", END)
    return builder.compile()


if __name__ == "__main__":
    print(build_graph().invoke({"name": "小明"}, context=Context(user_id="u-001")))
