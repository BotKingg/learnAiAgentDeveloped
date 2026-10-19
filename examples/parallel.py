"""运行：python -m examples.parallel。切片是占位逻辑，不是真正的语义摘要。"""
import operator
from typing import Annotated, Literal, TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class Item(TypedDict):
    index: int
    summary: str


class State(TypedDict):
    chapters: list[str]
    results: Annotated[list[Item], operator.add]
    report: str


class WorkerInput(TypedDict):
    index: int
    chapter: str


def fan_out(state: State) -> list[Send] | Literal["collect"]:
    if not state["chapters"]:
        return "collect"
    return [Send("summarize", {"index": i, "chapter": text})
            for i, text in enumerate(state["chapters"])]


def summarize(state: WorkerInput) -> dict:
    return {"results": [{"index": state["index"], "summary": state["chapter"][:24]}]}


def collect(state: State) -> dict:
    ordered = sorted(state.get("results", []), key=lambda item: item["index"])
    return {"report": "\n".join(f"{item['index'] + 1}. {item['summary']}" for item in ordered)}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("summarize", summarize)
    builder.add_node("collect", collect)
    builder.add_conditional_edges(START, fan_out, ["summarize", "collect"])
    builder.add_edge("summarize", "collect")
    builder.add_edge("collect", END)
    return builder.compile()


if __name__ == "__main__":
    result = build_graph().invoke(
        {"chapters": ["第一章：状态", "第二章：持久化", "第三章：人工介入"], "results": []},
        config={"max_concurrency": 2},
    )
    print(result["report"])
