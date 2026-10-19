"""离线文档工作流：抽取原文行、检查引用、按规则审核。没有 LLM 能力。"""
from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict, Field, StrictBool


class DocumentInput(TypedDict):
    text: str
    force_review: bool


class DocumentOutput(TypedDict):
    report: str
    status: str


class State(DocumentInput, total=False):
    evidence: list[dict]
    draft: str
    needs_review: bool
    approved: bool
    report: str
    status: str


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=20_000)
    force_review: StrictBool = False


def extract(state: State) -> dict:
    data = InputModel.model_validate(state)
    if not data.text.strip():
        raise ValueError("文档不能只包含空白")
    evidence = [{"line": i, "quote": line.strip()}
                for i, line in enumerate(data.text.splitlines(), start=1) if line.strip()][:3]
    return {"evidence": evidence}


def validate_evidence(text: str, evidence: list[dict]) -> bool:
    lines = text.splitlines()
    return bool(evidence) and all(
        type(item.get("line")) is int
        and 1 <= item["line"] <= len(lines)
        and isinstance(item.get("quote"), str)
        and bool(item["quote"].strip())
        and item["quote"] in lines[item["line"] - 1]
        for item in evidence
    )


def draft(state: State) -> dict:
    if not validate_evidence(state["text"], state["evidence"]):
        raise ValueError("引用不对应原文，停止生成报告")
    report = "\n".join(f"- L{item['line']}: {item['quote']}" for item in state["evidence"])
    return {"draft": report}


def quality_gate(state: State) -> dict:
    # 这是演示路由的启发式规则，不是语义质量评分，也不是模型置信度。
    return {"needs_review": state.get("force_review", False) or len(state["evidence"]) < 2}


def review(state: State) -> dict:
    decision = interrupt({"draft": state["draft"], "question": "是否批准？可提供修订稿。"})
    if not isinstance(decision, dict) or type(decision.get("approved")) is not bool:
        raise ValueError("approved 必须是布尔值")
    revised = decision.get("corrected_report", state["draft"])
    if not isinstance(revised, str):
        raise ValueError("corrected_report 必须是字符串")
    return {"approved": decision["approved"], "report": revised if decision["approved"] else ""}


def finalize(state: State) -> dict:
    if not state["needs_review"]:
        return {"report": state["draft"], "status": "auto_approved"}
    return {"status": "human_approved" if state["approved"] else "rejected"}


def build_graph(checkpointer=None):
    builder = StateGraph(State, input_schema=DocumentInput, output_schema=DocumentOutput)
    for name, node in [("extract", extract), ("draft", draft), ("gate", quality_gate),
                       ("review", review), ("finalize", finalize)]:
        builder.add_node(name, node)
    builder.add_edge(START, "extract")
    builder.add_edge("extract", "draft")
    builder.add_edge("draft", "gate")
    builder.add_conditional_edges(
        "gate", lambda state: "review" if state["needs_review"] else "finalize",
        ["review", "finalize"],
    )
    builder.add_edge("review", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


if __name__ == "__main__":
    app = build_graph()
    cfg = lambda tid: {"configurable": {"thread_id": tid}}
    print("自动路径：", app.invoke(
        {"text": "LangGraph 编排有状态流程。\nCheckpointer 保存运行状态。", "force_review": False}, cfg("auto"),
    ))
    paused = app.invoke({"text": "需要审核的短文。", "force_review": True}, cfg("manual"))
    print("暂停：", paused["__interrupt__"][0].value)
    print("批准：", app.invoke(Command(resume={"approved": True}), cfg("manual")))
    app.invoke({"text": "另一篇短文。", "force_review": True}, cfg("reject"))
    print("拒绝：", app.invoke(Command(resume={"approved": False}), cfg("reject")))
