"""SQLite 跨进程审批。publish 只生成模拟回执，没有任何外部写操作。"""
import argparse
import json
from pathlib import Path
from typing import Literal, TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    request_id: str
    draft: str
    approved: bool
    receipt: str


def review(state: State) -> Command[Literal["publish", "cancel"]]:
    decision = interrupt({
        "request_id": state["request_id"], "draft": state["draft"],
        "question": "是否批准这份草稿？",
    })
    if not isinstance(decision, dict) or type(decision.get("approved")) is not bool:
        raise ValueError("审批结果必须包含布尔类型的 approved")
    approved = decision["approved"]
    return Command(update={"approved": approved}, goto="publish" if approved else "cancel")


def build_graph(checkpointer):
    builder = StateGraph(State)
    builder.add_node("review", review)
    builder.add_node("publish", lambda s: {"receipt": f"模拟发布成功：{s['request_id']}"})
    builder.add_node("cancel", lambda s: {"receipt": "已取消"})
    builder.add_edge(START, "review")
    # review 用 Command 控制路由，不再添加固定的出边。
    builder.add_edge("publish", END)
    builder.add_edge("cancel", END)
    return builder.compile(checkpointer=checkpointer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "approve", "reject", "inspect"])
    parser.add_argument("--db", default=".local/approval.sqlite")
    parser.add_argument("--thread", default="approval-demo")
    parser.add_argument("--draft", default="文档助手生成的示例草稿")
    args = parser.parse_args()
    path = Path(args.db)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {"configurable": {"thread_id": args.thread}}
    with SqliteSaver.from_conn_string(str(path)) as saver:
        app = build_graph(saver)
        snapshot = app.get_state(config)
        if args.action == "inspect":
            result = {"values": snapshot.values, "next": snapshot.next}
        elif args.action == "start":
            if snapshot.values:
                parser.error("这个 thread 已存在，请换一个 --thread 或恢复已有审批")
            result = app.invoke({"request_id": args.thread, "draft": args.draft}, config)
        else:
            if not any(task.interrupts for task in snapshot.tasks):
                parser.error("这个 thread 没有待处理的审批")
            result = app.invoke(Command(resume={"approved": args.action == "approve"}), config)
        if "__interrupt__" in result:
            result["__interrupt__"] = [
                {"id": item.id, "value": item.value} for item in result["__interrupt__"]
            ]
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
