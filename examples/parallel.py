"""
实验 6：把多个章节分发给同一个 worker，再收集成有序报告。

运行：
    python -m examples.parallel

执行结构：
    START → fan_out → 多个 summarize 任务 → collect → END
                   ↘ 空列表直接 collect ↗

“同一个节点”可以接收多份不同输入，分别产生独立的任务。
本例只把章节截取前 24 个字符，不做真正的语义摘要，也不调用模型。
预期输出按原章节顺序显示 1、2、3，而不是依赖任务的完成先后。
"""

# operator.add 对列表的效果是拼接，用作 results 字段的 reducer。
import operator
# Annotated 让一个字段既有类型信息，又附带 reducer 的合并规则。
from typing import Annotated, Literal, TypedDict
from langgraph.graph import END, START, StateGraph
# Send 描述“给目标节点发送这一份输入”，不是直接同步调用节点函数。
from langgraph.types import Send


class Item(TypedDict):
    """一个 worker 的输出记录；index 为最终排序保留来源位置。"""

    # 从 0 开始的章节索引，不是任务完成顺序。
    index: int
    summary: str


class State(TypedDict):
    """主图状态，与单个 worker 输入区分开。"""

    # 主图拿到全部章节，fan_out 再逐个分发。
    chapters: list[str]

    # 多个任务会在同一步提交 results 更新，因此这里需要 reducer。
    # 返回 [{"index": 0, ...}] 和 [{"index": 1, ...}] 时，按列表拼接合并。
    # 如果是无 reducer 的普通字段，同一步的多个写入通常会报冲突，
    # 不能依赖“后完成的任务覆盖前一个”。
    results: Annotated[list[Item], operator.add]

    # collect 最后产生的展示文本，使用普通覆盖规则。
    report: str


class WorkerInput(TypedDict):
    """分发给单个 summarize 任务的数据，不需要携带全部主图状态。"""

    index: int
    chapter: str


def fan_out(state: State) -> list[Send] | Literal["collect"]:
    """根据章节数动态决定创建多少个任务；空输入也有明确的结束路径。"""

    # 空列表不会创建任何 summarize 任务，因此不能只等不存在的 worker 触发汇总。
    # 明确返回 collect，让它生成空报告并正常结束。
    if not state["chapters"]:
        return "collect"

    # enumerate 给每个章节附带稳定索引：
    # ["甲", "乙"] 变成 (0, "甲")、(1, "乙")。
    # 每个 Send 都指定目标 summarize，并提供该任务独有的 index、chapter。
    # 这是路由函数，返回 Send 列表；不要在这里直接执行模型或副作用操作。
    return [Send("summarize", {"index": i, "chapter": text})
            for i, text in enumerate(state["chapters"])]


def summarize(state: WorkerInput) -> dict:
    """处理一份 worker 输入，只返回主图需要收集的那个结果增量。"""

    # [:24] 是字符串切片，最多保留前 24 个字符；不足 24 个就原样保留。
    # 列表里只放本任务产生的一项，不返回主图已累积的全部结果。
    # 主图 results 的 operator.add 会负责把各任务返回的小列表合在一起。
    return {"results": [{"index": state["index"], "summary": state["chapter"][:24]}]}


def collect(state: State) -> dict:
    """按来源 index 排序，再生成明确顺序的最终报告。"""

    # .get("results", []) 为没有结果时提供空列表。
    # sorted 返回新列表，不会原地修改 state 中的已有结果列表。
    # lambda item: item["index"] 指定排序键，避免依赖并行完成顺序。
    ordered = sorted(state.get("results", []), key=lambda item: item["index"])

    # 人类阅读的章节号从 1 开始，因此显示 index + 1。
    # "\n".join(...) 用换行连接每条文本；空列表会产生空字符串。
    return {"report": "\n".join(f"{item['index'] + 1}. {item['summary']}" for item in ordered)}


def build_graph():
    """注册一个 worker 节点，让条件路由产生它的多份任务实例。"""

    builder = StateGraph(State)
    builder.add_node("summarize", summarize)
    builder.add_node("collect", collect)

    # START 上的条件路由会读取初始输入：
    # 有章节时返回 Send 列表，无章节时返回字符串 "collect"。
    # 第三个参数列出可能目标，便于图理解和展示路由去向。
    builder.add_conditional_edges(START, fan_out, ["summarize", "collect"])

    # 在本图这个同层 worker 结构中，任务更新合并后进入 collect 汇总。
    # 不要由此推断任意不等长、带循环的复杂分支都能自动成为正确的 join。
    builder.add_edge("summarize", "collect")
    builder.add_edge("collect", END)
    return builder.compile()


if __name__ == "__main__":
    result = build_graph().invoke(
        # results 是本次任务的收集容器，开始为空。
        # 这些章节很短，切片后仍会完整显示。
        {"chapters": ["第一章：状态", "第二章：持久化", "第三章：人工介入"], "results": []},

        # 最多允许的并发度是 2；3 份任务不意味着同时启动 3 份工作。
        # 并发度不是跨进程的供应商配额控制，也不保证 CPU 计算必然加速。
        config={"max_concurrency": 2},
    )
    print(result["report"])

# 练习：分别提供 []、1 个章节、3 个章节，检查都能结束。
# 再给某个 worker 制造失败，明确产品应全失败，还是允许不完整结果。
# 真实模型摘要还必须另外评测内容质量，不能从本实验的调度成功推出。
