"""
实验 9：把输入校验、证据定位、条件路由和人工审批接成完整流程。

运行：
    python -m examples.document_workflow

本文件完全离线，采用“抽取原文非空行”的固定规则，不调用 LLM。
它验证工作流和引用位置，不代表已经能理解、评价任意文档。

执行路径：
    START → extract → draft → gate
                               ├→ finalize → END
                               └→ review（暂停）→ finalize → END

main 依次演示三种独立任务：
1. 两行正文，不强制审核：auto_approved；
2. 短文，人工批准：human_approved；
3. 另一篇短文，人工拒绝：rejected，报告为空。

每份文档使用不同的 thread_id；本例没有实现“同一 thread 重开另一份文档”。
输入、输出 schema 用于接口组织，不构成秘密保护或用户权限隔离。
"""

from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

# TypedDict 主要描述字段，Pydantic 在这里才真正做运行时输入检查。
# StrictBool 避免把字符串 "false"、整数等宽松转换成审批控制布尔值。
from pydantic import BaseModel, ConfigDict, Field, StrictBool


class DocumentInput(TypedDict):
    """调用者向图提交的文档任务输入。"""

    text: str             # 原文；演示限制长度，不在此实现大文件存储。
    force_review: bool    # 是否明确要求走人工路径。


class DocumentOutput(TypedDict):
    """最终 invoke 对外返回的业务字段；不包含全部内部处理数据。"""

    report: str           # 自动稿、人工修订稿，或拒绝后的空字符串。
    status: str           # auto_approved、human_approved、rejected。


class State(DocumentInput, total=False):
    """内部状态在输入基础上逐步产生更多字段。

    total=False 表示“本类新增的字段在类型描述中可缺省”，不会自动创建默认值。
    它不会把继承来的 DocumentInput 字段全部改成运行时自动可用。
    节点必须在读取某个中间字段前，保证前置节点已经写入它。
    """

    # 每项形如 {"line": 2, "quote": "原文中的一句话"}。
    evidence: list[dict]
    # 引用核验通过后生成、供人工查看的草稿。
    draft: str
    # gate 写入的分支控制字段。
    needs_review: bool
    # 仅人工路径产生；自动路径不需要读取它。
    approved: bool
    # 对外结果，也属于内部状态的一部分。
    report: str
    status: str


class InputModel(BaseModel):
    """在 extract 节点真正执行的输入检查模型。"""

    # 对送进这个模型的额外字段报错，帮助尽早发现不符合任务边界的数据。
    # 这个检查发生在节点内部；图的输入 schema 本身可能已过滤外层字段。
    model_config = ConfigDict(extra="forbid")

    # 长度按字符串字符数限制，不等于 token 数、文件字节数或费用预算。
    # min_length=1 只能排除空字符串，纯空白还要在 extract 中另查。
    text: str = Field(min_length=1, max_length=20_000)

    # 不提供时模型的默认值是 False；提供时要求真正的布尔值。
    force_review: StrictBool = False


def extract(state: State) -> dict:
    """检查输入，按原始行号抽取前 3 条非空行作为教学用证据。

    输入是当前任务刚开始时的状态，不应混入另一份已完成文档的中间字段。
    因为 InputModel 禁止额外字段，本例使用一份文档一个 thread 的生命周期。
    """

    # 这里才运行 Pydantic 的校验；校验失败会抛异常，阻止后续节点继续。
    data = InputModel.model_validate(state)

    # "   " 长度大于 0，仍可通过上面的 min_length。
    # 再检查 strip 后是否为空，确保不是没有正文的假输入。
    if not data.text.strip():
        raise ValueError("文档不能只包含空白")

    # splitlines 按原文换行切分；enumerate(..., start=1) 保留人类使用的行号。
    # 先编号，再过滤空行，所以保留下来的行号仍然对应原文，而不是重新编号。
    # quote 去掉两侧空白，[:3] 只取前 3 条；并未判断哪 3 条语义上最重要。
    evidence = [{"line": i, "quote": line.strip()}
                for i, line in enumerate(data.text.splitlines(), start=1) if line.strip()][:3]
    return {"evidence": evidence}


def validate_evidence(text: str, evidence: list[dict]) -> bool:
    """验证引文确实出现于指定原文行，不评价结论是否被引文支持。

    只用于本例这种字典列表结构；真实模型输出还应先做独立 schema 校验。
    这是普通工具函数，由 draft 显式调用，不是自动注册的图节点。
    """

    lines = text.splitlines()

    # all([]) 在 Python 中为 True，所以额外加 bool(evidence) 拒绝空证据列表。
    # and 从左到右短路求值：先检查行号类型与范围，再进行索引和字符串操作。
    return bool(evidence) and all(
        # bool 是 int 的子类；type(...) is int 特意不把 True 当成第 1 行。
        type(item.get("line")) is int

        # 用户可见行号从 1 开始，必须落在原文现有行数以内。
        and 1 <= item["line"] <= len(lines)

        # 引用必须是非空白字符串，不能是数字、None 或空串。
        and isinstance(item.get("quote"), str)
        and bool(item["quote"].strip())

        # Python 列表下标从 0 开始，因此访问 line - 1。
        # 子串存在只证明引用位置真实，不证明引用足以支持某个推论。
        and item["quote"] in lines[item["line"] - 1]
        for item in evidence
    )


def draft(state: State) -> dict:
    """先检查证据，再生成带 L<行号> 标记的抽取式草稿。"""

    # 验证失败就明确终止，而不是把伪造引用包装成看似正常的报告。
    if not validate_evidence(state["text"], state["evidence"]):
        raise ValueError("引用不对应原文，停止生成报告")

    # f-string 把行号和原文片段插入文本；join 用换行连接每个条目。
    # 结果仍然只是抽取式格式化，不应冒称由模型完成了语义摘要。
    report = "\n".join(f"- L{item['line']}: {item['quote']}" for item in state["evidence"])
    return {"draft": report}


def quality_gate(state: State) -> dict:
    """产生路由需要的布尔值，不直接调用任何后续节点。"""

    # force_review=True 时强制审核；否则证据不足 2 条也审核。
    # .get(..., False) 为缺省字段提供读取时的回退值，不会把它写回 State。
    # 这只是让三条实验路径都能触发的规则，不是语义质量评分或模型置信度。
    return {"needs_review": state.get("force_review", False) or len(state["evidence"]) < 2}


def review(state: State) -> dict:
    """展示草稿，等待真实的 bool 决定，并接受可选的人工修订文本。"""

    # 第一次运行在这里暂停；恢复时整个 review 从头重跑，再取得 resume 值。
    # 审批前不执行发布副作用，避免重放时重复触发动作。
    decision = interrupt({"draft": state["draft"], "question": "是否批准？可提供修订稿。"})

    # 字符串 "false" 不是合法布尔决定，不能因为它是非空字符串就放行。
    if not isinstance(decision, dict) or type(decision.get("approved")) is not bool:
        raise ValueError("approved 必须是布尔值")

    # 没有 corrected_report 时使用原草稿；提供它时必须是字符串。
    # 即使选择拒绝，这里仍会检查传入字段类型，不接受任意混乱结构。
    revised = decision.get("corrected_report", state["draft"])
    if not isinstance(revised, str):
        raise ValueError("corrected_report 必须是字符串")

    # 批准就保留修订稿；拒绝就明确清空 report。
    # 本节点不生成最终 status，让 finalize 统一处理三条路径的终态。
    return {"approved": decision["approved"], "report": revised if decision["approved"] else ""}


def finalize(state: State) -> dict:
    """区分自动和人工路径，写入最终状态，不覆盖已经批准的人工稿。"""

    if not state["needs_review"]:
        # 自动路径没有经过 review，所以这里负责把 draft 复制为最终 report。
        # “自动通过”在本例仅代表规则通过，不是已经证明报告内容完全正确。
        return {"report": state["draft"], "status": "auto_approved"}

    # 人工路径的 report 已由 review 写入，这里只更新 status。
    # 如果不小心再次用 draft 覆盖 report，人工修订就会丢失；测试会检查它。
    return {"status": "human_approved" if state["approved"] else "rejected"}


def build_graph(checkpointer=None):
    """分别声明完整状态、输入接口和输出接口，再构建分支流程。"""

    # State 包含中间字段；DocumentInput 约束图接收的业务字段；
    # DocumentOutput 筛选最终输出。后者不阻止内部字段进入检查点或日志。
    builder = StateGraph(State, input_schema=DocumentInput, output_schema=DocumentOutput)

    # 用 (节点名, 函数对象) 列表依次注册，减少重复的 add_node 语句。
    # 列表里的函数名后不加括号，因为这里只注册，不马上运行。
    for name, node in [("extract", extract), ("draft", draft), ("gate", quality_gate),
                       ("review", review), ("finalize", finalize)]:
        builder.add_node(name, node)

    # 前半段是确定的线性流程：校验与抽取 → 草稿 → 审核规则。
    builder.add_edge(START, "extract")
    builder.add_edge("extract", "draft")
    builder.add_edge("draft", "gate")

    # 路由只读取 gate 已写入的 needs_review，返回目标节点名称。
    # 这里二选一，不是两个节点同时执行的并行扇出。
    builder.add_conditional_edges(
        "gate", lambda state: "review" if state["needs_review"] else "finalize",
        ["review", "finalize"],
    )

    # 人工路径恢复并完成 review 后，也进入同一个 finalize。
    builder.add_edge("review", "finalize")
    builder.add_edge("finalize", END)

    # 调用者提供 saver 时使用它；否则每次 build_graph 创建新的内存 saver。
    # 内存 saver 支持本进程里的暂停恢复，不支持重启进程后的可靠恢复。
    return builder.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


if __name__ == "__main__":
    app = build_graph()

    # 这个小函数按任务名生成运行配置；三个不同名字表示三个独立任务。
    cfg = lambda tid: {"configurable": {"thread_id": tid}}

    # 两行有效证据且不强制审核，直接产生 auto_approved。
    print("自动路径：", app.invoke(
        {"text": "LangGraph 编排有状态流程。\nCheckpointer 保存运行状态。", "force_review": False}, cfg("auto"),
    ))

    # 新 thread 开始短文任务，review 处暂停；输出里可读取中断载荷。
    # __interrupt__ 是框架返回的特殊结果键，不是我们声明的业务字段。
    paused = app.invoke({"text": "需要审核的短文。", "force_review": True}, cfg("manual"))
    print("暂停：", paused["__interrupt__"][0].value)

    # 用同一 thread 恢复，批准且不提供修订稿，所以使用现有 draft。
    print("批准：", app.invoke(Command(resume={"approved": True}), cfg("manual")))

    # 再创建第三个独立任务，专门观察拒绝路径，不复用已经完成的 manual 任务。
    app.invoke({"text": "另一篇短文。", "force_review": True}, cfg("reject"))
    print("拒绝：", app.invoke(Command(resume={"approved": False}), cfg("reject")))

# 练习：批准时加入 corrected_report，确认最终显示你的修订稿。
# 给 validate_evidence 传入不存在的行号或伪造引文，确认返回 False。
# 最后比较 invoke 的输出与 app.get_state(config).values：
# 输出 schema 隐藏的 evidence 仍可能存在检查点中，因此它不是权限控制。
