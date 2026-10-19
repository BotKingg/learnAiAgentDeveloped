"""
实验 5：把“等待人工决定”保存到 SQLite，再由另一个进程恢复。

在仓库根目录依次运行：
    python -m examples.approval start --thread lesson-001
    python -m examples.approval inspect --thread lesson-001
    python -m examples.approval approve --thread lesson-001

再用另一个会话观察拒绝：
    python -m examples.approval start --thread lesson-002
    python -m examples.approval reject --thread lesson-002

执行路径：
    START → review → 暂停
                     ↓ Command(resume=...)
                    review 重新执行 → publish 或 cancel → END

每条命令都是一个新 Python 进程，SQLite 文件才是它们共享的持久数据。
publish 只返回字符串形式的模拟回执，没有发布文件、发邮件或调用外部系统。
默认数据库位置是仓库根目录下的 .local/approval.sqlite。
重复练习时换一个新的 --thread；同一会话不能反复当成新请求覆盖。
"""

# argparse 解析 start、approve 等命令行输入，避免依靠手工修改代码切换操作。
import argparse
# json 用于把最终结果转换为方便阅读、也能被测试解析的文本。
import json
# Path 用于处理数据库文件路径、创建父目录。
from pathlib import Path
# Literal 为 Command 声明允许的目标节点名；TypedDict 描述业务状态字段。
from typing import Literal, TypedDict

# from_conn_string 返回连接生命周期管理器，with 退出时释放数据库连接。
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
# interrupt 产生暂停；Command 可以传入恢复值，也可以在节点里更新并路由。
from langgraph.types import Command, interrupt


class State(TypedDict):
    """一份审批请求的业务状态；TypedDict 不会自动创建所有字段。"""

    # 稳定的业务请求标识。演示里直接使用 thread 名称，真实系统可单独建 ID。
    request_id: str
    # 审批人应该看到的具体内容。生产应用还需绑定内容版本，避免批准错稿。
    draft: str
    # 在 review 恢复并校验决定后才写入，初始输入不需要提前伪造它。
    approved: bool
    # publish 或 cancel 产生的结果；暂停时不应该已经存在成功回执。
    receipt: str


def review(state: State) -> Command[Literal["publish", "cancel"]]:
    """暂停等待决定，恢复后校验类型，同时返回状态更新和下一节点。

    第一次进入：interrupt 抛出框架识别的暂停信号，下面的语句还不执行。
    恢复时：框架从这个节点开头重跑；再次遇到同一个 interrupt，
    它使用已提供的 resume 值作为返回值，然后继续往下执行。

    Literal 的类型提示描述可能到达的节点，不能代替下面对输入的实际校验。
    """

    # 这里构造的是交给外部界面或 CLI 读取的审批内容。
    # 只使用可序列化的数据，便于保存和跨进程传递。
    # interrupt 之前没有发邮件、扣款等副作用，重跑这段代码不会提前执行操作。
    decision = interrupt({
        "request_id": state["request_id"], "draft": state["draft"],
        "question": "是否批准这份草稿？",
    })

    # 第一次暂停时，执行不到这里；收到恢复值后才校验。
    # 字符串 "false" 在 Python 中是真值，不能简单用 bool(...) 把它当决定。
    # type(...) is bool 要求值确实是 True 或 False，而不是字符串或整数。
    if not isinstance(decision, dict) or type(decision.get("approved")) is not bool:
        raise ValueError("审批结果必须包含布尔类型的 approved")

    approved = decision["approved"]

    # update 提交要写入的状态字段；goto 选择下一节点。
    # 这不是 Command(resume=...)：resume 是外部调用时提供决定，
    # update/goto 是节点已经得到决定后，告诉框架怎样继续。
    return Command(update={"approved": approved}, goto="publish" if approved else "cancel")


def build_graph(checkpointer):
    """用外部提供的检查点器，搭建包含动态分支的审批图。"""

    builder = StateGraph(State)
    builder.add_node("review", review)

    # lambda s: ... 是匿名节点函数，s 是该节点看到的状态。
    # 两个节点都只返回 receipt 更新，不会替换整个状态。
    # publish 只是生成模拟回执；真实发布必须另做权限校验与业务幂等。
    builder.add_node("publish", lambda s: {"receipt": f"模拟发布成功：{s['request_id']}"})
    builder.add_node("cancel", lambda s: {"receipt": "已取消"})

    builder.add_edge(START, "review")

    # review 的下一步由它返回的 Command(goto=...) 决定。
    # 不再给 review 添加无条件出边；否则静态路径也可能被调度，
    # 造成“明明选择 cancel，另一条 publish 路径却仍然执行”的错误设计。
    builder.add_edge("publish", END)
    builder.add_edge("cancel", END)

    # 保存检查点，才能把暂停进度交给后面的另一次进程调用。
    # 内存 saver 只能在同一进程内恢复，下面 CLI 特意选 SQLite。
    return builder.compile(checkpointer=checkpointer)


def main():
    """解析命令，打开数据库，读取进度，然后开始、查看或恢复一个会话。"""

    # __doc__ 是文件顶部的模块文档字符串，--help 会展示它。
    # 因此命令行帮助中也能直接看到练习步骤和执行路径。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "approve", "reject", "inspect"])

    # 相对路径根据你执行命令时的工作目录解释，所以 README 要求从仓库根运行。
    parser.add_argument("--db", default=".local/approval.sqlite")
    parser.add_argument("--thread", default="approval-demo")
    parser.add_argument("--draft", default="文档助手生成的示例草稿")
    args = parser.parse_args()

    # SQLite 可以创建数据库文件，但父目录需要先存在。
    # parents=True 允许创建多级目录，exist_ok=True 允许目录已经存在。
    path = Path(args.db)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 同样的数据库 + 同样的 thread_id，才指向这里要恢复的同一个会话。
    # 只换数据库文件、却保留 thread_id，并不能找到另一个数据库里的检查点。
    config = {"configurable": {"thread_id": args.thread}}

    # with 块的有效期覆盖构建、读取和运行，不能先关掉连接再继续使用图。
    # 退出进程后，连接结束；文件里的已写入检查点仍可供下个进程读取。
    with SqliteSaver.from_conn_string(str(path)) as saver:
        app = build_graph(saver)

        # get_state 读取已有进度，不会因为读取而重新执行 review。
        # values 是业务状态；next 是待调度的节点；tasks 带有任务和中断信息。
        snapshot = app.get_state(config)

        if args.action == "inspect":
            # 查看操作只返回已有值和下一节点，便于观察暂停前后有什么变化。
            result = {"values": snapshot.values, "next": snapshot.next}

        elif args.action == "start":
            # 顺序运行 CLI 时，阻止把一个已有会话覆盖成新请求。
            # 这不是跨多进程并发的原子锁；真实服务还要设计竞争处理策略。
            if snapshot.values:
                parser.error("这个 thread 已存在，请换一个 --thread 或恢复已有审批")

            # 首次调用只提供初始输入，让 review 自己执行到 interrupt。
            # approved 与 receipt 会在后续节点产生，不在这里伪造默认结果。
            result = app.invoke({"request_id": args.thread, "draft": args.draft}, config)

        else:
            # 进入这个分支的 action 只可能是 approve 或 reject。
            # any(...) 检查是否至少有一个任务仍带待处理的中断。
            # 已完成或不存在的会话不能被当成待审批任务继续批准。
            if not any(task.interrupts for task in snapshot.tasks):
                parser.error("这个 thread 没有待处理的审批")

            # 表达式 args.action == "approve" 产生真正的 bool：
            # approve 对应 True，reject 对应 False。
            # 这次 invoke 传 Command 而非一份新业务输入，明确表示恢复旧任务。
            result = app.invoke(Command(resume={"approved": args.action == "approve"}), config)

        # 暂停返回值中有 Interrupt 对象，不直接假定它能被 json.dumps 编码。
        # 保留 id 和 value，转换成普通字典后，CLI 输出才能稳定成为 JSON。
        if "__interrupt__" in result:
            result["__interrupt__"] = [
                {"id": item.id, "value": item.value} for item in result["__interrupt__"]
            ]

        # ensure_ascii=False 让中文原样显示，方便人读，也不影响 JSON 结构。
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

# 练习：start 后关掉终端，再重新打开终端执行 inspect 和 approve。
# 如果仍使用同一个数据库文件和 thread_id，应能继续处理暂停的请求。
# 再次 approve 应被拒绝；另开 thread 执行 reject，应只产生“已取消”。
# 把本例改成真实发布前，需补审批人鉴权、版本绑定、幂等和并发控制。
