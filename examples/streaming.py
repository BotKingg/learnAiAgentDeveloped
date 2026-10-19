"""
实验 8：把“节点进度事件”和“状态更新结果”分别显示出来。

运行：
    python -m examples.streaming

预期会看到两类数据：
    custom  {'stage': 'started'}
    updates {'work': {'value': 1}}

本例没有模型，因此不演示 token 输出。
它采用 stream(..., version="v2") 的统一 StreamPart 格式。
这里的 v2 是流式返回格式，与 LangGraph 包版本 1.2.16 不是一回事。
"""

from typing import TypedDict

# 在节点运行上下文中获取 writer，用它发送自定义进度。
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    """只有一个普通状态字段，默认更新规则是覆盖。"""

    value: int


def work(state: State) -> dict:
    """先发送自定义进度，再提交业务状态更新。

    两个动作的区别：
    writer(...) 发出的只是事件，不会自动写入 State；
    return {...} 才提交节点要写入的状态更新。
    """

    # writer 依赖图运行时的上下文，适合在节点被框架调度时获取。
    # 不应把这段代码复制到任意普通函数环境中，就假定始终存在上下文。
    writer = get_stream_writer()

    # 向监听 custom 模式的消费者发送数据。
    # 这个 stage 字段没有定义在 State 中，也不会因此自动成为状态字段。
    # 可以用同样方式发送“下载开始”“已处理 3/10”等自定义进度。
    writer({"stage": "started"})

    # 节点返回时提交 value 的新值，监听 updates 的消费者会收到节点更新。
    # 输入 0，本节点返回 1；若重复单独提交 0，结果仍由那次输入决定。
    return {"value": state["value"] + 1}


def build_graph():
    """构建只有一个工作节点的图，让观察重点集中在流式事件上。"""

    builder = StateGraph(State)
    builder.add_node("work", work)
    builder.add_edge(START, "work")
    builder.add_edge("work", END)
    return builder.compile()


if __name__ == "__main__":
    # stream 返回迭代器。for 循环持续消费事件，从而驱动并观察本次执行。
    # 仅创建迭代器、却不消费它，不应被当成“任务已经运行完毕”。
    for part in build_graph().stream(
        {"value": 0},

        # 用列表同时订阅两种模式。若没有订阅 custom，
        # 就不能指望在这里看到 writer 发出的进度消息。
        stream_mode=["updates", "custom"],

        # v2 使用统一结构：type 标识事件类别，ns 是命名空间，data 是内容。
        # 默认 v1 的返回形状还会受模式数量、子图选项影响，本例显式指定 v2。
        version="v2",
    ):
        # custom 的 data 是我们发送的 {"stage": "started"}；
        # updates 的 data 是 {"work": {"value": 1}}。
        # work 是节点名称，不是 State 中额外创建出来的业务字段。
        print(part["type"], part["data"])

# 练习：分别只订阅 custom、只订阅 updates，比较看到的事件。
# 然后将输入改为 10，预测进度事件和状态更新哪个会随之改变。
# 产品界面需要单独处理完成、错误和中断，不能看见进度就宣称任务成功。
