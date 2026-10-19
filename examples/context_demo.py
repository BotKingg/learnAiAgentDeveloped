"""
实验 4：区分业务状态 State、运行上下文 context 和执行配置 config。

运行：
    python -m examples.context_demo

预期结果包含 greeting："u-001：你好，小明"。
本例通过 invoke(..., context=...) 注入用户标识，不调用模型。

这三类信息各有职责：
- State：节点读取和更新的业务字段，本例是 name、greeting；
- context：调用时注入的上下文或依赖，本例是 user_id；
- config：框架执行配置，如其他示例里的 thread_id、recursion_limit。
不要因为三者都能传数据，就把数据库连接或密钥塞进业务状态。
"""

# dataclass 自动生成初始化等常用方法，让 Context(user_id="...") 可以直接使用。
from dataclasses import dataclass
from typing import TypedDict
from langgraph.graph import END, START, StateGraph

# Runtime 是框架向节点提供的运行时对象，带有本次调用的 context。
from langgraph.runtime import Runtime


@dataclass(frozen=True)
class Context:
    """本次调用的上下文结构；frozen=True 限制普通属性重新赋值。"""

    # 这里只是演示值。真实服务应在认证后注入用户 ID，
    # 不应直接信任前端随便传来的 user_id。
    user_id: str


class State(TypedDict):
    """图实际处理的业务数据；与 Context 是两个不同的结构。"""

    # 姓名由调用者提交给 invoke 的第一个输入字典。
    name: str

    # 节点生成的结果字段，由部分状态更新写入。
    greeting: str


def greet(state: State, runtime: Runtime[Context]) -> dict:
    """同时读取业务状态和运行上下文，再生成一份状态更新。

    runtime 的参数名称和类型声明符合 LangGraph 的运行时注入约定。
    Runtime[Context] 说明这里期望得到哪种上下文，方便 IDE 与读者理解。
    本例无需自行创建 Runtime，也无需把它放进 state。
    """

    # state["name"] 取业务输入；runtime.context.user_id 取本次注入的用户标识。
    # 把上下文内容写入 greeting 后，这一部分会出现在图输出中。
    # 所以 context 不是“写进去就永远保密”的保险箱：输出和日志仍需审查。
    return {"greeting": f"{runtime.context.user_id}：你好，{state['name']}"}


def build_graph():
    """声明 context schema，再按普通方式注册节点和边。"""

    # context_schema 告诉图：本次运行上下文使用 Context 这个结构。
    # 它与 state schema 分开，不会自动成为一个名叫 context 的状态字段。
    builder = StateGraph(State, context_schema=Context)
    builder.add_node("greet", greet)
    builder.add_edge(START, "greet")
    builder.add_edge("greet", END)
    return builder.compile()


if __name__ == "__main__":
    # 第一个参数是业务输入；context=... 是本次调用提供的上下文。
    # 图进入 greet 时，框架把它包装在 runtime.context 中供节点读取。
    print(build_graph().invoke({"name": "小明"}, context=Context(user_id="u-001")))

# 练习：保留 name 不变，只改变 Context.user_id，预测 greeting 如何变化。
# 再思考：如果 greeting 会被写入检查点，是否应该把任何敏感上下文拼进去？
