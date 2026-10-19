"""
实验 1：看懂“输入状态 → 节点返回更新 → 最终状态”。

在仓库根目录运行：
    python -m examples.minimal

不需要 API Key，也不会调用网络或模型。
执行路径：
    START → greet → END
预期终端输出：
    你好，小明！

先预测：输入里只有 name，为什么输出里还会多出 greeting？
再观察：greet 只返回 greeting，为什么原来的 name 没有消失？
这两个问题就是本实验要掌握的“部分更新”语义。
"""

# TypedDict 用来向读者、IDE 和类型检查器说明字典应该有哪些字段。
# 它不是自动填默认值的表单，也不会在这里完成完整的运行时输入校验。
from typing import TypedDict

# StateGraph 是图的构建器；START、END 是框架的入口和结束标记。
# 它们不是需要你实现的业务函数。
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    """描述这个图可能读取或产生的业务字段，不负责创建字段的初始值。"""

    # 调用者提供的姓名。代码读取 state["name"]，所以运行时必须先有它。
    name: str

    # greet 节点产生的问候语；首次输入可以不含它，因为节点会写入它。
    greeting: str


def greet(state: State) -> dict:
    """读取姓名，检查空白输入，返回只包含 greeting 的状态更新。

    输入示例：{"name": " 小明 "}
    返回更新：{"greeting": "你好，小明！"}

    注意：这里返回的字典是更新量，不是图的完整最终状态。
    name 的更新规则是默认覆盖，但本节点根本没有提交 name 的新值。
    因此最终状态里的 name 仍是原始的 " 小明 "，两侧空格仍然存在。
    """

    # ["name"] 按键取值；缺少这个键时会抛出 KeyError，而不会自动得到 ""。
    # strip() 返回清理两侧空白后的新字符串，原字符串不会被原地修改。
    # 这里的局部变量 name 与状态字典中的 name 字段要分开理解。
    name = state["name"].strip()

    # 空字符串在 Python 的条件判断中为 False；not name 因而为 True。
    # 提前抛出具体错误，能避免生成一条看似正常、实际缺少姓名的问候。
    if not name:
        raise ValueError("name 不能为空")

    # 只提交本节点真正产生的 greeting。
    # LangGraph 会按字段规则合并更新，保留没有在这个更新里出现的字段。
    return {"greeting": f"你好，{name}！"}


def build_graph():
    """创建并编译图，但此时不会执行 greet，也不会生成任何问候语。"""

    # 第一步：声明状态 schema，得到可以添加节点和边的构建器。
    builder = StateGraph(State)

    # 第二步：把节点名称 "greet" 与函数对象 greet 绑定。
    # 这里写 greet，不写 greet(...)：函数要等图真正运行时才被调用。
    builder.add_node("greet", greet)

    # 第三步：明确执行顺序。START 使 greet 成为入口节点。
    builder.add_edge(START, "greet")

    # greet 完成后走到 END，本次图执行结束，不再调度其他业务节点。
    builder.add_edge("greet", END)

    # 第四步：编译成可执行对象。compile() 本身不等于运行。
    # 调用者拿到这个对象后，可以使用 invoke、stream、ainvoke 等接口。
    return builder.compile()


# 只有“作为模块入口运行”时才执行下面的演示。
# tests 导入本文件的 build_graph 时，不会顺便打印问候语。
if __name__ == "__main__":
    # 从内往外理解这一行：
    # 1. build_graph() 得到可执行图；
    # 2. invoke({...}) 提交初始输入并等待本次运行结束；
    # 3. ["greeting"] 从最终状态中取出问候语；
    # 4. print(...) 把它显示到终端。
    # 输入没有 greeting 是有意的：它会由 greet 节点生成。
    print(build_graph().invoke({"name": " 小明 "})["greeting"])

# 练习：先预测 invoke 的完整返回字典，再自己打印它验证。
# 然后加入 normalize 节点，让 name 的去空白结果真正写回状态。
