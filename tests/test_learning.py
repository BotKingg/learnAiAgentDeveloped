"""
学习实验的行为验证：把每个重要解释变成会通过或失败的检查。

在仓库根目录运行：
    python -m unittest discover -s tests -v

unittest 会发现下面以 test_ 开头的 35 个方法。
每个方法独立准备输入和依赖，不应依靠另一个测试先执行。

阅读一个测试时按三步看：
1. 准备：创建图、内存 saver、脚本模型或临时目录；
2. 执行：调用 invoke、恢复 Command，或者启动独立 Python 进程；
3. 断言：比较结果，或确认约定的异常确实出现。

常用断言：
- assertEqual(a, b)：a 和 b 必须相等；
- assertIn(x, container)：必须包含 x；
- assertNotIn(x, container)：必须不包含 x；
- assertRaises(Error)：这段代码本来就应该抛出 Error，否则测试失败。
“验证预期报错”的测试通过，不代表真实运行时会把错误悄悄修好。

这些测试关注框架执行、参数边界和状态行为；没有请求真实付费模型。
脚本模型能验证协议，不能证明真实模型的判断质量或供应商可用性。
"""

# 标准库用于事件循环、JSON、路径、子进程和独立临时目录。
# unittest 负责发现和执行断言；warnings 只在明确的实验性接口测试中处理。
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import TypedDict
import unittest
import warnings

# 这两项只改变当前测试进程及其子进程的 tracing 开关，不读取或输出密钥。
# 放在框架导入之前，让离线验证不向外部 tracing 服务上报实验数据。
# 验证教学机制时不向外部 tracing 服务发送数据。
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# Fake 模型在内存中返回预设输出，不请求真实供应商。
# 错误类、消息类与图构建器分别用于验证协议、错误边界和状态行为。

from langchain_core.language_models.fake_chat_models import FakeListChatModel, FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError, InvalidUpdateError
from langgraph.graph import END, START, MessagesState, StateGraph, add_messages
from langgraph.types import Command, interrupt

# 导入待验证的示例函数；示例中的 __main__ 块不会因为 import 就执行。
from examples import approval, context_demo, document_workflow as docs
from examples import message_memory, minimal, parallel, real_agent, retries, streaming, tool_loop


def cfg(thread):
    """为一个会话生成运行配置，避免在测试里重复写同一层字典。

    thread 是测试选择的会话标识，不是认证凭证，也不是系统线程对象。
    返回值必须把 thread_id 放在 configurable 内部。
    """

    # 其他测试用 cfg("a") 与 cfg("b") 创建不同会话；不在这里创建 saver。
    return {"configurable": {"thread_id": thread}}


class Basics(unittest.TestCase):
    """基础机制：部分状态更新、消息 reducer、会话隔离、输入契约与上下文。"""

    def test_greeting_and_partial_update(self):
        """验证问候语被生成，同时未被更新的 name 保留原值。

        输入故意包含两侧空格：节点局部 strip 后生成干净问候，
        但没有提交 name 更新，所以最终 name 仍带原始空格。
        """

        result = minimal.build_graph().invoke({"name": " 小明 "})
        # 问候语使用去空白后的局部变量，但这不代表 name 已写回。
        self.assertEqual(result["greeting"], "你好，小明！")
        # 没有节点提交 name 更新，最终应保留带空格的原输入。
        self.assertEqual(result["name"], " 小明 ")

    def test_empty_name_rejected(self):
        """只包含空白的姓名应在 greet 中被明确拒绝。

        with assertRaises 的代码块必须抛出 ValueError 才算通过。
        如果节点默默输出“你好，！”，这个测试反而应该失败。
        """

        with self.assertRaises(ValueError):
            minimal.build_graph().invoke({"name": " "})

    def test_typeddict_does_not_supply_missing_value(self):
        """证明 TypedDict 不会自动填出缺失的 name。

        只传 greeting，让 greet 读取 state["name"] 时产生 KeyError。
        这与空字符串输入是两个不同场景，所以分开测试。
        """

        with self.assertRaises(KeyError):
            minimal.build_graph().invoke({"greeting": ""})

    def test_message_id_update(self):
        """相同消息 ID 应更新原记录，而不是追加成两条。

        两次消息都用 u1：第一条内容“你好”应被“修订”替代。
        既检查长度，也检查内容，避免只更新其中一个方面却误判通过。
        """

        old = [HumanMessage(content="你好", id="u1")]
        result = add_messages(old, [HumanMessage(content="修订", id="u1")])
        # 相同 ID 应替换；如果追加成两条，长度断言会最先暴露错误。
        self.assertEqual(len(result), 1)
        # 长度正确还不够，原消息内容也必须确实更新。
        self.assertEqual(result[0].content, "修订")

    def test_remove_message(self):
        """用明确的 RemoveMessage 指令删除指定 ID 的消息。

        输入只有 u1，删除后应为空列表。
        返回截短历史不是这个指令的等价写法。
        """

        result = add_messages([HumanMessage(content="你好", id="u1")], [RemoveMessage(id="u1")])
        # 指定 ID 的唯一消息删除后，列表应真正变空。
        self.assertEqual(result, [])

    def test_threads_accumulate_and_isolate(self):
        """同一图中，a 的两轮累积，b 的首轮独立。

        每轮一条用户消息加一条回显，因此 a 最终 4 条，b 最终 2 条。
        如果所有用户错误共享同一个 thread，b 的消息数量会暴露问题。
        """

        app = message_memory.build_graph(InMemorySaver())
        # 第一轮保存到 a 的检查点；后续是否复用取决于 saver 和 thread_id。
        app.invoke({"messages": [("human", "一")]}, cfg("a"))
        same = app.invoke({"messages": [("human", "二")]}, cfg("a"))
        other = app.invoke({"messages": [("human", "三")]}, cfg("b"))
        # a 两轮，每轮用户 + 回显，期望总计 4 条。
        self.assertEqual(len(same["messages"]), 4)
        # b 刚开始，不能混入 a 的两轮消息。
        self.assertEqual(len(other["messages"]), 2)

    def test_new_memory_saver_has_no_previous_checkpoints(self):
        """新建内存 saver，不能恢复旧 saver 中的检查点。

        这里没有重启进程，只主动换一份空内存存储来验证存储归属。
        真正跨进程恢复另由后面的 SQLite 子进程测试验证。
        """

        app = message_memory.build_graph(InMemorySaver())
        # 第一轮保存到 a 的检查点；后续是否复用取决于 saver 和 thread_id。
        app.invoke({"messages": [("human", "一")]}, cfg("a"))
        new_app = message_memory.build_graph(InMemorySaver())
        # 会话名字相同不代表跨不同存储实例就能共享检查点。
        self.assertEqual(new_app.get_state(cfg("a")).values, {})

    def test_direct_chat_model_is_not_state_adapter(self):
        """复现把 Chat Model 直接当状态适配器的常见错误。

        FakeListChatModel 不访问网络，但仍遵循模型输入的类型约定。
        StateGraph 传进去的是 dict，模型期望消息列表等类型，因此应报错。
        """

        builder = StateGraph(MessagesState)
        # 替身仍遵循真实 Chat Model 的输入类型，足以复现状态 dict 不适配。
        builder.add_node("model", FakeListChatModel(responses=["不会到达"] ))
        builder.add_edge(START, "model")
        builder.add_edge("model", END)
        with self.assertRaisesRegex(ValueError, "Invalid input type"):
            builder.compile().invoke({"messages": [("human", "你好")]})

    def test_concurrent_write_without_reducer_raises(self):
        """同一步两个节点写一个普通字段，应识别更新冲突。

        a、b 都从 START 激活，没有前后依赖；value 没有 reducer。
        期望 InvalidUpdateError，不能把结果想成“后写覆盖前写”。
        """

        class State(TypedDict):
            """这个测试单独使用的最小状态 schema，不与其他测试共享字段设计。"""

            value: str
        builder = StateGraph(State)
        builder.add_node("a", lambda s: {"value": "a"})
        builder.add_node("b", lambda s: {"value": "b"})
        # 两个节点都直接从 START 出发，属于同一步的并行写入场景。
        for name in ("a", "b"):
            builder.add_edge(START, name)
            builder.add_edge(name, END)
        with self.assertRaises(InvalidUpdateError):
            builder.compile().invoke({"value": ""})

    def test_runtime_context(self):
        """运行时 context 应被框架注入节点，并参与生成问候。

        输入 state 只有 name，用户 ID 从 Context(user_id="u-42") 提供。
        只检查结果包含这个 ID，不把它误当输入 state 自带的字段。
        """

        result = context_demo.build_graph().invoke(
            {"name": "小明"}, context=context_demo.Context(user_id="u-42"),
        )
        # 确认节点实际读到了本次注入的 context，而不只是普通输入 name。
        self.assertIn("u-42", result["greeting"])

    def test_ainvoke(self):
        """异步调用应返回与同步示例相符的业务结果。

        普通 unittest 方法里没有正在运行的事件循环，
        因此用 asyncio.run 驱动 ainvoke 返回的协程直到完成。
        """

        # ainvoke 是协程；asyncio.run 在同步测试中创建循环并等待它完成。
        result = asyncio.run(minimal.build_graph().ainvoke({"name": "小明"}))
        # 问候语使用去空白后的局部变量，但这不代表 name 已写回。
        self.assertEqual(result["greeting"], "你好，小明！")


class ToolsAndFailures(unittest.TestCase):
    """工具协议与故障边界：校验、关联 ID、图预算、选择性重试及 Agent 工厂。"""

    def test_tool_loop_and_correlation_id(self):
        """验证工具确实执行，并与原工具请求正确关联。

        既找 ToolMessage 的内容 42，也检查 tool_call_id。
        仅最终回复出现“42”不够，因为固定文本也能伪造这样的答案。
        """

        result = tool_loop.build_graph().invoke({"messages": [("human", "演示")]})
        # 从完整轨迹筛出工具结果，避免把最终助手文本错当执行证据。
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        # 本脚本请求一次工具，因此必须恰好产生一条工具结果。
        self.assertEqual(len(tool_messages), 1)
        # 工具返回的整数经过消息包装后，这里表现为字符串 "42"。
        self.assertEqual(tool_messages[0].content, "42")
        # 结果必须回应原请求的关联 ID，否则协议可能错配。
        self.assertEqual(tool_messages[0].tool_call_id, "call-demo-1")
        # 最后再检查最终回复确实使用到了工具结果。
        self.assertIn("42", result["messages"][-1].content)

    def test_tool_rejects_out_of_range(self):
        """工具执行层必须拒绝超出范围的参数。

        直接调用工具而不经过模型，证明边界由执行代码保证，
        不会因为模型忽略提示词就失效。
        """

        with self.assertRaises(ValueError):
            tool_loop.multiply.invoke({"a": 1_000_001, "b": 7})

    def test_recursion_limit_stops_loop(self):
        """图预算不足时，工具循环应明确抛出步数超限错误。

        设置为 1，无法完成 model→tools→model 的整段执行。
        这个测试验证框架执行步数限制，不验证模型费用或精确 token 限额。
        """

        with self.assertRaises(GraphRecursionError):
            tool_loop.build_graph().invoke(
                {"messages": [("human", "演示")]}, config={"recursion_limit": 1},
            )

    def test_retry_succeeds_on_third_total_attempt(self):
        """前两次临时失败，第三次应成功，总尝试次数恰好为 3。

        同时检查计数和 answer：只看成功结果可能漏掉错误的额外重试。
        FlakyService 是故障注入替身，不访问真实网络。
        """

        service = retries.FlakyService(failures=2)
        result = retries.build_graph(service).invoke({"query": "x"})
        # max_attempts=3 计入首次调用，达到上限不应偷偷再执行一次。
        self.assertEqual(service.calls, 3)
        self.assertEqual(result["answer"], "完成：x")

    def test_retry_exhausts_after_three_total_attempts(self):
        """持续超时不能无限重试，达到总上限后应把错误抛出来。

        failures=9 大于允许的 3 次，故整个任务应该失败。
        退出异常上下文后仍可读取服务计数，确认没有额外第 4 次执行。
        """

        service = retries.FlakyService(failures=9)
        with self.assertRaises(TimeoutError):
            retries.build_graph(service).invoke({"query": "x"})
        # max_attempts=3 计入首次调用，达到上限不应偷偷再执行一次。
        self.assertEqual(service.calls, 3)

    def test_non_retryable_error_is_not_retried(self):
        """ValueError 不属于本例选择的 TimeoutError 重试范围。

        即使服务还配置了后续失败，第一次 ValueError 就应该向上传播。
        如果错误地把全部异常都重试，calls 会大于 1。
        """

        service = retries.FlakyService(failures=9, error_type=ValueError)
        with self.assertRaises(ValueError):
            retries.build_graph(service).invoke({"query": "x"})
        # 错误类型不匹配重试策略，第一次失败就应停止。
        self.assertEqual(service.calls, 1)

    def test_create_agent_factory_with_scripted_tool_model(self):
        """用离线模型验证 create_agent 也能完成相同工具协议。

        预设两条模型输出：先请求乘法，再给最终文本。
        它验证 Agent 工厂、工具调用和结果关联，不评测真实模型如何选工具。
        """

        class ScriptedModel(FakeMessagesListChatModel):
            """只用于测试的模型替身，按 responses 顺序返回预先构造的消息。"""

            def bind_tools(self, tools, **kwargs):
                """让测试替身接受 Agent 工厂的绑定请求并返回自身。

                真实模型要把工具 schema 发给供应商；这里输出已经预写好了，
                所以不需要真的实现供应商绑定过程，也不应因此请求网络。
                """

                return self
        model = ScriptedModel(responses=[
            AIMessage(content="", tool_calls=[{
                "id": "factory-call", "name": "multiply", "args": {"a": 6, "b": 7},
                "type": "tool_call",
            }]),
            AIMessage(content="答案是42"),
        ])
        # 传入脚本模型而非 ChatOpenAI，所以不会发起付费模型请求。
        agent = real_agent.build_agent(model, InMemorySaver())
        result = agent.invoke({"messages": [("human", "演示")]}, cfg("factory"))
        # next 找不到工具结果会失败，不能靠模型自己回答“42”冒充成功。
        tool_result = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        self.assertEqual(tool_result.content, "42")
        # 与这次测试构造的工具请求 ID 精确关联。
        self.assertEqual(tool_result.tool_call_id, "factory-call")
        self.assertEqual(result["messages"][-1].content, "答案是42")


class Approval(unittest.TestCase):
    """审批与恢复：暂停前不执行、批准与拒绝分支、类型检查和跨进程存储。"""

    def test_pause_has_no_publish_receipt(self):
        """第一次运行应暂停，且尚未产生发布成功回执。

        必须同时看到 __interrupt__、看不到 receipt；
        否则即使能弹审批窗口，也可能已经提前执行了动作。
        """

        app = approval.build_graph(InMemorySaver())
        result = app.invoke({"request_id": "r1", "draft": "草稿"}, cfg("a"))
        # 特殊返回键表明确实发生了动态中断，不是普通的业务字段。
        self.assertIn("__interrupt__", result)
        # 在取得审批之前，不应该已经有任何发布或取消结果。
        self.assertNotIn("receipt", result)

    def test_approve(self):
        """恢复值为真正的 True 时，选择 publish 分支。

        同一个 app、同一个 thread 先 start 再恢复。
        检查模拟回执包含请求 ID，确认不是误走取消路径。
        """

        app = approval.build_graph(InMemorySaver())
        app.invoke({"request_id": "r1", "draft": "草稿"}, cfg("a"))
        result = app.invoke(Command(resume={"approved": True}), cfg("a"))
        # 这个回执只由 publish 分支产生，且带上本次请求标识。
        self.assertEqual(result["receipt"], "模拟发布成功：r1")

    def test_reject(self):
        """恢复值为 False 时，必须选择 cancel。

        用与批准相同的准备步骤，只改变决定，
        从而单独验证拒绝分支，不依赖另一个测试的状态。
        """

        app = approval.build_graph(InMemorySaver())
        app.invoke({"request_id": "r1", "draft": "草稿"}, cfg("a"))
        result = app.invoke(Command(resume={"approved": False}), cfg("a"))
        # 拒绝必须到 cancel，不能与 publish 同时触发。
        self.assertEqual(result["receipt"], "已取消")

    def test_approval_requires_boolean(self):
        """字符串 "false" 不能被当作合格的审批结果。

        非空字符串在 Python 中为真；若代码只写 if decision，
        就可能错误放行。本例要求 bool 类型，因此应抛 ValueError。
        """

        app = approval.build_graph(InMemorySaver())
        app.invoke({"request_id": "r1", "draft": "草稿"}, cfg("a"))
        with self.assertRaises(ValueError):
            app.invoke(Command(resume={"approved": "false"}), cfg("a"))

    def test_interrupt_restarts_the_node(self):
        """用测试探针证明恢复时节点从头执行。

        visits 故意在 interrupt 前追加记录：暂停前一次，恢复时再一次。
        这是观察重放的测试技巧，不是鼓励生产节点在中断前做非幂等副作用。
        """

        class State(TypedDict):
            """这个测试单独使用的最小状态 schema，不与其他测试共享字段设计。"""

            value: str
        # 闭包共享一个列表，用它观察节点入口执行了几次；不是业务持久状态。
        visits = []
        def node(state):
            """每次进入节点记录一次，然后等待恢复值，最后提交 value。

            visits 是测试用的外部探针，不是可靠业务状态或生产审计日志。
            它故意暴露重跑行为，便于理解副作用为什么需要幂等。
            """

            # 故意放在中断前，用来证明恢复会重放之前的节点代码。
            visits.append("entered")
            answer = interrupt("继续吗")
            return {"value": answer}
        builder = StateGraph(State)
        builder.add_node("node", node)
        builder.add_edge(START, "node")
        builder.add_edge("node", END)
        app = builder.compile(checkpointer=InMemorySaver())
        app.invoke({"value": ""}, cfg("replay"))
        app.invoke(Command(resume="继续"), cfg("replay"))
        # 两次记录对应首次执行与恢复重跑，不是两个不同业务节点。
        self.assertEqual(visits, ["entered", "entered"])

    def test_sqlite_resume_in_separate_python_process(self):
        """用真正独立的 Python 进程验证磁盘持久化与重复恢复检查。

        先启动 start 子进程，退出后再启动 approve 子进程；
        两者靠同一个 SQLite 文件与 thread_id 共享进度，而不是共享内存。
        最后第三个进程重复 approve，应返回非零退出码。
        """

        # 从 tests/test_learning.py 回到仓库根，保证子进程的模块导入路径正确。
        root = Path(__file__).resolve().parents[1]
        # 临时目录隔离数据库；块结束自动删除，不污染学习者已有审批记录。
        with tempfile.TemporaryDirectory() as directory:
            # 使用当前测试的同一解释器，但每次 subprocess.run 都是独立进程。
            base = [sys.executable, "-m", "examples.approval"]
            args = ["--db", str(Path(directory) / "checkpoints.sqlite"), "--thread", "cross-process"]
            # check=True 会把非零退出码视为失败；不能只看到有输出就认定启动成功。
            started = subprocess.run(base + ["start"] + args, cwd=root, capture_output=True, text=True, check=True)
            self.assertIn("__interrupt__", json.loads(started.stdout))
            # 前一个进程已经退出，这次只能依靠磁盘中的 SQLite 检查点恢复。
            resumed = subprocess.run(base + ["approve"] + args, cwd=root, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(resumed.stdout)["receipt"], "模拟发布成功：cross-process")
            # 这里故意不使用 check=True，因为业务应拒绝重复批准并返回非零码。
            duplicate = subprocess.run(base + ["approve"] + args, cwd=root, capture_output=True, text=True)
            # 验证顺序提交的重复审批被拒绝；没有由此证明分布式并发互斥。
            self.assertNotEqual(duplicate.returncode, 0)


class ParallelAndStreaming(unittest.TestCase):
    """并行与事件流：完整汇总、空输入、事件分类及实验性接口验证。"""

    def test_parallel_collects_all_results_in_source_order(self):
        """三个 worker 的结果必须全部保留，最终报告按原章节顺序显示。

        集合比较验证没有漏掉任何来源 index；
        报告文本比较验证展示顺序，不能拿完成先后当章节先后。
        """

        result = parallel.build_graph().invoke({"chapters": ["甲", "乙", "丙"], "results": []})
        # 使用集合只检查来源是否齐全，不把内部合并顺序当成承诺。
        self.assertEqual({x["index"] for x in result["results"]}, {0, 1, 2})
        # 最终展示必须显式排序，保持原章节顺序。
        self.assertEqual(result["report"], "1. 甲\n2. 乙\n3. 丙")

    def test_parallel_empty_input(self):
        """空章节列表也应能正常结束并产生空报告。

        这一输入没有 worker 可执行，需要 fan_out 明确转到 collect。
        只测三个章节的正常路径发现不了这里的断路问题。
        """

        result = parallel.build_graph().invoke({"chapters": [], "results": []})
        # 再确认没有残留报告供后续流程误当成批准稿。
        self.assertEqual(result["report"], "")

    def test_stream_v2_custom_and_updates(self):
        """消费者应能区分自定义进度和节点状态更新。

        先消费完 stream，再按 type 分别取出 data 检查。
        不把不同类别混成同一种数据，也不额外假定跨类别的全局顺序。
        """

        parts = list(streaming.build_graph().stream(
            {"value": 0}, stream_mode=["updates", "custom"], version="v2",
        ))
        # 同一次流订阅收到多类事件，必须先根据 type 取对应 data。
        custom = [p["data"] for p in parts if p["type"] == "custom"]
        updates = [p["data"] for p in parts if p["type"] == "updates"]
        # 自定义进度来自 writer，不是节点的状态返回。
        self.assertEqual(custom, [{"stage": "started"}])
        # 更新外层的 work 是节点名，内层 value 才是业务字段。
        self.assertEqual(updates, [{"work": {"value": 1}}])

    def test_experimental_v3_state_projection(self):
        """验证当前锁定版本的实验性 v3 能产生最终状态。

        只在这一项测试内捕获实验性警告，避免输出噪声；
        正文仍标注该接口不稳定，不能用警告捕获来宣称已变成稳定 API。
        """

        with warnings.catch_warnings():
            # 仅在此测试上下文忽略实验性警告；不会修改全局稳定性判断。
            warnings.simplefilter("ignore")
            run = minimal.build_graph().stream_events({"name": "小明"}, version="v3")
            # 消费迭代器，把运行推进到结束，再读取最终 output。
            list(run.values)
            self.assertEqual(run.output["greeting"], "你好，小明！")


class DocumentWorkflow(unittest.TestCase):
    """综合流程：自动、人工修订、拒绝、输入边界、引文验证和内部状态可见性。"""

    def test_auto_approval(self):
        """两行有效内容且不强制审核，应沿自动路径完成。

        既检查状态 auto_approved，也检查第二行引文在最终报告中，
        防止只写成功标识、却没生成相应报告。
        """

        result = docs.build_graph().invoke({"text": "第一行\n第二行", "force_review": False}, cfg("auto"))
        self.assertEqual(result["status"], "auto_approved")
        # 引用行号与原文对应；这不是对语义摘要质量的评分。
        self.assertIn("L2: 第二行", result["report"])

    def test_manual_correction(self):
        """人工批准时提供修订稿，最终必须保留修订稿。

        先确认确实暂停，再用 Command 恢复并提交 corrected_report。
        如果 finalize 又用旧 draft 覆盖 report，这项断言会失败。
        """

        app = docs.build_graph()
        first = app.invoke({"text": "短文", "force_review": True}, cfg("edit"))
        self.assertIn("__interrupt__", first)
        result = app.invoke(Command(resume={"approved": True, "corrected_report": "人工修订稿"}), cfg("edit"))
        self.assertEqual(result["status"], "human_approved")
        # 若 finalize 覆盖了 review 的报告，这里会立刻发现修订稿丢失。
        self.assertEqual(result["report"], "人工修订稿")

    def test_manual_rejection(self):
        """人工拒绝后应有明确拒绝状态，并清空最终报告。

        不能只有 status 写 rejected，report 却还留着可被误用的正文。
        这项测试不执行任何真实发布操作。
        """

        app = docs.build_graph()
        app.invoke({"text": "短文", "force_review": True}, cfg("reject"))
        result = app.invoke(Command(resume={"approved": False}), cfg("reject"))
        # 先检查业务终态确实是拒绝。
        self.assertEqual(result["status"], "rejected")
        # 再确认没有残留报告供后续流程误当成批准稿。
        self.assertEqual(result["report"], "")

    def test_empty_document_rejected(self):
        """纯空白文档必须被拒绝，不能进入假成功流程。

        空白字符串长度大于零，能帮助检查 strip 之后的额外校验是否存在。
        Pydantic ValidationError 也属于 ValueError 系列，但此输入走空白检查。
        """

        with self.assertRaises(ValueError):
            docs.build_graph().invoke({"text": "  ", "force_review": False}, cfg("empty"))

    def test_string_boolean_rejected(self):
        """force_review="false" 应被 StrictBool 拒绝。

        字符串并不等于布尔值 False；不能靠 Python 的真假值规则猜用户意图。
        这里测试输入边界，尚未进入人工审批节点。
        """

        with self.assertRaises(ValueError):
            docs.build_graph().invoke({"text": "文档", "force_review": "false"}, cfg("bad-type"))

    def test_fabricated_evidence_rejected(self):
        """分别检查“引文伪造”和“行号越界”两类问题。

        第一份记录的行号有效但文本不存在；第二份引用文本存在但行号错误。
        引用存在性校验仍不能证明语义上的结论正确。
        """

        # 行号有效但引文不存在，必须返回 False。
        self.assertFalse(docs.validate_evidence("真实内容", [{"line": 1, "quote": "伪造内容"}]))
        # 引文真实但行号越界，也必须返回 False，不能盲目索引崩溃。
        self.assertFalse(docs.validate_evidence("真实内容", [{"line": 9, "quote": "真实内容"}]))

    def test_output_schema_filters_output_but_not_checkpoint(self):
        """证明输出 schema 筛选不等于内部数据保密。

        最终输出里不含 evidence，但 get_state 仍可读取它。
        因此必须单独实现 thread 访问授权，不能依赖字段被输出隐藏。
        """

        app = docs.build_graph()
        result = app.invoke({"text": "第一行\n第二行", "force_review": False}, cfg("private"))
        # output_schema 从最终返回接口排除了 evidence。
        self.assertNotIn("evidence", result)
        # 内部检查点仍有 evidence，证明接口筛选不是授权或保密机制。
        self.assertIn("evidence", app.get_state(cfg("private")).values)


if __name__ == "__main__":
    # 脚本作为入口时交给 unittest；日常学习优先使用顶部的 discover 命令。
    unittest.main()
