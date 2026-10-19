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

# 验证教学机制时不向外部 tracing 服务发送数据。
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_core.language_models.fake_chat_models import FakeListChatModel, FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError, InvalidUpdateError
from langgraph.graph import END, START, MessagesState, StateGraph, add_messages
from langgraph.types import Command, interrupt

from examples import approval, context_demo, document_workflow as docs
from examples import message_memory, minimal, parallel, real_agent, retries, streaming, tool_loop


def cfg(thread):
    return {"configurable": {"thread_id": thread}}


class Basics(unittest.TestCase):
    def test_greeting_and_partial_update(self):
        result = minimal.build_graph().invoke({"name": " 小明 "})
        self.assertEqual(result["greeting"], "你好，小明！")
        self.assertEqual(result["name"], " 小明 ")

    def test_empty_name_rejected(self):
        with self.assertRaises(ValueError):
            minimal.build_graph().invoke({"name": " "})

    def test_typeddict_does_not_supply_missing_value(self):
        with self.assertRaises(KeyError):
            minimal.build_graph().invoke({"greeting": ""})

    def test_message_id_update(self):
        old = [HumanMessage(content="你好", id="u1")]
        result = add_messages(old, [HumanMessage(content="修订", id="u1")])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, "修订")

    def test_remove_message(self):
        result = add_messages([HumanMessage(content="你好", id="u1")], [RemoveMessage(id="u1")])
        self.assertEqual(result, [])

    def test_threads_accumulate_and_isolate(self):
        app = message_memory.build_graph(InMemorySaver())
        app.invoke({"messages": [("human", "一")]}, cfg("a"))
        same = app.invoke({"messages": [("human", "二")]}, cfg("a"))
        other = app.invoke({"messages": [("human", "三")]}, cfg("b"))
        self.assertEqual(len(same["messages"]), 4)
        self.assertEqual(len(other["messages"]), 2)

    def test_new_memory_saver_has_no_previous_checkpoints(self):
        app = message_memory.build_graph(InMemorySaver())
        app.invoke({"messages": [("human", "一")]}, cfg("a"))
        new_app = message_memory.build_graph(InMemorySaver())
        self.assertEqual(new_app.get_state(cfg("a")).values, {})

    def test_direct_chat_model_is_not_state_adapter(self):
        builder = StateGraph(MessagesState)
        builder.add_node("model", FakeListChatModel(responses=["不会到达"] ))
        builder.add_edge(START, "model")
        builder.add_edge("model", END)
        with self.assertRaisesRegex(ValueError, "Invalid input type"):
            builder.compile().invoke({"messages": [("human", "你好")]})

    def test_concurrent_write_without_reducer_raises(self):
        class State(TypedDict):
            value: str
        builder = StateGraph(State)
        builder.add_node("a", lambda s: {"value": "a"})
        builder.add_node("b", lambda s: {"value": "b"})
        for name in ("a", "b"):
            builder.add_edge(START, name)
            builder.add_edge(name, END)
        with self.assertRaises(InvalidUpdateError):
            builder.compile().invoke({"value": ""})

    def test_runtime_context(self):
        result = context_demo.build_graph().invoke(
            {"name": "小明"}, context=context_demo.Context(user_id="u-42"),
        )
        self.assertIn("u-42", result["greeting"])

    def test_ainvoke(self):
        result = asyncio.run(minimal.build_graph().ainvoke({"name": "小明"}))
        self.assertEqual(result["greeting"], "你好，小明！")


class ToolsAndFailures(unittest.TestCase):
    def test_tool_loop_and_correlation_id(self):
        result = tool_loop.build_graph().invoke({"messages": [("human", "演示")]})
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0].content, "42")
        self.assertEqual(tool_messages[0].tool_call_id, "call-demo-1")
        self.assertIn("42", result["messages"][-1].content)

    def test_tool_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            tool_loop.multiply.invoke({"a": 1_000_001, "b": 7})

    def test_recursion_limit_stops_loop(self):
        with self.assertRaises(GraphRecursionError):
            tool_loop.build_graph().invoke(
                {"messages": [("human", "演示")]}, config={"recursion_limit": 1},
            )

    def test_retry_succeeds_on_third_total_attempt(self):
        service = retries.FlakyService(failures=2)
        result = retries.build_graph(service).invoke({"query": "x"})
        self.assertEqual(service.calls, 3)
        self.assertEqual(result["answer"], "完成：x")

    def test_retry_exhausts_after_three_total_attempts(self):
        service = retries.FlakyService(failures=9)
        with self.assertRaises(TimeoutError):
            retries.build_graph(service).invoke({"query": "x"})
        self.assertEqual(service.calls, 3)

    def test_non_retryable_error_is_not_retried(self):
        service = retries.FlakyService(failures=9, error_type=ValueError)
        with self.assertRaises(ValueError):
            retries.build_graph(service).invoke({"query": "x"})
        self.assertEqual(service.calls, 1)

    def test_create_agent_factory_with_scripted_tool_model(self):
        class ScriptedModel(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self
        model = ScriptedModel(responses=[
            AIMessage(content="", tool_calls=[{
                "id": "factory-call", "name": "multiply", "args": {"a": 6, "b": 7},
                "type": "tool_call",
            }]),
            AIMessage(content="答案是42"),
        ])
        agent = real_agent.build_agent(model, InMemorySaver())
        result = agent.invoke({"messages": [("human", "演示")]}, cfg("factory"))
        tool_result = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        self.assertEqual(tool_result.content, "42")
        self.assertEqual(tool_result.tool_call_id, "factory-call")
        self.assertEqual(result["messages"][-1].content, "答案是42")


class Approval(unittest.TestCase):
    def test_pause_has_no_publish_receipt(self):
        app = approval.build_graph(InMemorySaver())
        result = app.invoke({"request_id": "r1", "draft": "草稿"}, cfg("a"))
        self.assertIn("__interrupt__", result)
        self.assertNotIn("receipt", result)

    def test_approve(self):
        app = approval.build_graph(InMemorySaver())
        app.invoke({"request_id": "r1", "draft": "草稿"}, cfg("a"))
        result = app.invoke(Command(resume={"approved": True}), cfg("a"))
        self.assertEqual(result["receipt"], "模拟发布成功：r1")

    def test_reject(self):
        app = approval.build_graph(InMemorySaver())
        app.invoke({"request_id": "r1", "draft": "草稿"}, cfg("a"))
        result = app.invoke(Command(resume={"approved": False}), cfg("a"))
        self.assertEqual(result["receipt"], "已取消")

    def test_approval_requires_boolean(self):
        app = approval.build_graph(InMemorySaver())
        app.invoke({"request_id": "r1", "draft": "草稿"}, cfg("a"))
        with self.assertRaises(ValueError):
            app.invoke(Command(resume={"approved": "false"}), cfg("a"))

    def test_interrupt_restarts_the_node(self):
        class State(TypedDict):
            value: str
        visits = []
        def node(state):
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
        self.assertEqual(visits, ["entered", "entered"])

    def test_sqlite_resume_in_separate_python_process(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            base = [sys.executable, "-m", "examples.approval"]
            args = ["--db", str(Path(directory) / "checkpoints.sqlite"), "--thread", "cross-process"]
            started = subprocess.run(base + ["start"] + args, cwd=root, capture_output=True, text=True, check=True)
            self.assertIn("__interrupt__", json.loads(started.stdout))
            resumed = subprocess.run(base + ["approve"] + args, cwd=root, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(resumed.stdout)["receipt"], "模拟发布成功：cross-process")
            duplicate = subprocess.run(base + ["approve"] + args, cwd=root, capture_output=True, text=True)
            self.assertNotEqual(duplicate.returncode, 0)


class ParallelAndStreaming(unittest.TestCase):
    def test_parallel_collects_all_results_in_source_order(self):
        result = parallel.build_graph().invoke({"chapters": ["甲", "乙", "丙"], "results": []})
        self.assertEqual({x["index"] for x in result["results"]}, {0, 1, 2})
        self.assertEqual(result["report"], "1. 甲\n2. 乙\n3. 丙")

    def test_parallel_empty_input(self):
        result = parallel.build_graph().invoke({"chapters": [], "results": []})
        self.assertEqual(result["report"], "")

    def test_stream_v2_custom_and_updates(self):
        parts = list(streaming.build_graph().stream(
            {"value": 0}, stream_mode=["updates", "custom"], version="v2",
        ))
        custom = [p["data"] for p in parts if p["type"] == "custom"]
        updates = [p["data"] for p in parts if p["type"] == "updates"]
        self.assertEqual(custom, [{"stage": "started"}])
        self.assertEqual(updates, [{"work": {"value": 1}}])

    def test_experimental_v3_state_projection(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run = minimal.build_graph().stream_events({"name": "小明"}, version="v3")
            list(run.values)
            self.assertEqual(run.output["greeting"], "你好，小明！")


class DocumentWorkflow(unittest.TestCase):
    def test_auto_approval(self):
        result = docs.build_graph().invoke({"text": "第一行\n第二行", "force_review": False}, cfg("auto"))
        self.assertEqual(result["status"], "auto_approved")
        self.assertIn("L2: 第二行", result["report"])

    def test_manual_correction(self):
        app = docs.build_graph()
        first = app.invoke({"text": "短文", "force_review": True}, cfg("edit"))
        self.assertIn("__interrupt__", first)
        result = app.invoke(Command(resume={"approved": True, "corrected_report": "人工修订稿"}), cfg("edit"))
        self.assertEqual(result["status"], "human_approved")
        self.assertEqual(result["report"], "人工修订稿")

    def test_manual_rejection(self):
        app = docs.build_graph()
        app.invoke({"text": "短文", "force_review": True}, cfg("reject"))
        result = app.invoke(Command(resume={"approved": False}), cfg("reject"))
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["report"], "")

    def test_empty_document_rejected(self):
        with self.assertRaises(ValueError):
            docs.build_graph().invoke({"text": "  ", "force_review": False}, cfg("empty"))

    def test_string_boolean_rejected(self):
        with self.assertRaises(ValueError):
            docs.build_graph().invoke({"text": "文档", "force_review": "false"}, cfg("bad-type"))

    def test_fabricated_evidence_rejected(self):
        self.assertFalse(docs.validate_evidence("真实内容", [{"line": 1, "quote": "伪造内容"}]))
        self.assertFalse(docs.validate_evidence("真实内容", [{"line": 9, "quote": "真实内容"}]))

    def test_output_schema_filters_output_but_not_checkpoint(self):
        app = docs.build_graph()
        result = app.invoke({"text": "第一行\n第二行", "force_review": False}, cfg("private"))
        self.assertNotIn("evidence", result)
        self.assertIn("evidence", app.get_state(cfg("private")).values)


if __name__ == "__main__":
    unittest.main()
