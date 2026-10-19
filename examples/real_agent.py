"""可选联网示例。运行前设置 OPENAI_API_KEY、LLM_MODEL；会产生模型费用。"""
import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from examples.tool_loop import multiply


def build_agent(model, checkpointer):
    return create_agent(
        model=model, tools=[multiply],
        system_prompt="你是中文计算助手。计算乘法时使用 multiply 工具。不要编造工具结果。",
        checkpointer=checkpointer,
    )


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("LLM_MODEL"):
        raise SystemExit("请先安全配置 OPENAI_API_KEY 和支持工具调用的 LLM_MODEL。")
    model = ChatOpenAI(model=os.environ["LLM_MODEL"], timeout=30, max_retries=1)
    agent = build_agent(model, InMemorySaver())
    result = agent.invoke(
        {"messages": [("human", "请使用工具计算 6 × 7，并用中文回答。") ]},
        config={"configurable": {"thread_id": "real-agent-demo"}, "recursion_limit": 10},
    )
    for message in result["messages"]:
        print(message.type, repr(message.content), getattr(message, "tool_calls", ""))
