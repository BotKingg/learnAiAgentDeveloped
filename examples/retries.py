"""运行：python -m examples.retries。用模拟服务验证临时故障重试。"""
from typing import TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy


class State(TypedDict):
    query: str
    answer: str


class FlakyService:
    """仅用于测试的故障注入器，不是业务状态。"""
    def __init__(self, failures: int, error_type=TimeoutError):
        self.calls = 0
        self.failures = failures
        self.error_type = error_type

    def __call__(self, state: State) -> dict:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error_type("模拟依赖故障")
        return {"answer": f"完成：{state['query']}"}


def build_graph(service):
    builder = StateGraph(State)
    builder.add_node("request", service, retry_policy=RetryPolicy(
        max_attempts=3, initial_interval=0.01, jitter=False, retry_on=TimeoutError,
    ))
    builder.add_edge(START, "request")
    builder.add_edge("request", END)
    return builder.compile()


if __name__ == "__main__":
    service = FlakyService(failures=2)
    print(build_graph(service).invoke({"query": "测试请求"}))
    print("总尝试次数：", service.calls)
