"""
实验 7：精确观察“重试哪些错误”和“总共尝试了几次”。

运行：
    python -m examples.retries

预期：
    模拟服务前两次抛 TimeoutError，第三次成功；
    终端显示总尝试次数为 3。

RetryPolicy(max_attempts=3) 的 3 包括第一次执行，
所以不是“先执行一次，再额外重试 3 次”。
这里没有真实网络，错误由 FlakyService 主动制造，便于稳定复现。
"""

from typing import TypedDict
from langgraph.graph import END, START, StateGraph

# 当前使用的导入位置是 langgraph.types；
# 不照搬旧教程里的 from langgraph.pregel import RetryPolicy。
from langgraph.types import RetryPolicy


class State(TypedDict):
    """query 是请求输入，answer 是成功完成后由节点写入的结果。"""

    query: str
    answer: str


class FlakyService:
    """仅用于测试的故障注入器，不是业务状态。

    这个对象可以像函数一样被调用，因为它实现了 __call__。
    它在内存中保存 calls，只用于观察测试里的尝试次数。
    这个计数器不会随图检查点自动保存，不能用作生产的可靠计费或幂等依据。
    """

    def __init__(self, failures: int, error_type=TimeoutError):
        """配置前多少次调用失败，以及失败时抛出的异常类型。

        failures=2：前两次失败，第三次开始成功。
        error_type 是异常类，不是一个已经创建好的异常对象。
        """

        # 新服务实例从零开始计数；每个测试创建自己的实例，互不干扰。
        self.calls = 0

        # 在本实验中固定不变的故障注入配置。
        self.failures = failures
        self.error_type = error_type

    def __call__(self, state: State) -> dict:
        """节点执行一次，就累加一次调用次数；必要时主动失败。"""

        # 先计数再判断，因此第一次调用时 calls 为 1。
        # 重试再次进入同一个服务对象时，calls 会继续增加。
        self.calls += 1

        # 前 failures 次始终抛错。
        # self.error_type("...") 创建配置的异常实例，raise 把它交给框架。
        # 不在这里捕获并伪装成正常答案，否则框架可能认为节点已经成功。
        if self.calls <= self.failures:
            raise self.error_type("模拟依赖故障")

        # 只有真正成功时才返回 answer 更新。
        return {"answer": f"完成：{state['query']}"}


def build_graph(service):
    """把可调用的 service 注册为节点，并只为该节点配置重试策略。"""

    builder = StateGraph(State)

    # service 是函数或带 __call__ 的对象，都可以作为节点执行体。
    # 参数名使用 retry_policy；旧 retry 参数在当前系列里已弃用。
    builder.add_node("request", service, retry_policy=RetryPolicy(
        # 包括首次在内总共最多 3 次；全部失败后，异常继续向调用者抛出。
        max_attempts=3,

        # 教学演示等待时间很短，避免测试无谓等待。
        # 真实系统要根据供应商限流、超时与总体任务预算选择退避时间。
        initial_interval=0.01,

        # jitter 表示是否对退避时间添加随机扰动。
        # 演示关闭它便于观察；大量真实请求重试时通常需要避免同时重试。
        jitter=False,

        # 本例只重试内置 TimeoutError。ValueError 不匹配，因此不会重试。
        # 真实 SDK 可能使用自己定义的超时异常，需要按实际类型配置。
        retry_on=TimeoutError,
    ))

    builder.add_edge(START, "request")
    builder.add_edge("request", END)
    return builder.compile()


if __name__ == "__main__":
    # 单个 service 对象会被节点策略多次调用；第一次、第二次失败，第三次成功。
    service = FlakyService(failures=2)

    # invoke 会等待该节点执行或重试结束；成功时打印包含 answer 的状态。
    print(build_graph(service).invoke({"query": "测试请求"}))

    # 这是测试用的外部观测计数，帮助证明并非只执行了一次。
    print("总尝试次数：", service.calls)

# 练习 1：failures=9 时，总调用次数应为 3，并最终抛出 TimeoutError。
# 练习 2：error_type=ValueError 时，只调用 1 次就应失败。
# 真实模型 SDK 也可能自带重试；叠加节点重试前，应计算总尝试预算。
# 会发送通知或扣款的操作，还必须处理重复执行的业务幂等问题。
