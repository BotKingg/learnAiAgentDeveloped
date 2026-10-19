"""
实验 2：保留同一会话的消息，并观察不同会话之间的隔离。

运行：
    python -m examples.message_memory

本文件使用固定回显器，不调用 LLM，不会理解“我叫小明”的含义。
第一轮加入一条用户消息和一条回显；第二轮再各加入一条，总共 4 条。
要学的是状态持久化与消息合并，而不是模型的语言理解能力。

执行路径：
    START → echo → END
跨轮次共享状态的条件：
    使用同一个 checkpointer 实例，以及相同的 thread_id。
"""

# AIMessage 是框架中的助手消息对象，包含 content、id 等消息字段。
from langchain_core.messages import AIMessage

# MessagesState 已为 messages 配置 add_messages：
# 新消息通常按新 ID 追加，同 ID 消息更新，显式 RemoveMessage 可以删除。
from langgraph.graph import END, START, MessagesState, StateGraph

# InMemorySaver 把检查点保存在当前进程内存中，进程退出就丢失。
# 它适合观察多轮行为；跨进程恢复要看 approval.py 的 SQLite 示例。
from langgraph.checkpoint.memory import InMemorySaver


def build_graph(checkpointer):
    """由外部传入检查点器，让调用者明确选择状态保存在哪里。

    在函数里定义 echo，是为了把这个小节点与图的构建放在一起阅读。
    build_graph 还没有运行图，嵌套函数也不会在定义时自动执行。
    """

    def echo(state: MessagesState) -> dict:
        """读取最新消息，返回一条新增助手消息。"""

        # messages 是消息列表；[-1] 取最后一条；.content 取它的内容。
        # 在本图的正常调用顺序中，最后一条就是本轮刚提交的用户消息。
        # AIMessage(...) 创建消息对象，而不是向真实模型请求答案。
        # 外层列表表示“这次要加入的消息列表”，不是全部历史。
        # 只返回新消息，能避免把全部旧历史反复提交给 reducer。
        return {"messages": [AIMessage(content=f"已收到：{state['messages'][-1].content}")]}

    # 使用 MessagesState 时，普通的消息列表管理由其 reducer 完成。
    builder = StateGraph(MessagesState)
    builder.add_node("echo", echo)

    # 本图内部没有循环。多轮对话来自下方对 invoke 的多次调用，
    # 并由 checkpointer 恢复同一 thread 的历史。
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)

    # 编译时接入 checkpointer，运行时再由 thread_id 定位某个会话。
    return builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    # 创建一次内存检查点器，再让同一个 app 处理两次调用。
    # 如果每一轮都换一个全新的 InMemorySaver，就无法找到上轮历史。
    app = build_graph(InMemorySaver())

    # configurable 是运行配置中的可配置项容器。
    # thread_id 标识会话，不是 Python 线程，也不是访问权限凭证。
    config = {"configurable": {"thread_id": "chat-a"}}

    # ("human", "文本") 是消息的简写形式，会被转换成用户消息对象。
    # 第一轮结束后：用户“我叫小明” + 助手“已收到：我叫小明”。
    app.invoke({"messages": [("human", "我叫小明")]}, config)

    # 第二轮只传新增消息；同一 thread 的旧消息由框架从检查点恢复。
    # 然后 echo 再增加一条助手回复，因此最终有 4 条消息。
    result = app.invoke({"messages": [("human", "第二轮")]}, config)

    # 列表推导式把每条消息转换成 (角色, 内容)，便于直接观察顺序。
    print([(m.type, m.content) for m in result["messages"]])
    print("消息数：", len(result["messages"]))

# 练习：把第二轮的 thread_id 换成 chat-b，先预测再观察消息数量。
# 再关闭整个进程重跑，确认内存保存不代表磁盘持久化。
