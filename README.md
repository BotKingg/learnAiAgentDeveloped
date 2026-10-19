# learnAiAgentDeveloped

LangGraph 中文学习指南与实验代码。建议先运行无需 API Key 的离线示例。

从 [LangGraph_学习指南_修订版.md](LangGraph_学习指南_修订版.md) 开始。建议先完成离线实验，再自行接入真实模型。

- [原文审阅与修改说明.md](原文审阅与修改说明.md)：哪些信息已旧、哪些代码错误、哪些建议需要边界，以及官方来源。
- [验证记录.md](验证记录.md)：本次实际执行情况与未验证部分。
- `examples/`：10 个教学模块，其中 9 个可离线运行，`real_agent` 需自行配置模型并产生供应商费用。
- `tests/test_learning.py`：35 项离线测试；其中包含对预期错误行为的验证，不代表运行时无条件容错。
- `requirements.in`：直接依赖；`requirements.lock.txt`：完整锁定版本与 SHA256 哈希。

## 详细注释版代码入口

每份示例都包含：运行命令、学习目标、执行路径、预期结果、字段解释、函数讲解、关键语句注释和动手练习。建议先读文件顶部，再看数据结构和函数，最后读主入口，按实际调用顺序回头跟踪节点。

| 顺序 | 带详细中文注释的源码 | 重点阅读 |
| --- | --- | --- |
| 1 | [minimal.py](examples/minimal.py) | TypedDict、局部变量、部分更新、构建和执行的区别 |
| 2 | [message_memory.py](examples/message_memory.py) | 消息 reducer、新增消息、thread_id、内存持久化边界 |
| 3 | [tool_loop.py](examples/tool_loop.py) | 工具 schema、调用 ID、ToolMessage、条件路由与循环终止 |
| 4 | [context_demo.py](examples/context_demo.py) | State、Runtime context、执行 config 的职责 |
| 5 | [approval.py](examples/approval.py) | 中断重放、Command、SQLite、命令行参数和跨进程恢复 |
| 6 | [parallel.py](examples/parallel.py) | Send、worker 输入、reducer、空输入、排序和并发度 |
| 7 | [retries.py](examples/retries.py) | 可调用对象、故障注入、异常类型和重试总次数 |
| 8 | [streaming.py](examples/streaming.py) | writer、自定义进度、状态更新和 v2 StreamPart |
| 9 | [document_workflow.py](examples/document_workflow.py) | Pydantic、短路校验、证据行号、人工修订和输出 schema |
| 10 | [real_agent.py](examples/real_agent.py) | 模型依赖注入、create_agent、环境配置和真实调用边界 |

[测试文件也有逐项中文讲解](tests/test_learning.py)：每项说明准备条件、执行动作、预期结果，以及错误行为为什么必须被识别。你可以把注释里的预期先遮住，写出自己的预测，再运行对应测试验证。

已经克隆过仓库时，在本地仓库的 main 分支拉取本版：

```bash
git pull --ff-only origin main
```

## 快速运行

先克隆仓库，然后在仓库根目录执行。建议使用 Python 3.12：

```bash
git clone https://github.com/BotKingg/learnAiAgentDeveloped.git
cd learnAiAgentDeveloped
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock.txt
python -m unittest discover -s tests -v
python -m examples.minimal
python -m examples.message_memory
python -m examples.tool_loop
python -m examples.context_demo
python -m examples.parallel
python -m examples.retries
python -m examples.streaming
python -m examples.document_workflow
```

SQLite 审批实验，每条命令都是一个独立进程：

```bash
python -m examples.approval start --thread first-demo
python -m examples.approval inspect --thread first-demo
python -m examples.approval approve --thread first-demo
python -m examples.approval start --thread reject-demo
python -m examples.approval reject --thread reject-demo
```

再次从头实验时使用新的 thread 名称，避免覆盖现有状态。SQLite 文件保存在 `.local/approval.sqlite`；publish 节点只返回模拟回执，不向外部发布。

离线示例里的回显器、固定模型脚本、文本切片和引用抽取都已明确标注，不能拿它们的结果证明真实 AI 系统的语义能力。

`real_agent` 的模型适配和 Agent 工具循环由脚本模型测试；未调用真实模型服务。只有自行安全配置 OPENAI_API_KEY 和支持工具调用的 LLM_MODEL 后，才运行 `python -m examples.real_agent`。不要向聊天或代码仓库提交密钥。

这些文件是学习资料与可运行实验；真实模型服务和生产部署的验证范围见验证记录。


## 建议怎样学习

每个示例按“预测结果 → 运行 → 修改 → 制造错误 → 关闭参考自己重写”的顺序练习。

1. 先做 `examples/minimal.py`：增加 normalize 节点，再解释状态如何更新。
2. 做 `examples/message_memory.py`：观察两轮消息和不同 thread 的隔离。
3. 做 `examples/tool_loop.py`：找到工具请求、调用 ID、工具结果和最终消息。
4. 做 `examples/approval.py`：在独立进程之间恢复审批，分别执行批准与拒绝。
5. 再学 context、并行、重试、流式和完整文档审核流程。

每天完成一个能解释的小实验即可。真实模型接入放在理解工具循环之后；离线模拟输出不代表模型的真实理解能力。
