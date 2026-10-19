"""
LangGraph 学习实验包。

这个文件让 examples 作为一个明确的 Python 包被导入：
    from examples.minimal import build_graph

运行示例时，在仓库根目录使用模块方式：
    python -m examples.minimal

-m 会按包结构定位模块，方便其他示例使用 from examples... 导入共享工具。
如果直接在 examples 目录里执行某个文件，导入搜索路径可能与这里不同。
为了让运行方式一致，README 中统一使用 python -m examples.<模块名>。

本文件只提供说明，不会在 import examples 时启动图、访问网络或创建数据库。
真实模型的可选入口是 examples.real_agent，其余实验可离线运行。
"""
