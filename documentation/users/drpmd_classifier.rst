C 内部产物检查
==============

默认 ConnectivityClassifier 现在可以在 C 时间步循环内部检查，检查点不再强制返回
Python。质心计算、成键连通分量、片段质心/速度、无序分子式匹配、连续确认计数，
以及确认后的立即终止都在 C 中执行。HHOO 势能、Cayley 映射、判据和采样间隔不变。

构建与使用
----------

更新代码后必须重新构建旧动态库::

    python -m rpmdrate.drpmd_integrator --build --compiler gcc
    python -m rpmdrate.drpmd_native_pes --build-hhoo
    python -m rpmdrate.drpmd examples/drpmd/hhoo_native.json \
      --mode serial --trajectory-workers 4 --scheduler dynamic \
      --dynamics-backend native --classifier-backend native \
      --trajectories 100 --output native-check-results

这里原生示例只是流程验证，不是已收敛的反应速率输入。
``--classifier-backend`` / JSON 顶层 ``classifier_backend`` 支持：

* ``auto``（默认）：原生 CPU 动力学 + 默认分类器使用 C；其他情况使用 Python。
* ``native``：明确要求 C 内部检查；非原生动力学或自定义 Python 判据会报错。
* ``python``：保留旧检查回调，用于逐步对照和自定义实验。

Python 接口::

    summary = run(config, mode="serial", trajectory_workers=4,
                  dynamics_backend="native", classifier_backend="native",
                  output="native-check-results")

多进程入口仍需要 ``if __name__ == '__main__':``。
底层 ``Dynamics(..., classifier_backend="native")`` 使用相同设置。

检查与终止语义
--------------

成键仍是 ``|qbar_i-qbar_j| < cutoff``；出口仍要求每一对片段的距离
``>= exit_distance`` 且相对位置与相对速度的点积 ``> 0``。
只按质心和片段元素组成匹配，不引入逐珠子投票、电子态判断或原子身份追踪。

每 ``check_interval`` 步检查一次；不是每个 C 块检查一次。连续 ``persistence``
个检查时刻的有效通道标签相同才退出；未知标签清零，改变标签从 1 开始。
第 0 步不检查；最终步不是检查步时，不额外检查。确认发生在块内部时，立即在
该步停止，不能传播到块末尾后才回报。每条碰撞有独立的分类工作区与计数，
不同输出块之间保留计数，不同碰撞之间重新初始化。

这并不移除全部 Python 返回点。``save_every`` 的输出回调、``native_block_steps``
的中断响应块上限、轨迹结束/失败仍会返回 Python。例如 1000 步、块上限 256、
每 10 步检查，且轨迹未提前结束：无输出时积分块调用从 100 降为 4；每 100 步
输出时降为 10。检查仍然执行 100 次。

自定义 callable 和 ConnectivityClassifier 子类不能安全地自动翻译成 C，
在 auto 模式下仍在检查点调用 Python；native 模式明确拒绝。默认分类器配置
只在一条轨迹开始时编译/复制一次，中途修改参数或 monkey-patch 方法不受支持；
需要自定义行为时应使用子类/回调或 python 后端。

热化采样结束时的 ``validate_reactants`` 是每个采样任务一次的 Python 输入完整性
检查，不属于生产轨迹的逐次出口判断，本次没有改动。GPU/Torch 的积分路径仍是
Python，此版本不提供 GPU 上的 C 内部产物检查。

实现与可观测性
--------------

``_drpmd_classifier.c/.h`` 持有只读参数表和实例私有工作区，检查中没有内存分配。
``_drpmd_engine.c`` 通过原生函数调用检查并维护 C 连续计数，无 Python 函数指针。
``drpmd_classifier.py`` 只在边界将元素/分子式转为整数表及管理资源生命周期。

单链 summary 的 ``classifier_backend`` 表示实际后端，``native_product_checks``
统计 C 检查数，``native_block_calls`` 统计所有 C 积分块（含热化）。动态池在
``worker_reports`` 保存相同信息；未执行碰撞的 worker 的后端为 ``none``。
未判定、求力失败及缺失结局继续保留在请求的 N 分母中。

FP64、无 fast-math。C 与 NumPy 的浮点求和顺序不承诺在阈值附近跨平台逐位相同；
阈值、出口距离和检查频率仍需物理收敛验证。非有限分类中间量明确报错。

验证与性能复现
--------------

    python -m unittest tests.drpmd_classifiertest -v
    OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE \
    python benchmarks/classifier/run.py --beads 32 128 --threads 1 2 \
      --check-intervals 1 10 --steps 1000 --rounds 3 --output classifier-benchmark.json

基准的两边均为同一 C 动力学及同一原生 HHOO Fortran PES，区别仅是检查后端。
包含分类器准备和首次求力；模型初始化和热化初态准备不计时。
同时比较无 observer 和每 100 步格式化 CSV 的情况；CSV 使用内存流，不能代表
实际磁盘 I/O。逐轮保存时间、实际积分块调用数、终止状态/步数和最终 q/p 误差。
实际收益依赖 PES 成本和检查频率，不规定性能通过阈值，也不声称线性加速。
