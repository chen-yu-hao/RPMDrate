dRPMD：母轨迹、子轨迹与多 GPU 并行
=====================================

本实现给 ``serial``、``bead``、``gpu`` 三种势能/传播后端统一增加**轨迹级并行**，
没有把它变成互斥的第四种模式。原单链物理实现原样保留在 ``rpmdrate/drpmd_core.py``；
``rpmdrate.drpmd`` 保持原来的 Python API 和命令行入口。不指定并行选项时，旧调用与输出结构不变。

母子轨迹的含义
----------------

设总碰撞轨迹数为 N、轨迹并行数为 n，实际启动 min(N,n) 条独立热化母轨迹。
每个进程先完成 ``thermalization_steps``，随后每推进 ``sampling_steps`` 步，
复制完整珠子坐标/内部动量，随机取向、采样冲量参数和入射通量加权碰撞能，
关闭恒温器与约束，传播一条碰撞子轨迹。子轨迹结束后丢弃其动力学状态，
继续推进原母轨迹；绝不会把产物状态拿回去热化。

各母轨迹分担 floor(N/n) 或 ceil(N/n) 条碰撞子轨迹，余数分配到前面的链，
总数严格等于 N。例如 N=10003、n=8，对应 1251,1251,1251,1250,1250,1250,1250,1250。
初始热化长度是**每条母轨迹**的长度，不会除以 n；N/n 指该链负责的碰撞子轨迹数。

这一组织方式参考 ``RPMD.computeRecrossingFactor`` 的“约束热化母轨迹→复制→无恒温子轨迹”，
但 dRPMD 的母轨迹位于远距离反应物区域，不在过渡态分界面；这里也不使用 recrossing 的正负动量配对估计式。
当前每次采样发射一条子轨迹，不提供 ``childrenPerSampling`` 批量分叉参数。
母轨迹之间同时运行，每条母轨迹内部的子轨迹依次运行；不是预先生成全部初态再启动碰撞。

``temperature_K`` 在一个任务内保持不变。采样间隔是**时间或步数**，不是温度间隔。
可以继续用 ``sampling_steps``，也可以改用 ``sampling_time_fs``：
该时间必须是 ``dt_fs`` 的正整数倍，同时提供两者时必须一致。
新生成的母轨迹首个样本位于初始热化之后再传播一个采样间隔的位置，与原实现一致。

CPU 使用
----------

每条轨迹单核、同时运行 32 条母链/碰撞子轨迹，总计 10000 条碰撞：

.. code-block:: bash

   python -m rpmdrate.drpmd examples/drpmd/hhoo_smoke.json \
     --mode serial --trajectory-workers 32 --trajectories 10000 \
     --output results-serial-ensemble

同时运行 4 条轨迹，每条使用 8 个 PES 工作进程和 8 线程 Cayley 内核：

.. code-block:: bash

   python -m rpmdrate.drpmd examples/drpmd/hhoo_smoke.json \
     --mode bead --trajectory-workers 4 --workers 8 --trajectories 10000 \
     --output results-bead-ensemble

``--workers`` 始终是**每条轨迹内部**的参数。上述珠子计算与动力学阶段的计算核心需求约为 4×8，
另有协调进程和通信开销；不是把 8 个核心共享给全部 4 条轨迹。
母轨迹进程不是 daemon，因此可以安全创建其独立的珠子 PES 进程池。
此处是单节点多进程，不是 MPI 跨节点实现。

GPU 使用
----------

两卡、每卡同时一条轨迹（两条独立母链）：

.. code-block:: bash

   python -m rpmdrate.drpmd examples/drpmd/hhoo_smoke.json \
     --mode gpu --gpu-devices 0 1 --trajectories-per-gpu 1 \
     --trajectories 10000 --output results-gpu-one

两卡、每卡同时四条轨迹（八条独立母链）：

.. code-block:: bash

   python -m rpmdrate.drpmd examples/drpmd/hhoo_smoke.json \
     --mode gpu --gpu-devices 0 1 --trajectories-per-gpu 4 \
     --trajectories 10000 --output results-gpu-many

也支持只指定 ``--trajectory-workers``，按所选 GPU 顺序循环分配。
同时指定每卡并行数和总并行数时，总并行数必须等于卡数×每卡并行数。
如果 N 小于计划的并行数，不启动空链，输出会记录实际启用的槽位。

使用 ``spawn``，CUDA 在各工作进程中初始化；每个进程独立加载模型、绑定设备，
拥有独立随机数、热化状态和输出目录。GPU 负责该轨迹所有珠子的批量 PES/梯度，
Cayley 动力学仍由该进程的一个 CPU 线程计算。

GPU 编号是 ``CUDA_VISIBLE_DEVICES`` 过滤后的**逻辑编号**。例如：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=2,5 python -m rpmdrate.drpmd input.json \
     --mode gpu --gpu-devices 0 1 --trajectories-per-gpu 2 --output results

这里的逻辑卡 0、1 分别对应可见列表中的物理卡 2、5。
不使用 DDP/NCCL，也不在进程之间传递 CUDA 张量。
每卡多轨迹实现的是**多个独立进程/模型上下文共用一张卡**，
不是把所有轨迹融合成一个跨轨迹张量 batch。
显存占用会随每卡进程数增加，GPU 是否真正重叠执行及加速效果需实机测量；
程序不启用 MPS，不承诺线性加速，不会在 CUDA 不可用时悄悄退回 CPU。

JSON 与 Python API
-------------------

在现有输入中增加以下对象即可保存执行设置（模式和珠子 workers 仍由命令行/API 指定）：

.. code-block:: json

   "parallel": {
     "gpu_devices": [0, 1],
     "trajectories_per_gpu": 4
   }

CPU 可改为 ``"parallel": {"trajectory_workers": 32}``。显式命令行/API 并行参数覆盖同名 JSON 设置。
总数仍由顶层 ``trajectories`` 指定，``--trajectories`` 仍可覆盖它。

.. code-block:: python

   import json
   from pathlib import Path
   from rpmdrate.drpmd import run

   def main():
       source = Path("examples/drpmd/hhoo_smoke.json").resolve()
       config = json.loads(source.read_text())
       config["trajectories"] = 10000
       return run(config, base=source.parent, mode="serial",
                  trajectory_workers=32, output="results-python")

   if __name__ == "__main__":
       main()

GPU 调用为 ``run(..., mode="gpu", gpu_devices=[0,1], trajectories_per_gpu=4)``。
多进程调用必须放在可导入脚本的 ``if __name__ == "__main__"`` 保护下，
不要从没有主入口保护的 Notebook 单元格/交互式 stdin 直接启动。

随机数、输出与失败
------------------

每条链的种子由总 seed 和稳定 chain_id 通过 NumPy SeedSequence 派生，
每条链内部再分离热化与碰撞随机流。固定输入、seed、并行链数及后端数值行为时，
结果不依赖工作进程完成顺序；改变链数会改变热化样本划分，因此不承诺逐轨迹相同。
各链的 seed 和全局轨迹范围写入清单，能用原单链接口独立复现其中一条链。

启用并行选项后的输出：

.. code-block:: text

   results/
     input.json                 # 物理输入、分配方案、种子、GPU 映射和势模型身份
     trajectories.csv           # 全局轨迹编号 + chain_id + local_trajectory + 文件路径
     trajectories.json
     summary.json               # 全部 N 个任务的合并统计
     progress.json
     logs/chain_0001.log
     worker_status/chain_0001.json
     chains/chain_0001/
       thermal_state.npz
       trajectory_000001_initial.npz
       trajectory_000001.csv
       trajectory_000001_final.npz
       trajectories.csv
       summary.json

各链目录内的轨迹编号是局部编号，全局编号和对应文件路径以顶层索引为准。
各进程不共同追加同一个轨迹文件，已有输出目录会被拒绝。
若某链失败，其他链仍可完成；没有产生结果的已分配轨迹保留为 failed/unknown，
不从总分母 N 中删除。``not_launched_trajectories`` 单独区分未发射任务。
后续热化失败时，已写出的子轨迹终态可以恢复到汇总中。

``--thermal-state`` 可接收先前并行输出的根目录；Python 还可提供每条链一个文件的列表。
每个文件都验证温度、珠子数、质量、反应物分组、距离和势能身份。
多链运行拒绝直接把同一个文件作为全部链的首样本；不同路径也不自动保证统计独立。
每个导入的首样本会直接使用，没有隐藏的额外热化。

独立随机流不等于样本无自相关，也不证明已热平衡。
统计仍需检查热化长度、采样间隔、珠子数、时间步、碰撞参数上限等收敛；
现有 Wilson 区间的 iid 假设不包含母链内相关性。
上述 HHOO 示例仍是 smoke input，增加轨迹数不会使其自动成为生产参数。

测试
------

.. code-block:: bash

   make drpmd-test python=python
   python -m unittest tests.drpmd_paralleltest -v

新测试包括总数/余数分配、GPU 映射、进程隔离、独立热化链、单链重放、母子状态隔离、
嵌套珠子进程池、故障与硬退出、终态恢复、热化态重用及命令行。
CUDA 单卡多轨迹、双卡单轨迹和双卡多轨迹测试需要实际设备；无设备时跳过，
不能将调度单元测试或 CPU 对照误称为 CUDA 实机验证。

参考实现与运行时说明：

* 本仓库 ``rpmdrate/main.py`` 的 ``computeRecrossingFactor``。
* https://docs.pytorch.org/docs/stable/notes/multiprocessing.html
* https://docs.pytorch.org/docs/stable/notes/cuda.html
* https://docs.pytorch.org/docs/stable/cuda_environment_variables.html
