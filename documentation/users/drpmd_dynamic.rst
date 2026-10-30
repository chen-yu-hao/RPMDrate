动态任务池：保留热化链，不保留 RNG 链
==================================

多轨迹计算现在默认使用 ``scheduler=dynamic``。
``trajectory_workers`` 同时确定持久化工作进程数和独立物理热化链数。
链保存完整 q/p、温度、质量、势能身份和 Andersen 步时钟；不保存、传输或恢复
随机数发生器状态。每个采样任务按输入 seed、链号、样本号生成新的局部随机流，
不沿用旧静态实现的 RNG 序列，也不承诺跨后端逐位一致。

原生 C/Fortran 传播与 PES 不变。Python 只在任务边界调度和传递文件路径，
不在积分步之间跨进程传输珠子数组。每个工作进程只初始化一次 PES/原生/GPU 上下文。

使用
----

每轨迹单线程，四个动态工作进程::

    python -m rpmdrate.drpmd examples/drpmd/hhoo_native.json \
      --mode serial --trajectory-workers 4 --scheduler dynamic \
      --dynamics-backend native --trajectories 1000 --output dynamic-results

珠子并行、共享内存、原生 PES 都可组合；例如两个轨迹工作进程、各两线程::

    python -m rpmdrate.drpmd examples/drpmd/hhoo_native.json \
      --mode bead --workers 2 --trajectory-workers 2 --scheduler dynamic \
      --trajectories 1000 --output dynamic-hybrid

GPU 设备固定绑定到工作进程，不绑定到热化链。每卡单/多轨迹示例::

    python -m rpmdrate.drpmd examples/drpmd/hhoo_smoke.json \
      --mode gpu --gpu-devices 0 1 --trajectories-per-gpu 1 \
      --scheduler dynamic --trajectories 1000 --output gpu-one-per-card

    python -m rpmdrate.drpmd examples/drpmd/hhoo_smoke.json \
      --mode gpu --gpu-devices 0 1 --trajectories-per-gpu 2 \
      --scheduler dynamic --trajectories 1000 --output gpu-two-per-card

GPU 示例需要 CUDA 设备及 CUDA 版 PyTorch；CPU CI 不证明实际多卡速度。
不得用含 ``cpu_backend=native``/``dynamics_backend=native`` 的 CPU 输入直接运行 GPU。
以上示例只演示配置，不是收敛化学速率输入。

JSON::

    "parallel": {
      "trajectory_workers": 4,
      "scheduler": "dynamic",
      "task_prefetch": 8
    }

Python::

    if __name__ == "__main__":
        summary = run(config, trajectory_workers=4, scheduler="dynamic",
                      task_prefetch=8, output="dynamic-results")

无并行选项时仍保留原单链输出/API；``--scheduler static`` 可显式回到旧固定分块，
用于兼容/对照。``task_prefetch`` 仅适用于 dynamic，默认是工作进程数的两倍（不超过 N）。

调度语义
--------

总轨迹数仍是 N。各热化链保留 floor(N/n) 或 ceil(N/n) 个样本的固定配额；
固定的是 **样本来源**，不是各核心要跑多少条碰撞。这样不会因为某一链碰撞较快，
就改变各链对统计集合的贡献。

任一空闲工作进程都可领取以下任一任务：

* 推进某条未被占用的热化母链，获得一个初态并赋予碰撞动能。
* 从全局就绪队列领取任意链的碰撞初态，完成一条无恒温器的传播。

同一母链最多一个采样任务在执行。采样完成后原子性更新 ``thermal_latest.npz``，
下一次任务从该链最新 q/p 与步时钟继续，不重复初始热化，也不使用碰撞末态作为母链。
因此不仅碰撞可换核心，热化采样也不必等待原进程跑完它正在执行的长碰撞。

``task_prefetch`` 限制“就绪碰撞 + 在途采样”的数量；不会一次将 N 份坐标压入内存。
初态只在任务边界存取磁盘，适合长轨迹。极短轨迹可能被文件与调度开销支配；
这是单节点共享文件系统实现，不是 MPI。

工作池耗尽任务前，空闲进程继续取任务。最后剩余少于工作进程数的长轨迹仍会形成
不可避免的尾部空闲；本实现不会把一条正在积分的轨迹拆给多个轨迹工作进程。

检查点、失败与输出
------------------

``chains/chain_XXXX/thermal_state.npz`` 保留本次第一个采样态；
``thermal_latest.npz`` 是最新母链态。dynamic 使用上次输出目录作为
``--thermal-state`` 时优先读 latest，首次导入样本直接使用，不隐式再热化。
这表示新采样任务复用物理母链，不是自动恢复所有未完成的碰撞任务。

碰撞文件仍位于来源链目录，名称中的编号为该链内编号；顶层表提供唯一全局
``trajectory``、``chain_id``、``local_trajectory``、``sampling_worker``、``worker_id``
以及实际执行的 ``gpu_device``。``scheduler.jsonl`` 逐任务记录派发和完成时序，
``progress.json`` 记录就绪/活动/完成数量；日志及性能计数按 worker 保存。

母链和任务结果用临时文件加原子替换发布；采样完成凭据写在初态和母态之后。
工作进程崩溃后，已经提交的凭据可以恢复；没有完成凭据的碰撞保留为 failed/unknown，
不自动重跑或改抽一个初态。采样失败会停止该链后续样本，其他链继续。
单纯工作进程初始化失败不会丢弃任何母链配额，剩余进程可以接手所有链。
所有未完成/未发射的轨迹仍保留在请求的 N 分母内。

调度压力测试
------------

    python benchmarks/scheduling/run.py --workers 4 --trajectories 16 \
      --rounds 3 --output scheduler-benchmark.json

该测试对每个全局轨迹编号指定相同的 sleep 耗时，比较真实 static/dynamic 驱动。
它包含启动、热化、文件 I/O 与退出，使用短程自由粒子势，只证明负载调度机制，
不是 HHOO PES 加速比，更不是化学速率结果。原始结果保留每轮墙钟时间与动态事件。
