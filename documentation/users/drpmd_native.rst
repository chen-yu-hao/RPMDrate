原生 C/Fortran 传播与并行效率
===========================

原生 PES 与原生传播是两个独立选项。前者只把求力放入 Fortran/OpenMP；
后者进一步把整个时间步循环、Cayley 因子构造、动量更新、能量计算、
热化 COM 约束、Andersen 动量替换和失败回滚放入 C。
C 循环直接调用 PES 的 C ABI 函数指针，不逐步返回 Python。
HHOO 的 Fortran 数值内核及 txt_1014 权重保持不变。

Python 保留 JSON/Python 输入、参数校验、轨迹调度、碰撞初态抽样、
用户自定义产物判据及输出。热化随机数使用原 NumPy Generator 分批产生，
以保持原算法的随机数流；不是换成另一个 C 随机数发生器。
因此这是原生数值核心，而不是移除所有 Python 依赖。
GPU/Torch 和任意 Python PES 继续使用原 Python 传播路径。

构建与使用
----------

在 Linux/GCC 环境下::

    python -m pip install -r requirements-drpmd.txt
    python -m rpmdrate.drpmd_integrator --build --compiler gcc
    python -m rpmdrate.drpmd_native_pes --build-hhoo

``drpmd_integrator --build`` 现在同时构建 C 积分块，仍生成
``rpmdrate/_drpmd_native.so``。必须重新构建旧库才能使用原生传播。
默认编译不使用 fast-math。可运行示例为 ``examples/drpmd/hhoo_native.json``；
该示例仅用于短程流程和性能测试，不是收敛速率输入。

每条轨迹单线程，同时四条轨迹::

    python -m rpmdrate.drpmd examples/drpmd/hhoo_native.json \
      --mode serial --trajectory-workers 4 --trajectories 100 \
      --dynamics-backend native --output native-four-trajectories

每条轨迹四线程，同时两条轨迹（约需要八个计算核心配额）::

    python -m rpmdrate.drpmd examples/drpmd/hhoo_native.json \
      --mode bead --workers 4 --trajectory-workers 2 --trajectories 100 \
      --dynamics-backend native --output native-hybrid

在相同原生势能上切回旧 Python 时间步循环做对照::

    python -m rpmdrate.drpmd examples/drpmd/hhoo_native.json \
      --mode serial --dynamics-backend python --output python-loop-reference

JSON 新增两个可选顶层字段::

    "dynamics_backend": "native",
    "native_block_steps": 256

``dynamics_backend`` 可为 ``auto``（默认）、``python``、``native``。
``auto`` 仅在使用 NativePotential 且已构建新 C 内核时选择原生路径；
输出会明确记录实际选择。显式 ``native`` 不会静默降级。
``native_block_steps`` 限制一次 C 调用最多传播多少步；到达产物检查或输出步时，
块会提前结束。因此 ``check_interval``、``save_every`` 和连续确认的时刻不变，
不会为了性能跳过反应通道判断。块边界也提供处理用户中断的机会。

Python API 同样可用（多进程入口需要 main guard）::

    from rpmdrate.drpmd import run
    summary = run(config, mode="serial", trajectory_workers=4,
                  pes_backend="native", dynamics_backend="native",
                  output="native-run")

低层 ``Dynamics(..., backend="native", block_steps=256)`` 配合现有
``propagate`` 或 ``thermalize`` 使用 C 积分块。
单独调用 ``Dynamics.step`` 仍保留原 Python 单步接口，便于自定义逐步实验。

数值和并发边界
--------------

每个 C context 拥有独立的参数、Cayley 因子和工作区。没有可写全局动力学状态。
质量、温度、珠子数和时间步变更时应创建新的 Dynamics，不要原地修改参数。
PES 与完整 Cartesian bead ring 分别使用 OpenMP；不切断周期珠子耦合。
小型动量与能量循环使用 C 标量循环，避免为几百个元素反复启动 OpenMP。
Cayley 使用 Dynamics 配置的线程数，PES 使用 NativePotential 配置的线程数。
两者依次执行，不能把两个线程数相加当成同时占用的核心数。

原生 PES 必须遵守只读输入、只读模型及局部工作区的线程安全约定。
原有非线程安全 PES 应继续使用进程隔离或共享内存后端；它们不能直接进入本 C 循环。
C 入口严格检查形状、缓冲区别名、有限值、每珠子状态和实际 PES 线程数。
每步保存完整检查点，求力或状态失败会恢复 q/p/V/gradient，并报告完成的步数。
这保证失败轨迹不会被记成不反应或从统计分母中丢弃。

热化随机动量按固定步时钟替换，每批随机缓冲区受大小限制。
改变 native_block_steps 不改变成功轨迹的随机数次序。
能量采用固定顺序 C 求和，与 NumPy 的求和舍入可能略有差异；
不把这种末位差异当成物理精度提升。

输出清单和单链 summary 新增 ``dynamics_backend``、``native_block_calls`` 等记录。
外层多链运行的各链目录包含同样信息。热化态仍保存完整 q/p。

并行效率测试
------------

可复现测试::

    OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE \
    python benchmarks/native/run.py \
      --threads 1 2 4 --beads 32 128 --steps 1000 --rounds 3 \
      --trajectories 8 --output native-efficiency.json

当前基准时间步为 0.05 fs，因此 1000 步是每条 50 fs。
``--steps 20000`` 可测试每条 1 ps，但这仍不是收敛化学速率计算。

单轨迹测试分别比较 Python/C 时间循环的 NVE 和受约束热化，
使用同一 Fortran PES、同一初态及相同随机数流，顺序随机化并保存每轮数据。
多轨迹测试固定总数为 8，不是每个进程 8 条；每条初态、传播步数均固定。
比较 1x1、2x1、4x1，以及 1x2、1x4、2x2 的资源分配。
工作进程预热和启动时间单独记录；暖态计时包含每轨迹一次任务派发和结果回传。
不包含文件 I/O、产物分类或初态准备，不能直接等同于完整生产任务墙钟时间。

加速比 S(c)=T(1)/T(c)，并行效率 E(c)=S(c)/c，c 为计算槽总数。
每条轨迹内部线程数与同时并行轨迹数是不同指标。
报告包含 CPU 型号、逻辑 CPU affinity、可读取的 cgroup quota、编译器、
源码 SHA-256、原始计时、实际 OpenMP 线程数及最终 q/p 对照误差。
共享云主机噪声、SMT、CPU 配额和线程等待策略均可能影响结果。
不设虚假的性能通过阈值，也不把超过配额的线程当作独立物理核心。

文档依据：Python ctypes CDLL 调用会在原生执行期间释放 GIL；GNU Fortran
OpenMP 要求并行代码局部工作区安全。
参考 https://docs.python.org/3/library/ctypes.html
以及 https://gcc.gnu.org/onlinedocs/gfortran/OpenMP.html 。
