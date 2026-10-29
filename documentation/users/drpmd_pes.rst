低开销 CPU 势能面接口
====================

轨迹级并行 ``--trajectory-workers`` 与每条轨迹的 PES 调用后端相互独立。
``serial``、``bead``、``gpu`` 模式含义不变；这里新增 CPU 后端选择，
不改动热化母轨迹、碰撞抽样或 Cayley 周期珠子耦合。
默认仍为原有 ``process``，便于兼容旧输入和做对照。

调用方式
--------

原有输入可以直接测试共享内存，不必修改势能模块::

    python -m rpmdrate.drpmd examples/drpmd/hhoo_smoke.json \
      --mode bead --workers 2 --trajectory-workers 4 \
      --pes-backend shared_memory --trajectories 1000 --output shared-results

这里总计 1000 条碰撞轨迹，4 条独立热化母轨迹，每条母轨迹约发射 250 条子轨迹。
每条轨迹有自己的共享内存和 2 个 PES 工作进程；Cayley 动力学也使用 2 线程。
PES 与 Cayley 分阶段运行，不是两套线程同时各占 2 核；另有调度进程开销。
建议按约 ``trajectory_workers * workers`` 规划计算核心，并实测是否超额订阅。
示例仍是 smoke test，未提供用于生产速率的收敛参数。

也可将 ``cpu_backend`` 放入 JSON 的 ``potential`` 对象::

    "potential": {
      "type": "hhoo",
      "weights": "../H-H-O-O/pes/txt_1014",
      "cpu_backend": "shared_memory",
      "startup_timeout": 60.0,
      "evaluation_timeout": 300.0
    }

超时单位为墙钟秒；昂贵的从头算 PES 应相应增大 ``evaluation_timeout``。
``--pes-backend`` 或 Python ``run(..., pes_backend=...)`` 覆盖该配置。
CPU 后端不是 GPU 选项；指定 ``shared_memory/native`` 后不能直接改为 GPU 模式，
应移除 CPU 选择或使用默认 ``process`` 选项。GPU 设备分配和每卡轨迹并行实现未改动。

共享内存后端
------------

``SharedMemoryPotential`` 为一条轨迹创建固定大小的 ``q/V/dVdq`` 缓冲区、
持久化 spawn 进程和固定珠子分块。每个进程独立初始化 PES 一次。
每次调用只通过 Pipe 发送步次/完成/错误信息，坐标和梯度不再被 pickle 后经队列传输。
珠子数少于进程数时只创建有工作的进程。

这不是全程零拷贝：母进程仍需把新坐标复制到共享区，通用 Python/f2py 回调保留
局部输入保护拷贝，返回值也复制为调用者独立持有的数组。这样下一次求力不会覆盖
积分器仍在使用的旧梯度。共享内存只用于 PES，完整环的自由传播仍保留全部珠子。

工作进程按不重叠切片写结果，完成信号之后母进程才读取；没有每珠子 Python 任务调度。
异常、非有限值、超时、工作进程硬退出都会报错并回收进程及共享区。
同一实例中的调用会串行化；并发轨迹各自使用独立实例。
这是对非线程安全 Fortran 模块最保守的优化，不要求将旧模块改成可重入代码。

HHOO 原生 Fortran/OpenMP 后端
----------------------------

在仓库根目录构建::

    python -m rpmdrate.drpmd_native_pes --build-hhoo

需要支持 OpenMP 的 gfortran。生成
``examples/H-H-O-O/pes/libhhoo_drpmd.so``，采用
``-O3 -fPIC -frecursive -fopenmp -shared``，不使用 fast-math。
它链接仓库中未修改的 ``hhoo_1014_analytic.f90`` 与新增的薄 C ABI 适配器，
没有改写网络、权重、描述符或解析梯度。

将位于 ``examples/drpmd/`` 的输入中的势能配置写为::

    "potential": {
      "type": "hhoo",
      "weights": "../H-H-O-O/pes/txt_1014",
      "native_library": "../H-H-O-O/pes/libhhoo_drpmd.so",
      "cpu_backend": "native"
    }

相对路径均相对于输入 JSON 的目录。然后运行::

    python -m rpmdrate.drpmd examples/drpmd/input.json --mode bead --workers 4 \
      --trajectory-workers 2 --trajectories 1000 --output native-results

每个轨迹进程只加载一份 PES 库，用一次 C ABI 调用让 Fortran OpenMP 循环计算全部珠子。
它没有珠子级 Python 子进程，也不经过 f2py。``serial`` 模式使用同一库的单线程版本。
``input.json`` 和单链 ``summary.json`` 记录实际选择的 ``pes_backend``；原生后端另记录
最后一次调用实际使用的 OpenMP 团队大小 ``pes_last_threads_used``。

权重先串行加载，之后只读；原 HHOO 数值子程序为 RECURSIVE 且工作数组局部化。
不能把未经检查的 COMMON/SAVE 全局工作区直接放到 OpenMP 循环里。
同一个进程/库只允许同一模型内容；换权重或库文件需要启动新进程。
共享内存与 process 不改变 PES 身份，可以复用同一物理参数的热化态；
native 身份额外包含共享库内容哈希，库变更会使热化态身份检查失败。

通用 C/Fortran ABI 与 Python API
------------------------------

接口约定见 ``examples/native_pes/drpmd_pes.h``，可运行 C 示例为
``examples/native_pes/harmonic.c``（仅谐振子测试势，不是化学势能面）。
导出版本、OpenMP 能力、串行初始化和批量求值四个函数；Fortran 用 ``bind(C)``。
坐标和梯度为 Fortran 连续的 ``(3,Natoms,Nbeads)`` FP64 数组，单位分别为
bohr、Hartree/bohr；能量是每珠子 Hartree，梯度是正的 ``dV/dq``，不是力。
整数 ABI 为 32 位，珠子逐个返回状态，错误不得 STOP/abort。

Python 可以直接使用独立的原生库，无须 Python 势能模块::

    from rpmdrate.drpmd_native_pes import NativePotential
    from rpmdrate.drpmd import Dynamics

    with NativePotential("/path/libpes.so", model="/path/weights.dat", threads=4) as pes:
        energy, gradient = pes.evaluate(q)
        dynamics = Dynamics(masses, temperature, nbeads, dt_au, pes, threads=4)
        # thermalize / propagate 的 API 不变。

已有 Python/f2py module 输入也可设置 ``cpu_backend: native``、``native_library``
和可选的 ``native_model``；此时调用原生实现而不是模块函数。
该模块的内容哈希仍作为势能来源记录；使用者必须确保原生实现与声明的模块对应。
也可使用 ``rpmdrate.drpmd_native_pes`` 中的标量 module 参考包装，做同内核传输对照。

已满足 FP64/对齐/Fortran 连续要求的输入直接传指针，不复制坐标；其他布局先转换。
输出由 Python 分配，生命周期独立。原生库必须遵守只读输入、只读模型和局部工作区约定；
Python 无法阻止有缺陷的原生代码越界或擅自修改输入。ABI 和数组形状检查并不是沙箱。

基准与验证
----------

运行真实 HHOO 权重基准::

    python benchmarks/pes/run.py --kernel hhoo --workers 2 \
      --beads 32 128 512 --calls 100 --rounds 5 \
      --nve-steps 64 --output pes-benchmark.json

对照项为：同一 Fortran 内核的单线程直接调用、旧进程队列、共享内存、OpenMP 直接调用，
另列现有 NumPy 的单进程、旧队列和共享内存实现。原生与 NumPy 是不同实现，
不应用它们的速度差来单独评价 IPC 开销。
启动时间单独计量；预热后随机化测量顺序；保留每轮原始时间、分位数、机器信息、
实际线程数及能量/梯度误差。另测包含 Cayley 的 NVE 步耗时（统一单线程动力学、无文件输出）。
没有性能门槛测试，不能把一台机器上的微基准加速比推广到所有节点和珠子数。

控制开销可用 ``--kernel harmonic --nve-steps 0`` 单独测量，但不能把谐振子结果
冒充 HHOO 势能或真实反应轨迹性能。

回归测试::

    python -m unittest tests.drpmd_transporttest -v
    make drpmd-test python=python

覆盖共享内存清理/异常/超时/硬退出/独立缓冲区、C ABI 和实际 OpenMP 线程、
HHOO 原始内核的能量/梯度一致性、热化与 NVE 轨迹，以及多轨迹嵌套共享内存/原生后端。
原始 Fortran、CUDA 与旧接口回归仍在完整测试集中。
