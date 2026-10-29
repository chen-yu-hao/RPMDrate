直接 RPMD：热化初态与三种执行模式
====================================

本功能实现 direct RPMD（dRPMD）碰撞轨迹：在远距离约束反应物质心进行
PIMD 热化，保存完整珠子坐标和动量，然后去掉恒温器与约束，用
kick–Cayley–kick 传播碰撞，按产物通道统计。它不需要过渡态几何或 PMF。

此实现沿用本仓库 ``19ba260`` 的完整 Cayley 中点映射，增加每个对象独立的
缓存和 OpenMP 内核。旧 ``rpmdrate.main.RPMD``、PMF 和 recrossing 接口未被替换。
新入口不需要编译旧 NumPy/f2py 扩展。

安装与快速运行
----------------

在仓库根目录使用 Python 3.10 或更新版本。NumPy 与 threadpoolctl 是 CPU
必需依赖，GCC/OpenMP 用于本地加速内核；CUDA 模式另需匹配驱动的 PyTorch。
参考运行环境是 Linux；macOS 需要另行配置带 OpenMP 的编译器，原生 Windows
的构建脚本尚未验证，可在 WSL/Linux 使用。

.. code-block:: bash

    python -m pip install -r requirements-drpmd.txt
    make drpmd-native python=python

    # 1. 单核、单条碰撞轨迹
    python -m rpmdrate.drpmd examples/drpmd/hhoo_smoke.json \
      --mode serial --trajectories 1 --output drpmd-results-serial

    # 2. 一条完整环聚合物轨迹，PES 与动力学均使用多核
    python -m rpmdrate.drpmd examples/drpmd/hhoo_smoke.json \
      --mode bead --workers 4 --trajectories 1 --output drpmd-results-bead

    # 3. HHOO 五网络 PES 与其梯度在 CUDA 上批量计算
    # 先按 https://pytorch.org/get-started/locally/ 安装适合硬件的 PyTorch。
    python -m rpmdrate.drpmd examples/drpmd/hhoo_smoke.json \
      --mode gpu --trajectories 1 --output drpmd-results-gpu

也支持仓库原入口：

.. code-block:: bash

    python rpmdrate.py --drpmd examples/drpmd/hhoo_smoke.json \
      --mode bead --workers 4 --output drpmd-results-wrapper

``hhoo_smoke.json`` 只检查程序流程：8 beads、100 步热化、100 步碰撞。
其中 HO₂ 几何是示例种子，键阈值与距离也是待验证的输入参数，不是论文数据、
经过验证的势能面平衡结构或收敛的生产设置。该短任务通常以 ``unresolved``
结束，这是预期结果，不应将其解释为零反应速率。

``free_particles.json`` 提供无相互作用粒子的独立流程检查。它应得到散射返回
反应物与零反应计数；它不是化学反应势能面。

三种模式的实际工作分配
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - 模式
     - 势能面
     - 动力学
   * - ``serial``
     - 一个进程，BLAS 线程限制为 1
     - C 内核一个线程；未编译时允许 NumPy 串行版本
   * - ``bead``
     - 持久化 spawn 进程池，每个 worker 初始化一次 PES，按珠子分块
     - C/OpenMP 并行 momentum kick 和完整 Cartesian ring 的 Cayley 解
   * - ``gpu``
     - 所有珠子的模型能量与自动微分梯度在 CUDA 设备计算，FP64
     - 一个 CPU 线程运行 Cayley；本模式没有宣称整条轨迹全驻留 GPU

``--workers N`` 在 bead 模式同时指定 PES 进程数与动力学 OpenMP 线程数。
PES 与动力学阶段顺序执行，PES worker 不嵌套额外 BLAS/OpenMP 线程。
一条轨迹内，各 Cartesian 分量都拥有完整的 P 个珠子；**不会把环切成独立小环**。
循环三对角求解沿每条 ring 的递推仍是顺序的，独立 ring 数量为 ``3*Natoms``。
因此小体系能利用的自由环求解并行度有限；更多核或更多珠子不保证净加速。
PES 分发、Python 检查和主机/设备复制也有开销，应以实际势与 bead 数测量。

选择 bead 而未编译 OpenMP 内核、选择 GPU 而没有可用 CUDA、选择 GPU 却只提供
普通 Fortran module，均明确报错。不会把串行或 CPU 执行标成多核/GPU。

``trajectories`` 大于 1 时依次运行多条碰撞轨迹；这些模式控制每条轨迹内部的
执行方式。本功能没有额外开启一层多轨迹并行。

输入文件与单位
----------------

JSON 顶层与 ``potential`` 必须为对象，未知顶层字段会报错。
**fragments 使用从 0 开始的原子编号**，与旧输入的从 1 开始编号不同。
所有低层数组保持 ``(3, Natoms, Nbeads)`` 的 RPMDrate 排列。

.. list-table:: 主要输入字段
   :header-rows: 1
   :widths: 32 68

   * - 字段
     - 含义
   * - ``atoms``, ``masses_amu``
     - 原子符号与同位素质量。HHOO 模型严格要求 H,H,O,O 顺序。
   * - ``fragments``
     - 两个非空、无重复、覆盖全部原子的索引列表，例如 ``[[0],[1,2,3]]``。
   * - ``initial_geometry_bohr``
     - ``Natoms x 3`` 初始种子几何，单位 bohr。热化前按指定距离平移两反应物。
   * - ``temperature_K``, ``nbeads``, ``dt_fs``
     - 物理温度、珠子数、步长；内部转换为原子单位。
   * - ``thermalization_steps``
     - 新几何的热化步数，至少覆盖一次 Andersen 重采样；生产值需自行收敛。
   * - ``sampling_steps``
     - 相邻热化样本间的传播步数。有限间隔不保证样本独立。
   * - ``andersen_interval``
     - 热化期间每隔多少整步重采样珠子动量，必须为正整数。
   * - ``initial_separation_bohr``, ``bmax_bohr``
     - 初始两个 centroid COM 的距离 R₀ 与最大碰撞参数；要求 R₀ > bmax > 0。
   * - ``evolution_steps``, ``trajectories``
     - 单条生产轨迹的最多步数与轨迹条数。
   * - ``bond_cutoffs_bohr``, ``channels``
     - 明确的 centroid 成键阈值与产物计量映射，示例见下文。
   * - ``exit_distance_bohr``
     - 判定为解离产物时所有产物片段 COM 之间的最小距离。
   * - ``check_interval``, ``persistence``
     - 每隔多少步分类、相同分类需连续保持多少次检查。
   * - ``save_every``
     - CSV 能量/centroid 轨迹的输出间隔，另保存第一步和最后有效步。
   * - ``seed``
     - 非负整数。热化与碰撞采样使用独立 SeedSequence 子流。
   * - ``electronic_weight``
     - 用户指定的非负电子统计因子，默认 1；不会自动假设单/三重态权重。

热化与初态
------------

采用与原 Fortran 相同的哈密顿量和温度约定：

.. math::

   H_P=\sum_{s,i}\frac{p_{is}^2}{2m_i}+\sum_s V(q_s)
       +\frac12\sum_{s,i}m_i\omega_P^2(q_{is}-q_{i,s-1})^2,
   \qquad \omega_P=P/\beta

原子单位下平衡权重为 ``exp(-beta*H_P/P)``，每个珠子动量分量的方差为
``m_i*P/beta``。热化时固定每个反应物的 centroid COM，并投影其总质心动量；
分子内部振动、转动和非 centroid 珠子涨落保留。位置与动量一起随时间演化，
不是只在复制的平衡几何上抽取一次速度。

达到指定热化步数不等于已经证明平衡；应检查可观测量、独立重复与相关时间。
程序还检查热化后 centroid 成键图仍与指定反应物片段一致。失败时停止，不会
把已经裂解或连成复合物的样本当作原反应物初态。阈值本身仍需物理验证。

生产轨迹开始前，对两个反应物分别做均匀随机三维旋转，同时旋转所有珠子坐标
和内部动量。只替换片段 centroid COM 的相对平动，使总 centroid COM 动量为零。
之后关闭恒温器和热化约束，采用物理势半 kick、完整 Cayley、物理势半 kick。

保存的完整热化样本可直接重用：

.. code-block:: bash

    python -m rpmdrate.drpmd examples/drpmd/hhoo_smoke.json \
      --mode bead --workers 4 \
      --thermal-state drpmd-results-serial/thermal_state.npz \
      --output drpmd-results-reuse

重用的第一个样本不会隐式重新热化。核对温度、质量、珠子数、反应物分组、
热化距离和 PES 标识后，再抽取碰撞平动和取向。HHOO/TorchScript 的标识包含
模型文件内容 SHA-256，防止路径相同而权重已改变的样本混用。Python module
记录入口模块文件哈希及配置；通用模块的传递依赖/外部数据仍应由用户版本管理。

``thermal_state.npz`` 是第一个实际使用的热化样本，包含完整 q、p、质量、温度
与元数据。它不是随机流或整个任务的恢复点；不会重建中断任务的 RNG 状态。
``*_final.npz`` 可用于检查最终状态，通过 Python API 继续传播需自行保存/恢复
分类历史；CLI 不接受其冒充 thermal state。

碰撞抽样与速率估计
--------------------

相对碰撞能量采用入射通量加权分布：

.. math::

   E_{\rm col}\sim\operatorname{Gamma}(2,k_B T),\quad
   v=\sqrt{2E_{\rm col}/\mu},\quad
   b=b_{\max}\sqrt U,\quad \phi=2\pi U'.

``U,U'`` 为独立均匀随机数；初始纵向位移为 ``sqrt(R0**2-b**2)``，使 COM
间距恒为 R₀。产物 c 的计数估计与该抽样匹配：

.. math::

   k_c(T)=g_{\rm el}\,\pi b_{\max}^2\sqrt{\frac{8k_BT}{\pi\mu}}\,
          \frac{N_c}{N}.

输出单位为 ``cm^3 molecule^-1 s^-1``。不能把普通 Maxwell 速率抽样替换进来后
仍使用这个计数公式；若改为普通 Maxwell 分布，则需相应使用 ``mean(v*I_c)``。

最终状态分为：

* ``product``：已满足出口与持续性判据的指定产物通道。
* ``nonreactive``：已分离、向外运动的反应物返回。
* ``unresolved``：达到最多步数仍未判定，或形成未配置的产物计量。
* ``failed``：PES、数组或数值传播失败；停止后续轨迹，保留错误与最近有效状态。

存在 unresolved/failed 时，``rate_cm3_molecule_s`` 为 null，分别报告已观察
产物的下界与将所有未知结局计入该通道的上界。各通道上界不能相加解释成总速率。
这些是有限样本缺失结局的记账边界，不是置信区间。全部解算完成时另外输出
Wilson 95% 区间；它假设样本独立，不包含热化相关、PES 误差和数值收敛误差。

产物通道
----------

示例为 H + HO₂ 配置以下几何计量通道：

.. code-block:: json

    {
      "channels": {
        "2OH": ["HO", "HO"],
        "H2+O2": ["H2", "O2"],
        "H2O+O": ["H2O", "O"]
      }
    }

内部化学式按元素名排序；输入 ``OH`` 和 ``HO`` 会规范化成同一计量，显示标签
可保持 ``2OH``。程序检查每个通道的元素数守恒，并拒绝重复的计量通道。
先由 centroid 键长生成连通分量，再要求所有片段 COM 充分分离且相对径向速度
为正，最后要求分类连续出现 ``persistence`` 次。短暂的成键变化不会立即终止。
相同计量的原子交换通道需要用户通过 Python API 提供能区分原子身份的 classifier。

此几何分类不识别电子自旋。当前 HHOO 文件没有在代码层自动提供单重态/三重态
分解；需对对应 PES 分别运行，再按用户确定的电子统计与物理模型组合结果。

势能面接口
------------

现有 Fortran/f2py PES 可以通过可导入模块复用：

.. code-block:: json

    {
      "potential": {
        "type": "module",
        "module": "hhoo_pes",
        "function": "get_potential",
        "initializer": "hhoo_1014_init",
        "initializer_kwargs": {"weights_path": "/absolute/path/to/txt_1014"}
      }
    }

``get_potential(q)`` 返回 ``(V,dVdq)`` 或 ``(V,dVdq,info)``。坐标与梯度必须为
``(3,Natoms,P)``，V 为 ``(P,)``；分别使用 bohr、hartree/bohr、hartree。
梯度必须是正的 ``dV/dq``；动量更新中的减号负责转换为力。
``initializer`` 不存在时显式设为 null；有返回状态时要求其为 0。
``with_dimensions: true`` 支持需要显式 ``q,natoms,nbeads`` 的旧包装接口。
非零状态（包括负数）、NaN/Inf 与错误形状都立即失败。

模块可放在输入 JSON 同目录；多进程要求模块能被 spawn 子进程导入，不能用
交互式局部 lambda 代替。初始化资源路径应写为绝对路径。不要在回调中返回
已经加入弹簧、umbrella 或恒温项的梯度，这些不属于独立的物理势能。

已有 HHOO 模型直接使用：

.. code-block:: json

    {"potential": {"type": "hhoo", "weights": "../H-H-O-O/pes/txt_1014"}}

该类型提供 NumPy 解析梯度与 Torch 批量模型。两者保留原来的 7 个描述符、
5 个 7→50→50→1 网络、输入/输出缩放、``z/sqrt(1+z*z)`` 激活与五网平均，
没有重训练。CPU/Torch 结果已直接与仓库原始 Fortran 能量和梯度比较。

其他 GPU 势可导出 TorchScript；能量模型输入为 ``(P,Natoms,3)``，输出为
``(P,)`` 或 ``(P,1)``，每个珠子的能量必须彼此独立且可微：

.. code-block:: json

    {
      "potential": {
        "type": "torchscript",
        "model": "pes.pt",
        "length_scale": 0.529177210903,
        "energy_scale": 0.03674932217565499
      }
    }

上例适用于输入 Å、输出 eV 的模型。scale 将 bohr 转到模型长度单位、将模型
能量转为 hartree，自动微分包含对应的链式因子。模型应支持 FP64、eval 模式且
不混合不同 bead 的信息。TorchPotential 的显式 ``device='cpu'`` Python API
仅用于参考验证；CLI 的 GPU 模式要求 CUDA。

输出文件
----------

* ``input.json``：有效输入、模式、线程/进程数、原生内核情况、PES provenance。
* ``thermal_state.npz``：完整已热化珠子坐标/动量，可用于上述显式重用。
* ``trajectory_000001_initial.npz`` / ``_final.npz``：完整碰撞初态/最终有效状态。
* ``trajectory_000001.csv``：时间、每珠平均 ring Hamiltonian 与 centroid 坐标。
* ``trajectories.csv``：碰撞参数、结局、通道、步骤、能量漂移与失败原因。
* ``summary.json``：分类计数、速率估计/边界、单位、时间与收敛说明。

能量诊断为 ``H_P/P``，不是量子热力学能量估计器。输出目录必须尚不存在，避免
覆盖已有轨迹。PES 初始化/热化失败不会生成看似成功的速率报告。

验证与生产计算
----------------

.. code-block:: bash

    make drpmd-test python=python

基础测试不要求 PyTorch；装有 PyTorch 时增加 Torch CPU 一致性测试，装有
gfortran 时直接编译仓库原始 Fortran 做回归，装有 CUDA 时运行真实 GPU
数值一致性测试。没有相应依赖的用例明确标记 skip。GitHub Actions 验证
CPU 与 Torch CPU；普通托管 runner 不提供 CUDA 硬件。

覆盖：密集 Cayley 参考、P=1/2/奇数/偶数、缓存隔离、单核/多核完整轨迹、
谐振子热化分布、质心约束、通量抽样、成键分类、未判定边界、PES 错误与完整
状态保存，以及原始 Fortran HHOO/Cayley 和 Torch CPU 的直接比较。

物理生产结果仍需独立检验：时间步、珠子数、热化长度、样本间隔、轨迹条数、
最大碰撞参数、初始/出口距离和最长传播时间。对长寿命复合物保留 unresolved，
不能依据一个假定的统计分支比例补齐结果。对小型 HHOO 网络，多进程通信或
GPU 数据传输可能超过计算节省；实际性能以 ``summary.json`` 的计时和目标
硬件测量为准。

方法参考
----------

* Suleimanov et al., J. Phys. Chem. Lett. 9, 2133–2137 (2018),
  https://doi.org/10.1021/acs.jpclett.8b00783 。
* del Mazo-Sevillano et al., J. Phys. Chem. Lett. 10, 1900–1907 (2019),
  https://doi.org/10.1021/acs.jpclett.9b00555 。
* Korol, Bou-Rabee and Miller, J. Chem. Phys. 151, 124103 (2019),
  https://doi.org/10.1063/1.5120282 ，预印本 https://arxiv.org/abs/1907.07941 。

这里的通量抽样/计数公式由碰撞通量积分一致地配对；软件验证不代表模型训练域、
自旋指派或任何特定反应的实验符合程度已经得到验证。
