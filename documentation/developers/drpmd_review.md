# dRPMD 实现与 Cayley 审查记录

审查基线：`chen-yu-hao/RPMDrate`，`master` 提交
`19ba2602bad5b2145b0d0ba0716d7ba3d90a1003`。日期：2026-10-29。

## 对原有 Cayley 改动的结论

重点检查 `6a4e0aa` 引入的循环三对角传播与 `19ba260` 的批处理/缓存优化。
原实现的中点方程为

\[
M=I+\tfrac14\Delta t^2\omega_P^2L,\qquad
Mq_{mid}=q+\tfrac{\Delta t}{2m}p,
\]
\[
q'=2q_{mid}-q,\qquad
p'=p-\Delta t\,m\omega_P^2Lq_{mid}.
\]

P=1 漂移、P=2 双边 Laplacian、P≥3 Thomas 分解与秩二周期修正是一致的。
质量、原子数、P、beta、步长变化后的缓存更新也通过了直接数值对照。
目前没有在这些检查范围内发现自由环 Cayley 数学错误。

新 C/OpenMP 内核实现同一映射，每个 Python 对象拥有独立因子和工作区。
不同 Cartesian ring 可并行；每条 ring 内保留完整 P 个珠子。新增 NumPy 实现
是相同映射的串行版本。没有将精确 free-ring rotation 混入对照，也没有把珠子
分块后各自闭合成较小环。

## 原代码中与本功能相关的限制

以下问题属于旧 PMF/recrossing 路径；本次通过独立 dRPMD 路径规避，未顺带
重构这些旧模块。

| 位置 | 观察 | 新功能的处理 |
|---|---|---|
| `_main.f90` 的 `system`、`CAYLEY_WORK` | dt/beta/mass 与工作区是模块级可写状态，不能多线程同时调用同一个实例 | 对象独立缓存；OpenMP 在一次完整调用内部工作 |
| `verlet_step_nve_cayley` | 每步仍调用反应坐标/TS 相关函数，不是无需 TS 的通用碰撞入口 | 独立的无约束、无恒温器 `Dynamics` / `propagate` |
| `main.py: RPMD.activate` | 生产传播已为 Cayley，仍构建密集 normal-mode 矩阵，并引用 NumPy 2 已删除的 `longcomplex` | 新入口不依赖 `RPMD.activate` 或旧 f2py 构建 |
| `_main.f90` 的 Andersen 步数 | `int(sampling_time/dt)` 可为 0，随后出现 `mod(step,0)` | 新输入要求正整数 thermostat interval，分段采样保持时钟 |
| `_math.f90: randomn/random_init_seed` | Box–Muller spare 缓存不随 reseed 清空 | 新功能使用显式 NumPy Generator 与独立 SeedSequence 子流 |
| HHOO 原模型初始化 | 初始化读取可写权重状态，不宜并发首次调用 | 每个 spawn PES 进程仅初始化一次，失败状态立即传播 |

旧项目的 NumPy 2 构建/兼容性不在本次功能测试通过的声明范围内。

## 完成的实现

- `rpmdrate/drpmd.py`：热化、完整初态保存/复用、通量碰撞抽样、NVE 传播、
  三种执行模式 CLI、产物分类、未判定/失败结局与速率统计。
- `rpmdrate/drpmd_integrator.py`、`_drpmd_native.c`：可重入 Cayley、OpenMP
  kick 和完整 ring 求解，以及无编译器时的 NumPy 串行版本。
- `rpmdrate/drpmd_backends.py`：原有 f2py/Python 批接口、持久化珠子进程池、
  CUDA FP64 模型/自动微分接口，显式单位与错误校验。
- `rpmdrate/drpmd_hhoo.py`：使用仓库原始 `txt_1014` 的 NumPy 解析梯度与
  Torch 五网络批量模型；没有训练新势或替换权重。
- 三种模式示例、中文用户指南、独立 Makefile 目标及 GitHub Actions 工作流。

## 实际验证

本地环境：Python 3.12.14、NumPy 2.3.5、GCC 13.3、GNU Fortran 13.3、
PyTorch 2.15.0+cpu。Fortran 编译器仅在任务临时目录解包，PyTorch 仅安装到
任务临时依赖目录；这些目录不是项目运行要求。

通过最终 Makefile 入口运行完整集合：**64 项用例，63 项通过，1 项因缺少
CUDA 设备跳过**，本次耗时 17.834 秒。标准环境可用下列命令运行：

```bash
make drpmd-test python=python
```

安装 PyTorch 后会增加可运行的 Torch 检查；`FC=gfortran` 指定原始 Fortran
对照用的编译器。缺少对应可选环境时，相关用例明确显示 skip。

| 验证 | 范围与结果 |
|---|---|
| 新内核与独立密集解 | P=1、2、3、5、8、32；正、负、零步长；通过 |
| 新内核与原始 Fortran | P=1、2、3、5、32、128；质量、原子数、beta、dt 改变；通过 |
| 自由环与非线性轨迹 | 能量、centroid、时间反演；完整 64 步 HHOO 对比原 Fortran PES+Cayley；通过 |
| 单核/多核 | NumPy、native 1/4 线程，PES 多进程与完整 CLI 同种子状态；通过 |
| 热化 | 可解析自由环/谐振子的 q,p 分布，含内部振动 centroid 模式与正确 P 温度因子；通过 |
| 碰撞与统计 | Gamma(2) 能量、均匀圆盘 b²、取向与 COM、持续产物分类、未知结局边界；通过 |
| HHOO 移植 | 原 Fortran、NumPy 解析梯度、Torch CPU 自动微分、对称性和有限差分；通过 |
| 错误与保存 | 非零 PES 状态、失效 initializer、错误 shape/NaN、模型内容变化、完整 NPZ、目录防覆盖；通过 |
| CUDA | 无硬件；实机一致性用例跳过，无 CUDA 时显式失败行为通过 |

此外实际执行了 `hhoo_smoke.json` 的串行与 4 核 CLI。相同种子下最终状态的
最大差异为 q `1.78e-14 bohr`、p `7.82e-13` 原子单位。两条 100 步轨迹都为
`unresolved`，summary 中速率为 null，而非伪造零速率。
无相互作用示例在第 1500 步满足向外/持续性退出，正确记录为 nonreactive。
这些数值是指定测试的结果，不是跨平台误差上界或反应速率收敛证据。

## 尚需在目标体系/硬件完成的工作

CUDA 能量/梯度与性能需要在实际 GPU 上验证；没有提供未经实测的加速比。
PES 物理精度、训练域、自旋归属，以及时间步、珠子数、热化、冲量参数、距离、
样本相关性和长寿命复合物的收敛需要用户针对研究体系完成。
新功能不自动指定单/三重态权重，也不会用假设分支补齐 unresolved 轨迹。
