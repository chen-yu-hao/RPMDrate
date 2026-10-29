# HHOO: 122 条轨迹续算与 2OH 步长复核

## 数据与版本

源运行：`40909399592`；源代码：`60ea1c30672f67b36dc7c070bf0a787c4be4d117`。
新分支以这个不可变提交为基础，不合并之后其他分支的势能面或动力学改动。
八个 `hhoo-b*-s*` 原始 artifact 保留完整初始、最终珠子 q/p 和分组信息。
读取时检查 512 个记录、122 个 unresolved、两个指定 2OH 候选、珠子数、质量、温度、片段分组、势模型 SHA-256 和每个状态文件 SHA-256。

## 续算定义

200 ps 指从原始碰撞发射起累计 200 ps：从 20 ps 末态继续最多 180 ps。
82 条属于 bmax=6 bohr，40 条属于 bmax=10 bohr。
32 beads；dt=0.1 fs；使用原来的 NVE kick-Cayley-kick，关闭恒温器，不重新热化、发射、旋转或重采样动量。
仍保留原来的产物/非反应出口与持续性判据，提前解决的轨迹可以早停。

原始最终 NPZ 没有保存分类器的连续确认次数，所以在 20 ps 边界保守地从零开始计数，不能声称与原来连续运行的早停时刻完全相同。q/p 完全继承，动力学时钟不归零。新的检查点包含绝对步数、梯度、q/p、能量参考、分类标签与确认次数。

检查点每 5 ps 或 120 秒墙钟保存一次。完整 q/p 和 CSV 也会保存在终止点。CSV 每 10 fs 保存 centroid 坐标、六组原子对距离、原始反应物分组 COM 距离和能量分量。COM 在重排后只是一组固定原子标签的诊断量，不等价于真实产物片段距离。

## 能量与步长检查

对象：原始 `hhoo-b6-s3` 中轨迹 8（chain 1/local 8）与轨迹 51（chain 4/local 3）。
每一条从同一个原始碰撞初态 q/p 重放三次：dt=0.1、0.05、0.025 fs。不是从已经形成 2OH 的末态反向重建，也不重新热化。
三个步长使用相同的物理检查间隔 1 fs、同样连续三次确认；每 1 fs 保存诊断，最长 20 ps。
0.1 fs 重放必须再现原始通道、停止步数和最大能量漂移（容差 1e-7 Hartree）后，长续算才会启动。

能量指标是 `H_P/P`，不是量子热力学能量估计器；漂移每个积分步计算。续算的总漂移始终相对最初发射能量 E(0)，同时单独保留相对 20 ps 续算起点的局部漂移。
绝对电子能量有较大负常数，因此不以 abs(E_total) 作为相对误差的归一化分母；复核提供漂移/初始相对碰撞能量作为诊断，但它不是严格的可接受性阈值。
不同步长轨迹可能在强非线性动力学中分离，因此不能要求逐点 q/p 长期重合，也不能仅因一条结果不同就断言算法错误；应检查能量误差、产物出口、相同时间窗口与统计收敛。

## 并行与产物

16 个续算 shard，每个分配 7 或 8 条原始轨迹；每个 job 同时四个单线程轨迹。最多四个 job 同时运行。不使用开销较高的 bead 多进程模式。
工作流先验证数据与 restart 自测，再执行六次步长复核；通过后启动 122 条续算。
每个 job 有有限超时；若失败，尽力上传现有检查点和错误报告，不会将失败视为非反应。

## 汇总

原来 388 条 nonreactive 与两条 2OH 记录不重新采样。用 122 条续算结局更新相应的原始记录，bmax=6/10 各自统计分母仍为 256，而非 122。
达到 200 ps 仍未解决的记录继续为 unresolved。缺失/失败记录也保留在未知结局分母中。
步长复核结果单独报告，不作为六条新独立样本，不混入 dt=0.1 fs 的原始系综。
输出的速率下上界是未知结局的记账界限，不是置信区间；父链相关性、初始距离/出口阈值、自旋归属、势模型精度和步长误差仍需单独评估。

## 入口

```bash
python -m rpmdrate.drpmd_integrator --build --compiler gcc
python -m benchmarks.hhoo_followup selftest
python -m benchmarks.hhoo_followup prepare --source source-data --out plan
python -m benchmarks.hhoo_followup group --kind audit --source source-data --out audit-results
python -m benchmarks.hhoo_followup group --kind continuation --part 0 --source source-data --out continuation-0
```

`source-data` 下必须有八个原始 artifact 的命名目录。
`integrate(..., restart=...)` 可恢复新格式运行中检查点；当前 CLI 没有自动跨 GitHub job 重启，不能把单纯重试整个 job 误称为检查点续跑。
