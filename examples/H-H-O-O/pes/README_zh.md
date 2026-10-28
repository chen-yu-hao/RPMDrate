# H–H–O–O 势能面：Fortran 解析梯度版本

本包以你提供的 `hhoo_1014.f90` 和 `txt_1014` 为依据，在保留原能量模型的基础上增加解析笛卡尔梯度。交付日期：2026-10-28。

**主要文件：`hhoo_1014_analytic.f90`。** 这是一个自包含的 Fortran 源文件；运行时还需要原权重文件 `txt_1014`。不需要 C/C++ 势内核、GPU 或自动微分框架。

## 1. 这个版本改变了什么？

保留原来的 7 个几何描述符、5 个 `7→50→50→1` 网络、各网络的输入/输出缩放、激活函数 `z/sqrt(1+z*z)`，以及五网能量平均。权重文件按原样包含，未重新训练、删减网络或修改数值。

增加一条解析链式求导路径：网络输出 → 7 个描述符 → 6 个逆距离 → 12 个笛卡尔坐标。一次调用同时得到能量和所有坐标的导数，不通过坐标扰动做中心差分。

此外增加了显式初始化、批量珠子接口和错误状态。原来两参数的 `hhoo_1014(x,v)` 保留；新增三参数接口另取名 `hhoo_1014_grad(x,v,g)`，避免破坏原来的调用约定。兼容能量接口目前复用解析梯度计算，不是专门优化的纯能量路径。

## 2. 原子顺序、单位和符号

| 量 | 形状/约定 | 单位 |
| --- | --- | --- |
| `x` | `(3,4)`，四列依次 H、H、O、O | bohr |
| `v` | 一个构型的势能 | hartree |
| `g` | `(3,4)`，**正梯度 `dV/dx`** | hartree/bohr |
| 物理力 | `force = -g` | hartree/bohr |
| `q` | `(3,4,Nbeads)`，第三维为构型/珠子 | bohr |
| `V` | `(Nbeads)`，每个珠子一个能量 | hartree |
| `dVdq` | `(3,4,Nbeads)`，正梯度 | hartree/bohr |

输入已经是 bohr 时，不要再次做 Å→bohr 换算。输出也已经是 hartree；无需套用另一套 eV 势的能量转换。

你的 RPMD 动量更新使用 `p = p - dt * dVdq` 的符号，因此 `get_potential` 直接返回这里的正梯度。不要先取负再传入 `dVdq`。

## 3. Fortran 最小调用

把本文件与调用程序一起编译，用它替代旧 `hhoo_1014.f90` 参与链接；新旧文件都定义了 `hhoo_1014`，不能同时链接。下面所有过程都是外部入口，调用者不需要 `use` 内部模型模块：

```fortran
integer :: result
double precision :: x(3,4), v, g(3,4), force(3,4)

! 在串行区初始化一次；可以换成明确的绝对路径。
call hhoo_1014_init('txt_1014', result)
if (result /= 0) stop 'PES initialization failed'

! 为 x 赋值；四列必须是 H,H,O,O，单位 bohr。
call hhoo_1014_grad(x, v, g)
force = -g
```

包内 `example_fortran.f90` 是完整的可运行示例。安装 GNU Fortran 的环境下，在包目录运行：

```bash
gfortran -O3 -frecursive hhoo_1014_analytic.f90 example_fortran.f90 -o example_hhoo
./example_hhoo
```

示例几何仅用于检查调用和单位，不代表指定反应的初始构型、过渡态或经过验证的训练域。

### 批量接口

```fortran
call get_potential(q, V, dVdq, Natoms, Nbeads, result)
if (result /= 0) then
    ! 停止或重试这条轨迹；不能继续使用此次输出。
end if
```

`Natoms` 必须是 4，`Nbeads` 必须至少为 1。这里没有自动除以珠数，也没有额外添加虚拟弹簧、umbrella bias 或恒温项；这些属于动力学程序。

## 4. Python / f2py 接口

包中有手写接口文件 `hhoo_pes.pyf`，只暴露公共函数，不把内部权重数组交给 Python。

### Linux 上编译

依赖：Python、NumPy、Python 开发头文件、GCC 和 GNU Fortran。`build_python.py` 直接使用 f2py 生成包装代码，再调用编译器；不要求安装 Meson 或 Ninja。

```bash
python build_python.py
python example_python.py
```

构建脚本只编译当前包，不安装软件。默认读取 `FC=gfortran`、`CC=gcc`，并使用 `-O3 -frecursive` 编译 Fortran；特殊环境可以设置 `FC`、`FCFLAGS`、`CC`、`CFLAGS`、`LDFLAGS`、`FCLIBS`。这是 Linux/GNU 构建辅助；其他平台可使用自己的 f2py 构建流程处理同一 `.pyf` 和 `.f90`。

### 调用

```python
import numpy as np
import hhoo_pes

status = hhoo_pes.hhoo_1014_init('/absolute/path/to/txt_1014')
if status != 0:
    raise RuntimeError(f'PES initialization failed: {status}')

# q 的形状必须为 (3, 4, Nbeads)，单位 bohr。
q = np.asfortranarray(q, dtype=np.float64)
V, dVdq, status = hhoo_pes.get_potential(q)
if status != 0:
    raise RuntimeError(f'PES evaluation failed: {status}')
```

`Nbeads` 和 `Natoms` 从数组形状推断，Python 调用只传 `q`。单构型也可以使用 `v, g = hhoo_pes.hhoo_1014_grad(x)`。

### 连接当前 RPMD 输入

若运行目录中有 `txt_1014`，势接口可写成：

```python
import hhoo_pes
get_potential = hhoo_pes.get_potential
initialize_potential = hhoo_pes.initialize_potential
```

`initialize_potential()` 是无参数入口，符合当前 RPMD 输入加载代码的调用方式。它在首次调用时读取当前工作目录中的 `txt_1014`；如果此前已成功显式初始化，则直接复用模型。反应输入的原子顺序仍须是 H、H、O、O。

本次验证覆盖势函数和以上包装接口；没有在生产 RPMD 中运行完整 PMF、recrossing 或速率收敛计算。

## 5. 初始化、并发和错误处理

- 每个进程在并行计算前串行初始化一次。首次懒加载不能由多个线程同时触发。
- 成功初始化后权重只读；计算例程使用独立局部工作数组。源文件将计算路径声明为 `recursive`，GNU 构建也保留 `-frecursive`。不要使用把局部临时量统一静态化的编译选项。
- 本版本没有自动开启 OpenMP。可以在外层并行处理独立构型或轨迹，但初始化必须先完成。
- 同一进程只加载一套权重。重复传入相同字面路径返回成功，初始化后传入不同路径返回 13；等价但写法不同的路径也视为不同。需要换模型时使用新的进程。
- `hhoo_1014_init` 返回状态；批量 `get_potential` 失败时返回非零状态，并将此次整批输出置为 NaN，避免把非法几何误当成功计算。
- `hhoo_1014_grad`、旧能量接口和无参数初始化入口没有状态输出；遇到错误使用 `error stop`。需要恢复错误时使用有状态的初始化和批量接口。

| 状态 | 含义 |
| --- | --- |
| 0 | 成功 |
| 1 | `Natoms` 不是 4 |
| 2 | `Nbeads < 1` |
| 3 | 非有限、重合或数值不可表示的几何 |
| 10 | 权重文件打开失败 |
| 11 | 权重文件读取或关闭失败 |
| 12 | 网络头部、权重数值或输入缩放宽度非法 |
| 13 | 已初始化后试图更换权重路径 |
| 20 | 能量或梯度计算得到非有限结果 |

读取器采用原 `txt_1014` 的固定结构，检查每个网络的五行头部、数值有限性和输入缩放宽度；连接索引的 101 行沿用原读取逻辑跳过。读取前将临时权重填为 NaN，以便检出未赋值的缺失项。

没有通过距离截断、描述符裁剪或力裁剪掩盖异常；有效输入仍使用原来的势能定义。

## 6. 数值验证与复现

`validation/` 包含原始能量参考、独立 NumPy 导数参考、验证脚本和结果。验证用于判断本实现是否保留原势并正确求导，不代表势的拟合误差或反应速率已得到验证。

验证包括原始 Fortran 能量对照、独立 complex-step 导数、单构型与批量的一致性、平移/旋转/同元素置换，以及异常输入和初始化状态。

本次交付源码使用 GNU Fortran 13.3、`-O2`、FP64 做独立回归，**40 项检查全部通过**。结果如下；它们是指定样本和编译环境中的观察值，不是跨平台误差上界。

| 检查 | 样本/结果 |
| --- | --- |
| 与未修改原始 Fortran 能量比较 | 64 个构型，最大绝对差 0 hartree |
| 与独立 NumPy 能量比较 | 64 个构型，最大绝对差 2.84×10⁻¹⁴ hartree |
| 解析梯度与独立 complex-step 比较 | 24 个构型的全部 12 个分量，最大绝对差 3.64×10⁻¹⁵ hartree/bohr |
| 单构型与批量接口 | 64 个构型，能量和梯度差均为 0 |
| 净梯度最大分量 | 1.11×10⁻¹⁶ hartree/bohr |
| 净力矩最大分量 | 1.11×10⁻¹⁶ hartree |
| H/H、O/O 交换及平移/旋转协变 | 梯度最大差不超过 7.22×10⁻¹⁵ hartree/bohr |
| 非法几何、错误原子数、缺失/损坏权重 | 按预期返回失败状态；加载失败后可用有效文件重试 |

Python/f2py 扩展另以 `-O3 -frecursive` 构建，单构型、批量、兼容能量、显式/默认初始化和错误返回均已检查；Fortran 示例也已编译运行。记录见 `python_interface_results.json`。

在安装了标准 GNU Fortran 与 NumPy 的环境中，可以自行重编并复现独立回归：

```bash
python validation/validate_release.py
```

该脚本使用 `FC`、`FFLAGS` 和可选的 `GFORTRAN_LIB` 配置编译器；详细命令见脚本帮助。独立回归的 `FFLAGS` 与可选 Python 构建脚本使用的 `FCFLAGS` 是各自的环境变量。

完整数据和编译命令见 `validation/validation_results.json`。`manifest.json` 记录交付源文件和原权重的 SHA-256；包内没有训练数据。

## 7. 文件目录

| 文件 | 用途 |
| --- | --- |
| `hhoo_1014_analytic.f90` | 单文件 Fortran 势与解析梯度，含全部公共入口 |
| `txt_1014` | 用户提供的原始五网络权重，原样保存 |
| `example_fortran.f90` | 最小 Fortran 调用示例 |
| `hhoo_pes.pyf` | 可选 Python / f2py 接口定义 |
| `build_python.py` | Linux/GNU 的可选 Python 扩展构建脚本 |
| `example_python.py` | 单构型和批量 Python 调用示例 |
| `python_interface_results.json` | Python 包装接口与 Fortran 示例的运行记录 |
| `validation/` | 可复现的正确性检查与结果 |
| `manifest.json` | 交付文件和原输入的校验记录 |

压缩包只包含源码、权重、说明和验证记录；请在目标机器重新编译。
