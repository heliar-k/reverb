# torch.Tensor 支持设计文档

> 本文档阐述在本 fork（纯 numpy、无 TensorFlow）中引入 `torch.Tensor` 作为 numpy
> 之外第二种数据格式的方案，重点回答"GPU 上的 torch Tensor 能为传输和训练加速
> 带来什么、带不来什么"。外部事实的考证见
> [torch-tensor-research.md](torch-tensor-research.md)（41 处官方来源引用），
> 本文只做设计决策。

## 目录

- [1. 两个前提事实](#1-两个前提事实)
- [2. 核心结论：GPU 常驻无法穿越传输层](#2-核心结论gpu-常驻无法穿越传输层)
- [3. 总体方案：边缘转换，服务器零改动](#3-总体方案边缘转换服务器零改动)
- [4. 开发计划](#4-开发计划)
- [5. 具体改动点](#5-具体改动点)
- [6. dtype 覆盖与边界情况](#6-dtype-覆盖与边界情况)
- [7. 非目标](#7-非目标)
- [8. 测试与验收](#8-测试与验收)

---

## 1. 两个前提事实

**架构事实（本仓库）**：所有传输（gRPC / in-process / SHM）、压缩、checkpoint 都
收敛到唯一枢纽类型 `TensorBuffer`（`reverb/cc/support/tensor_proxy.h:47`，
`{dtype, shape, bytes}`）。numpy 的假设只存在于两个函数：

- 写入：`TensorBuffer::FromNdArray`（`tensor_proxy.cc:156`）— 唯一的 numpy→bytes
  入口，pybind `type_caster` 自动调用；
- 读出：`TensorBuffer::ToNdArray`（`tensor_proxy.cc:233`）— 唯一的 bytes→numpy
  出口。

之下全部是 opaque bytes，dtype/shape 无关。服务器端 `schema.proto` 明确不解释
tensor 字节。

**外部事实（已考证）**：

- CUDA 张量的设备指针**仅在创建它的进程内有效**（NVIDIA CUDA 编程指南，跨进程
  需 CUDA IPC）；经 CPU opaque-byte replay server 传输时，一次 D2H 拷贝**不可
  避免**，无论用什么协议。
- CPU 上 numpy↔torch **双向零拷贝**（`torch.from_numpy` / `tensor.numpy()` /
  DLPack）。
- 社区标准模式（SB3 / RLlib / TorchRL 均如此）：**CPU 存储，采样时
  `pin_memory()` + `.to('cuda', non_blocking=True)` 异步上 GPU**。TorchRL 的全
  CUDA buffer 仅限同进程，不适用于网络分离的 replay server。

## 2. 核心结论：GPU 常驻无法穿越传输层

用户直觉"tensor 提前放 GPU 加速传输"需要修正：**传输段本身必须是 host 内存**。
client→server→worker 任何一程跨进程/跨机，protobuf 字节流里的设备指针毫无意义。

GPU 常驻真正能加速的是传输段的**两端**：

| 环节 | GPU 能做什么 |
|---|---|
| 写入侧（D2H 之前） | GPU 预处理：frame stacking、压缩、降采样 —— **减少传输字节数**，然后一次 D2H |
| 传输段 | 无。字节流走 CPU 内存 |
| 采样侧（H2D 之后） | pinned memory + `non_blocking=True` 异步 H2D，与训练计算**重叠**；批量组 batch、GPU 上训练 |

同机 colocate server 与 learner 时存在跳过 D2H/H2D 往返的可能（CUDA IPC），见
§4 Phase 3，但那是独立优化，不属于"torch.Tensor 支持"本身。

## 3. 总体方案：边缘转换，服务器零改动

转换放在两个边缘，且都是零拷贝或一次必要拷贝：

```
torch.Tensor (CPU/CUDA)
   │  detach() → [.cpu() 若 CUDA] → .numpy()     ← 写入边缘：CPU 零拷贝 / CUDA 一次 D2H
   ▼
TensorBuffer → TensorProto bytes → server（ opaque，零改动 ）→ 读出
   │
   ▼  numpy leaf
torch.from_numpy(leaf)  →  batch  →  pin_memory + to('cuda', non_blocking)   ← 采样边缘：零拷贝 + 异步 H2D
```

**关键决策：转换做在 Python 层，C++ 层零改动。** 否决的备选是在
`FromNdArray` 里 `py::module::import("torch")`（与现有 `import("numpy")` 同手法）：

1. `structured_writer.py:372` 的 spec 校验在 Python 层读 leaf 的 `.dtype`，假设
   numpy dtype —— C++ 层转换绕不过它，Python 层仍要改；
2. Python 层一次归一化后，三客户端（`Client`/`LocalClient`/`ShmClient`）和三种
   writer 全部自动受益，改动面最小；
3. 保持 C++ 核心与 ML 框架解耦 —— 这正是本 fork 去掉 TF 的动机。

**torch 必须是可选依赖**：懒导入（只在真的收到 torch.Tensor 时 import），
打包为 `pip install dm-reverb-numpy[torch]` extra。无 torch 环境下行为与今天
完全一致。

**FAQ：为什么不在 C++ 层持有 `at::Tensor`（链接 libtorch）？** 因为省不掉任何
拷贝：传输介质（gRPC/SHM/checkpoint）本质是字节流，CUDA 的 D2H 和 CPU 的
memcpy 无论在哪一层做都原样存在；设备指针跨进程无效，C++ 持有 CUDA tensor 没有
传输优势。而代价是把本 fork 删掉的 TF 式重依赖（构建、wheel、ABI/CUDA 对齐）
以 libtorch 形式请回来。底层最优恰恰在于它是 **opaque bytes 而非任何框架类型** ——
框架无关性是字节换来的。C++ 侧真正值得做的优化是框架无关的零拷贝升级
（`tensor_proxy.h:39-44`，等 profile 说话）。唯一让 C++ torch 成立的形态是
「全 GPU 常驻 server」，那是另一个产品，见 §7 非目标。

## 4. 开发计划

### Phase 1 — MVP：写入接受 torch，采样可选返回 torch

- 新增 `reverb/torch_support.py`：
  - `to_numpy_leaf(x)`：`torch.Tensor` → `detach()` → CUDA 则 `.cpu()` →
    `.numpy()`；非 tensor 原样返回（走现有 `np.asarray` 路径）。`bfloat16` 等
    无 numpy 对应的 dtype 明确报错。
  - `is_available()`：懒导入探测。
- 写入路径挂点：三个 writer 入口归一化（见 §5）。转换在调用线程持 GIL 执行，
  与现状 `FromNdArray` 一致，无线程模型变化。
- 采样路径：`LocalClient`/sampler 输出处加 opt-in（如 client 构造参数
  `output_format="torch"`），对每 leaf `torch.from_numpy`（零拷贝）；torch 不
  支持的 dtype（`uint16/32/64`、string）自动 fallback 保留 numpy leaf。
- 默认输出仍是 numpy —— 零行为变更。

### Phase 2 — 性能：让必要的拷贝更快

- 写入侧 CUDA tensor：`.cpu()` 前先 `pin_memory()`（pinned D2H 显著更快）；
  注意 pinned buffer 复用需要同步语义，先测量再优化。
- 采样侧提供 batch 级 helper：`batch.pin_memory().to('cuda', non_blocking=True)`，
  文档给出与 learner 训练循环重叠的用法（PyTorch 官方 CUDA notes 模式）。
- benchmark：写入吞吐 numpy vs torch-CPU（预期持平，零拷贝）vs torch-CUDA
  （多一次 D2H，给出实测开销）。

### Phase 3 — 可选：同机 GPU-direct（独立设计文档）

仅当 server 与 learner 同机 colocate 且 profile 证明 D2H/H2D 往返是瓶颈时：
基于本 fork 的 SHM 传输层做 CUDA IPC tensor 池（`torch.multiprocessing` 的
CUDA 路径，spawn/forkserver、Linux-only、producer 必须保活、consumer 尽快释放）。
限制多、语义复杂，**不进入 Phase 1/2 范围**。

## 5. 具体改动点

| 文件 | 改动 | 说明 |
|---|---|---|
| `reverb/torch_support.py` | 新增 | 懒导入 torch；`to_numpy_leaf` / `from_numpy_leaf` |
| `reverb/client.py:490,617,683` | `_BaseClient.writer` / `trajectory_writer` / `structured_writer` 入口 | 对嵌套结构逐 leaf 过 `to_numpy_leaf`（tree.map） |
| `reverb/structured_writer.py:372` 之前 | spec 构建前归一化 `step_spec` | 消除 `.dtype` 的 numpy 假设 |
| `reverb/trajectory_writer.py:248` `append` | 入口归一化 | 逐步 append 的数据同样转换 |
| `reverb/client.py:482` 附近 | 采样组包处加 `output_format` 分支 | `np.asarray` 之后可选 `torch.from_numpy` |
| `pyproject.toml` | extras `[torch]` | 可选依赖 |
| C++ 全部 | **零改动** | `FromNdArray` 收到的已是 numpy |

注意：`tensor_proxy.cc` 里记录的零拷贝升级方向（`tensor_proxy.h:39-44` 的
ponytail 注释：裸指针 + DeferredFreeQueue）与本设计正交，不受影响。

## 6. dtype 覆盖与边界情况

写入（torch → numpy，`torch.from_numpy` 文档的 dtype 列表即支持面）：

- 支持：`float64/32/16`、`complex64/128`、`int64/32/16/8`、`uint8`、`bool` —
  覆盖 RL 常见需求。
- **报错**：`bfloat16`（numpy 无原生对应；后续可经 `ml_dtypes` 支持）、
  量化/稀疏 tensor。
- 自动处理：`requires_grad` → `detach()`；非连续 tensor → `.numpy()` 保留
  strides 后由现有 `ascontiguousarray` 兜底（`tensor_proxy.cc:170` 逻辑）；
  0-d tensor 保持 0-d（不提升维度，对齐 numpy 标量语义）。

读出（numpy → torch）：上表的反向；proto 里存在但 torch 不支持的
（`uint16/32/64`、`string`）fallback 为 numpy leaf，不报错。

## 7. 非目标

- GPU 常驻数据穿越传输层 / 服务器存储 GPU tensor（物理不可能，见 §2）。
- in-process `LocalClient` 内 DLPack 直传 CUDA tensor：会破坏 checkpoint 与
  bytes 语义，lifetime 管理复杂，不做。
- 修改 C++ 服务器、proto、SHM 布局。
- 让 torch 成为硬依赖。

## 8. 测试与验收

- 单元：`torch_support` 各分支（CPU / CUDA / 非连续 / `requires_grad` /
  bfloat16 报错 / 无 torch 环境 import 安全）。
- 集成：三客户端写入同一批 torch.Tensor 与写入等价 numpy 后读出逐位一致；
  `output_format="torch"` 往返。
- 回归：全量现有 Python/C++ 测试（torch 缺失时全部跳过新用例）。
- benchmark（Phase 2）：写入吞吐三档对比；采样 + 异步 H2D 对 learner step
  时间的改善。
- 无 GPU CI 环境：CUDA 用例 `pytest.mark.skipif(not torch.cuda.is_available())`。
