# torch.Tensor 支持 — Phase 1 Spec

> 上游设计文档：[torch-tensor-design.md](../design/torch-tensor-design.md)（架构论证、
> GPU 事实考证、Phase 2/3 规划）；外部事实来源：
> [torch-tensor-research.md](../research/torch-tensor-research.md)。
> 本文档是 **Phase 1 MVP** 的可构建规格，不含 Phase 2/3。

## 已拍板决策（2026-07-25）

| 决策点 | 结论 |
|---|---|
| 范围 | 只做 Phase 1：写入接受 torch + 采样可选返回 torch。Phase 2 性能工作等 Phase 1 落地后实测再定 |
| 写入 API | **隐式自动转换**：writer 直接接受 `torch.Tensor`，用户零感知 |
| 采样 API | **client 构造参数**：`LocalClient(..., output_format='torch')`，一次设定全局生效 |
| 测试环境 | 开发/CI 有 torch（CPU）、无 GPU → CUDA 用例 `skipif(not torch.cuda.is_available())` |

## 目标

用户可以把 `torch.Tensor`（CPU 或 CUDA）当作 numpy 的直接替代品写入 reverb，
并可以选择让采样输出直接是 `torch.Tensor`。默认行为与今天完全一致（numpy in，
numpy out）。

## 功能需求

### FR1 写入路径：隐式自动转换

- `Client` / `LocalClient` / `ShmClient` 三者的 `writer` / `trajectory_writer` /
  `structured_writer` 接受嵌套结构中任意 leaf 为 `torch.Tensor`。
- 转换语义（对每个 tensor leaf，在写入入口处逐 leaf 执行）：
  - `requires_grad` → `detach()`；
  - CUDA tensor → `.cpu()`（一次 D2H 拷贝，物理不可避免，见设计文档 §2）；
  - CPU tensor → `.numpy()` **零拷贝**视图；
  - 非连续 tensor 不特殊处理：`.numpy()` 保留 strides，由现有
    `ascontiguousarray` 兜底（`tensor_proxy.cc:170`）；
  - 0-d tensor 保持 0-d，不提升维度（对齐 numpy 标量语义）。
- 不支持的 dtype **明确报错**（列出实际 dtype）：`bfloat16`、量化、稀疏 tensor。
  报错发生在写入调用处，不允许静默截断或转换。
- 转换后走现有 numpy 路径，C++ 层、proto、SHM、checkpoint **零改动**。

### FR2 采样路径：`output_format` 构造参数

- `_BaseClient` 及其子类构造函数接受 `output_format='numpy' | 'torch'`，
  默认 `'numpy'`。
- `'torch'` 时采样输出的每个数值 leaf 经 `torch.from_numpy`（**零拷贝**）返回；
  proto 中存在但 torch 不支持的 dtype（`uint16/32/64`、`string`）**自动 fallback
  保留 numpy leaf，不报错**。
- 非法 `output_format` 值在构造时即 `ValueError`。

### FR3 torch 为可选依赖

- 无 torch 环境：import reverb 及全部现有功能不受影响；只有真的传入
  `torch.Tensor` 或设置 `output_format='torch'` 时才懒导入 torch，此时若 torch
  缺失则报 `ImportError` 并提示 `pip install dm-reverb-numpy[torch]`。
- `pyproject.toml` 增加 `[torch]` extra。

## 非功能需求

- **性能**：torch-CPU 写入吞吐与 numpy 写入持平（零拷贝，仅每 leaf 一次类型
  判断）。本次不做 benchmark 门槛（属 Phase 2），但不允许引入每 leaf 的
  重复 `import torch` 开销（懒导入结果缓存）。
- **线程模型**：转换发生在调用线程（持 GIL），与现状 `FromNdArray` 一致，
  不改变 worker 线程零 GIL 约定（`tensor_proxy.h:39`）。
- **向后兼容**：默认路径（纯 numpy 输入、默认 `output_format`）行为逐位不变。

## 实现接缝（已核实，供 implement 参考）

| 改动 | 位置 |
|---|---|
| 新增 `reverb/torch_support.py`：懒导入缓存、`to_numpy_leaf`、`from_numpy_leaf`（含 dtype fallback） | 新文件 |
| 写入归一化挂点 | `client.py` `_BaseClient.writer`(:490) / `trajectory_writer`(:617) / `structured_writer`(:683) 入口对嵌套结构 tree.map；`structured_writer.py:372` spec 校验前归一化 `step_spec`；`trajectory_writer.py:248` `append` 入口 |
| 采样 `output_format` 分支 | `client.py:482` 附近组包处；构造参数沿 `_BaseClient`(:206) 及三子类传递 |
| `pyproject.toml` | extras |

## 验收标准

1. 三客户端写入同一批 `torch.Tensor`（CPU）与写入等价 numpy，读出逐位一致。
2. `structured_writer` 的 `step_spec` 直接接受 torch.Tensor  leaf。
3. `output_format='torch'`：数值 leaf 返回 `torch.Tensor` 且与默认 numpy 输出
   值一致；`string`/`uint16` leaf fallback 为 numpy。
4. `bfloat16` tensor 写入报明确错误；非法 `output_format` 构造期报错。
5. 无 torch 环境（可用 `import` 屏蔽模拟）：现有测试全过；torch 用例 skip。
6. 全量现有 Python/C++ 测试通过，无回归。

## 测试策略（有 torch 无 GPU）

- 单元：`to_numpy_leaf` / `from_numpy_leaf` 各分支（CPU、非连续、
  `requires_grad`、0-d、bfloat16 报错、numpy passthrough、dtype fallback）。
- 集成：验收标准 1–4。CUDA 专属路径（D2H）写好用例但
  `pytest.mark.skipif(not torch.cuda.is_available())`，本地有卡时手动验证。

## 非目标（Phase 1）

同设计文档 §7，另外明确：pinned memory 优化、benchmark、CUDA IPC、`output_format`
支持 `'torch'` 之外的值、采样侧自动 batch 组批 —— 均不在本期。
