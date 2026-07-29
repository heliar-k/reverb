# Tickets: torch.Tensor 支持 Phase 1（2026-07-25 完成）

> 归档自原 `tickets.md` 活动板。本组 ticket 全部结案。

基于 `docs/spec/torch-tensor-spec.md`(上游设计 `docs/design/torch-tensor-design.md`,外部考证
`docs/research/torch-tensor-research.md`)。核心架构结论:全链路收敛于 `TensorBuffer` 字节枢纽,
转换只做在两个边缘,C++/proto/SHM/checkpoint 零改动。两张 ticket 串行交付后,
双轴 code-review 追加一张修复。已拍板:隐式自动转换(写入)、client 构造参数(采样)、
torch 为可选依赖(懒导入 + `[torch]` extra)。

## 写入路径接受 torch.Tensor

**What to build**:新增 `reverb/torch_support.py`(懒导入缓存、`to_numpy_leaf`:
detach→CUDA `.cpu()`→`.numpy()` 零拷贝,bfloat16 等无 numpy 对应 dtype 明确
ValueError、`to_numpy_tree`);四个写入入口接入归一化:`TrajectoryWriter.append`、
`StructuredWriter.append`、legacy `Writer.append`、`infer_signature` 的 step_spec。

**Blocked by**:None。

- [x] 三 writer + legacy writer 写入 torch.Tensor,读出与等价 numpy 逐位一致
- [x] `create_reference_step`/`infer_signature` 接受 torch step_spec
- [x] bfloat16 写入报 `ValueError`(含实际 dtype);0-d 保持 0-d;非连续由
  C++ `ascontiguousarray` 兜底(与 numpy 同语义)
- [x] 单元 10 例 + 集成 7 例;无 torch 环境(bazel)全量回归通过

## 采样侧 `output_format='torch'`

**What to build**:`_BaseClient` 构造参数 `output_format='numpy'|'torch'`(非法值
构造期 ValueError),`Client`/`LocalClient`/`ShmClient` 透传,`Server(in_process=True)`
经 `output_format` 参数传给 `LocalClient`;`sample()` 两分支逐 leaf
`torch.from_numpy` 零拷贝,torch 不支持的 dtype 自动 fallback 保留 numpy;
`pyproject.toml` 增加 `[torch]` extra。

**Blocked by**:写入路径(`from_numpy_leaf` 同属 torch_support 模块)。

- [x] `output_format='torch'` 时两条采样路径(emit_timesteps 开/关)leaf 为
  torch.Tensor,值与默认 numpy 输出一致
- [x] string(S2)leaf fallback 为 numpy 不报错
- [x] 无 torch 环境设 `output_format='torch'` 采样报 ImportError 并提示安装
  extra(FR3,仅 bazel 环境可执行的反向用例)
- [x] 默认 numpy 行为不变(向后兼容)

## code-review 追加修复

**What to build**:双轴评审(Standards + Spec)发现项修复。

**Blocked by**:上面两张完成。

- [x] `Writer.append_sequence` 接入归一化(评审发现的真 bug:此前 bfloat16/CUDA
  经此路径报隐晦 TypeError,违反 FR1 明确报错)
- [x] `Server.__init__` 构造期校验 `output_format`(原 `in_process=False` 路径
  静默吞掉非法值,违反 FR2);附 ponytail 标注
- [x] gRPC `Client`/`ShmClient` torch 写入集成测试(验收标准 1 的三客户端覆盖)
- [x] ruff SIM117 修复(嵌套 with 合并);`_BaseClient` docstring 补
  `output_format`;两个 append 的 Raises 段补 torch dtype ValueError

> **实现说明(2026-07-25)**:commits `d92913b`(写入)、`0e172b6`(采样)、
> `7841b43`(评审修复)。两个调研发现改变了实现重心:① C++ 边界的
> `np.asarray` 经 `__array__` 协议本就能吃 CPU torch tensor,Python 挂点的真正
> 价值在 CUDA D2H、bfloat16 明确报错、`infer_signature` dtype 读取三处;
> ② torch 2.13 起 `from_numpy` 支持 uint16/32/64,spec 中"uint fallback"实际
> 不发生,测试钉住实际行为,string 是唯一稳定 fallback。
> 测试基建:bazel hermetic py3.11 无 torch,torch 用例经
> `runfiles + /tmp/venv311(torch cpu)` 在系统外执行,bazel 侧全部 skip
> (恰好构成 FR3 无 torch 验证)。遗留:`server.py:20` I001 为存量 lint
> (`ruff check --fix` 可修,与本期无关);CUDA 端到端用例 skipif 待有卡环境;
> Phase 2(pinned memory + benchmark)待 profile 驱动。
