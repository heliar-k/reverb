# Tickets: 消除三客户端传输层代码冗余

基于 `docs/client-transports.md` 文档审计和全库扫描，将 Python/C++/pybind 三层中发现的高价值重复代码合并，降低维护面。

工作方式：从**前沿**（所有阻塞已完成的 ticket）开始，四张 ticket 无相互依赖，可并行推进。

---

## 合并 Python 层客户端 API 三份副本

**What to build**：`client.py` 中 `Client`/`LocalClient`/`ShmClient` 三个类的 `trajectory_writer` 和 `structured_writer` 方法合并为 `_BaseClient` 上的公共实现，差异部分（`max_chunk_length`、`validate_items`）参数化。同时移除 `_BaseClient.writer` 中与底层 C++ `Validate()` 重复的参数校验。

**Blocked by**：None — can start immediately。

- [x] `trajectory_writer` 三份实现合并为 `_BaseClient` 上的一个方法，`max_chunk_length` 和 `validate_items` 通过子类属性或参数控制
- [x] `structured_writer` 三份实现同样合并
- [x] `_BaseClient.writer` 中冗余的参数范围校验移除，信任 C++ 层 `options.Validate()` （经核查非冗余：legacy `Writer` 路径无 `options.Validate()`，gRPC `Client::NewWriter` 零校验，`InProcessClient::NewWriter` 仅校验三个 `<1` 不含 `chunk_length>max`；移除会将 ValueError 退化为 C++ CHECK abort 或静默构造损坏的 writer，故保留并加注，见 client.py `_BaseClient.writer` docstring）
- [x] 全量 Python 测试通过，三种客户端 `trajectory_writer`/`structured_writer`/`writer` 行为不变

---

## 提取 SHM 写入-读取-确认公共循环

**What to build**：`shm_client.cc` 中 `MutatePriorities`/`Reset`/`Checkpoint`/`ServerInfo` 四个方法共享的「序列化请求 → 加锁 → 写 ring → 读 ring → 解析响应 → 错误映射」模式，提取为模板方法 `SendInsertFlowRequest<Req, Resp>`，消掉约 150 行复制粘贴。

**Blocked by**：None — can start immediately。

- [x] 模板方法 `SendInsertFlowRequest` 覆盖四个方法的公共流程，请求/响应类型参数化
- [x] `MutatePriorities`/`Reset`/`Checkpoint`/`ServerInfo` 各自改为调用该模板
- [x] 错误映射（`ShmError::NOT_FOUND` → `NotFoundError` 等）内聚在模板中，四处行为一致
- [x] 全量 C++ 构建和 SHM 相关测试通过

---

## 消除 C++ 层 NewStructuredWriter 和 signature 填充三份副本

**What to build**：`client.cc`/`in_process_client.cc`/`shm_client.cc` 三份几乎逐字相同的 `NewStructuredWriter` 实现提取为单一自由函数；三份相同模式的 signature 填充逻辑提取为模板函数，各客户端通过钩子提供 `TableInfo` 来源。

**Blocked by**：None — can start immediately。

- [x] `MakeStructuredWriter` 自由函数替代三份 `NewStructuredWriter` 重复实现
- [ ] `PopulateFlatSignatureMap` 模板函数消除三份 signature 填充副本，客户端提供 signature 获取钩子
- [ ] `client.h`/`in_process_client.h`/`shm_client.h` 中移除已不再需要的重复声明
- [ ] 全量 C++ 构建和传输层测试通过

> **实现说明（2026-07）**：`MakeStructuredWriter` 已提取为 `structured_writer.h/.cc` 中的自由函数，三端 `NewStructuredWriter` 收敛为传入各自 `NewTrajectoryWriter` 钩子的薄封装。
>
> `PopulateFlatSignatureMap` 刻意不提取——经核查三端 signature 填充路径为「概念相似但实现不同」而非复制粘贴：gRPC 走 `MaybeUpdateServerInfoCache` 从缓存 RPC 取 `RepeatedPtrField<TableInfo>` 调 `FlatSignatureFromTableInfo`；in-process 遍历本地 `flat_hash_map<string,shared_ptr<Table>>` 调 `FlatSignatureFromSignatureProto` 且有 `has_value()` 条件分支；SHM 从 live `SERVER_INFO` ring 取 `std::vector<TableInfo>` 调 `FlatSignatureFromTableInfo`。容器类型、助手函数、控制流均不同，共享的仅「循环 + 赋给 `signatures[name]`」约三行。模板化需双钩子（元素转换 + 源获取）换三行收益，净增复杂度，故判为伪需求。
>
> 头声明无冗余可删：`NewStructuredWriter` 仍是三端的公开 API 入口（pybind/tests/Python 直接调用），提取的是函数体而非方法本身；仅同步修正了 `in_process_client.h`/`shm_client.h` 中描述内部实现的注释指向 `MakeStructuredWriter`。

---

## 消除 pybind 层重复序列化逻辑

**What to build**：`pybind.cc` 中重复三次的 `TableInfo` → `py::bytes` 序列化循环、重复三次的 `pair<uint64,double>` → `KeyWithPriority` proto 转换、重复三次的 `Sampler::Options` 构建，各提取为一个共享辅助函数。

**Blocked by**：None — can start immediately。

- [x] `SerializeTableInfoToPyBytes` 辅助函数消除三处重复序列化
- [x] `UpdatesToKeyWithPriorityProtos` 辅助函数消除三处重复 proto 转换
- [x] `BuildSamplerOptions` 辅助函数消除三处重复构造
- [x] 全量 pybind 构建和 Python API 测试通过，所有客户端方法行为不变

---

# Tickets: SHM 并发正确性（2026-07 分支扫描）

来源：对 `feat/numpy-embed` 全分支潜在 bug 扫描（三个并行审计代理 + 人工抽查验证）。
同批发现中已修复的不在此列：`fix(numpy)` bytes 序列化、`fix(shm)` pool offset 校验、
`fix(shm)` 池耗尽死锁（见对应 commit）。以下两张为协议级改动，单独推进。

---

## SHM 单连接多 sampler 竞态（扫描 #1）

**What to build**：修复同一 `ShmClient` 连接上并发多个 sampler 的 SPSC 破坏与响应错配。
现状：`ShmClient::NewSampler`（shm_client.cc:480）不限制 sampler 数量，每个 `ShmSampler`
的 worker 都在同一对 `sample_c2s`/`sample_s2c` ring 上生产/消费 —— 双生产者无 CAS 写
`head`（槽交错损坏）；且 `ShmError.request_seq`（shm_protocol.proto:161）从未赋值，
服务端异步完成 sample 乱序时 sampler A 可消费 B 的 `SAMPLE_RESP`，**静默返回错误表的
数据**。复现：`client.sample("a")` 与 `client.sample("b")` 并发（或同一表两个迭代器）。

分两期：
1. **近期（本 ticket 必须）**：`NewSampler` 在同一连接已有活 sampler 时返回
   `FailedPreconditionError`，把 SPSC 不变式从“隐式约定”变成强制约束；文档说明
   多表采样用多连接或串行迭代。
2. **远期（可选，另议）**：`request_seq` 贯穿 `ShmSampleRequest`/`SAMPLE_RESP`/`ShmError`，
   响应按 seq 路由到对应 sampler，恢复多 sampler 能力；`sample_c2s` 改 MPMC 或每
   sampler 独立 ring 对。

**Blocked by**：None — can start immediately。

- [x] `NewSampler` 拒绝同一连接的第二个活 sampler（`FailedPrecondition`），sampler 关闭后释放名额
- [x] 并发 `sample("a")`/`sample("b")` 回归测试：第二个 sampler 得到明确错误而非静默错数据（shm_sample_test `SecondConcurrentSamplerRejected`）
- [x] Python 层 `ShmClient.sample()` 并发调用行为文档化（client.py `sample()` docstring）
- [x] （远期）`request_seq` 路由方案评估并记录结论（做/不做）

> **远期 request_seq 评估结论（2026-07-25）：暂不做。** 恢复多 sampler 需
> `request_seq` 贯穿请求/响应 + 按 seq 路由 + `sample_c2s` 改 MPMC 或每 sampler
> 独立 ring 对 —— 协议面改动大，而单连接单 sampler + 多连接已覆盖并发采样需求。
> 若未来单连接多路采样成为真实瓶颈，按 ticket 中的远期方案重开。

> **实现说明**：permit 为 `ShmClient::sampler_active_` 原子布尔，NewSampler
> `exchange` 认领、`ShmSampler::Close` 释放（析构经 Close 幂等）。直接
> `ShmSampler::Create`（测试）不受限——约束在客户端 seam 而非 sampler 类。

---

## Ring 多槽消息发布竞态（扫描 #4）

**What to build**：修复 `Ring::WriteSlots`（ring.cc:250）多槽消息的发布时序。现状：
按槽序 0..n-1 逐个 release-store `seq`，消费者可 acquire 到槽 k 的新 `seq` 而槽 k+1
的 store 还在生产者 store buffer —— `Read` 返回 `InternalError("ring continuation
slot missing")`，而这是**正常竞态**非数据损坏。服务端下一圈自愈，但客户端
`ReadBlocking`（shm_client.cc:74）把非 NotFound 错误当致命错误传播，直接杀掉 writer
流/sampler。触发：任何 >240B 消息（INSERT body、多列 SAMPLE_RESP）在负载下。

方向（任选其一，倾向 a）：
a. **批量发布**：先写全部数据槽（relaxed），最后单次 release-store 首槽 `seq` 作为
   “整条消息就绪”信号；消费者缺续槽时视为 `NOT_READY` 下轮重试。
b. 消费者侧容忍：`Read` 遇缺续槽返回 NotFound(NOT_READY) 而非 InternalError —— 最小
   改动但留下“部分发布被消费者目击”的语义，需确认生产者不会长期停在半条消息。

**Blocked by**：None — can start immediately。与上一张 ticket 无依赖，可并行。

- [x] 多槽消息在并发读写下不再产生 "continuation slot missing" 致命错误
- [x] ring_test 新增多槽消息并发压力用例（大消息 + 高频读写，断言无 InternalError，`MultiSlotConcurrentReadNeverSeesPartialMessage`——修复前稳定复现 red）
- [x] 客户端长消息（大 INSERT / 多列 SAMPLE_RESP）在负载下不再被杀流，shm_insert_test/shm_sample_test 回归通过

> **方案评估结论（2026-07-25）：采纳 a 的精化版——槽 0 最后发布。** 续槽照旧
> 逐个 release，仅把槽 0 的 seq store 挪到循环末尾；消费者对槽 0 的 acquire 与
> 该 release 构成 synchronizes-with，整条消息（含续槽 seq）对其原子可见，
> 消费者代码零改动。方案 b（容忍 NOT_READY）会把“半条消息被目击”常态
> 化、稀释真损坏信号，否。崩溃语义反而变强：槽 0 未发布 = 整条未就绪，
> "continuation slot missing" 从此只指示真损坏。

---

# Tickets: torch.Tensor 支持 Phase 1(2026-07-25 完成)

基于 `docs/torch-tensor-spec.md`(上游设计 `docs/torch-tensor-design.md`,外部考证
`docs/torch-tensor-research.md`)。核心架构结论:全链路收敛于 `TensorBuffer` 字节枢纽,
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

---

# Tickets: numpy↔字节流零拷贝(Tier 1+2,2026-07-26 完成)

调研结论(见上一 session 报告):numpy↔`TensorBuffer` 两边缘各有一次强制
memcpy(`tensor_proxy.cc` FromNdArray/ToNdArray),torch↔numpy 已零拷贝。
本期按分层方案的 Tier 1+2 落地,TDD 推进,三 slice 串行。

## Slice 1:采样侧 ToNdArray 零拷贝

**What to build**:`TensorBuffer` 存储从 `std::string bytes_` 改为
`shared_ptr<void> owner_ + string_view bytes_`(公共构造签名不变,链路零改动);
`ToNdArray` 用 `PyArray_SimpleNewFromData` + capsule 持 `owner_` 副本,
返回数组独立于源 TensorBuffer 存活。维度操作(InsertBatchDim 等)顺带共享
宿主,消掉原有的隐式 string 拷贝。

- [x] C++ `ToNdArraySharesStorage`(指针相等)/`ToNdArrayOutlivesSource`
  /`FinishMimicSharedStorage`(Writer::Finish 对象图回归)
- [x] Python seam(in_process_test):采样数组 `owndata == False`;
  del client+gc 后值不变
- [x] 现有 RoundTrip 全套 + in_process 44 例绿(值语义不变)

## Slice 2:写入侧 FromNdArray 零拷贝(opt-in)

**What to build**:`FromNdArray(zero_copy=true)` 数值 dtype 走视图:
`Py_INCREF` 持 numpy + 裸指针;任意线程析构将 PyObject* 入
DeferredFreeQueue,主线程在 FromNdArray/ToNdArray 入口(持 GIL)统一
DECREF。开启方式:环境变量 `REVERB_ZERO_COPY_APPEND=1`(type_caster 首次
Append 读一次)。**快照语义丢失**(append 后原地复用 buffer 会写脏数据),
故 opt-in 且开启时源数组置 read-only,让原地改写立刻 `ValueError` 而非静默
写脏(best-effort:torch 侧或其他 view 写入绕过该 flag,文档已注明)。

- [x] C++ 四例:指针相等+guard 置位/默认快照语义钉住/源释放后 buffer 存活
  /无 GIL 析构入队、主线程 drain 后 refcount 回落
- [x] Python seam(新 target `zero_copy_write_test`,bazel env 属性置 1):
  read-only 置位、append 后原地改写抛 ValueError、roundtrip 值一致
- [x] 默认路径行为不变(全部存量回归绿)

## Slice 3:SHM 插入直写 pool

**What to build**:`RunShmWorker` 的 `SerializeToString`+memcpy 两跳改为
`ByteSizeLong` 预算→ALLOCATE→`SerializeToArray(pool.At(offset))` 一跳;
序列化失败(理论不可达)走统一 alloc_failed 释放路径不泄漏 offset。
无行为变化,不新写测试:shm_insert/shm_sample/shm_test 回归即保护网。

- [x] 三个 SHM 测试目标绿

> **实现说明(2026-07-26)**:诊断插曲——stash/pop + 被杀的构建留下不一致
> .so 导致一次假 segfault(ABI 混合),全量重编后消失;教训:ABI 敏感改动
> 的诊断前先全量重编。torch 侧零拷贝随本期自动成立:采样 `from_numpy`
> 本即零拷贝(view 链 torch→numpy→capsule→owner);写入 CPU tensor 经
> `.numpy()` 视图接上 zero_copy 路径,无需额外改动;CUDA 的 D2H 物理不可免。
> 遗留:Concat/`set_tensor_content` 的物化拷贝属 wire format 强制(Tier 3,
> 协议层,待 profile);snappy 输出 buffer 算法固有;DeferredFreeQueue 进程
> 退出时残留随进程回收(ponytail 标注)。
>
> **code-review(双轴,2026-07-26)**:Standards 0 硬违规,唯一行动项
> RunShmWorker 失败模板重复→已提取 `fail_alloc` lambda(7 处收敛);文档级
> 重复以 tensor_proxy.h 类注释为权威版。Spec 验收全覆盖,两处修复:
> SubSlice 数值路径视图化(去物化,Slice 1 意图补齐);ToNdArray 空数组
> 分配失败回退 `py::none()`(修本期引入的小回归)。int 截断注记同 proto
> API 上限,不处理。修复后 14 个受影响测试目标全绿。
