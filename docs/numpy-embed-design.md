# Reverb 纯 numpy 内嵌模式设计文档

> 本文档阐述 Reverb 从"依赖 TensorFlow 的 C++/gRPC 库"改造为"纯 numpy 内嵌库"
> 的设计方案、动机、与原版的差异，以及不得不做的变更。面向想要理解或贡献本 fork
> 的开发者。

## 目录

- [1. 为什么这么设计](#1-为什么这么设计)
  - [1.1 痛点：TensorFlow 是单机内嵌场景的负担](#11-痛点tensorflow-是单机内嵌场景的负担)
  - [1.2 目标：保留 C++ 核心，去掉 TF 依赖，支持进程内直连](#12-目标保留-c-核心去掉-tf-依赖支持进程内直连)
  - [1.3 设计决策汇总](#13-设计决策汇总)
- [2. 与原有版本的差异](#2-与原有版本的差异)
  - [2.1 数据载体：`tf::Tensor` → `TensorBuffer`](#21-数据载体tftensor--tensorbuffer)
  - [2.2 客户端：只有 gRPC `Client` → 新增 `InProcessClient` / `LocalClient`](#22-客户端只有-grpc-client--新增-inprocessclient--localclient)
  - [2.3 Signature 编解码：TF `nested_structure_coder` → 纯 Python `signature_codec`](#23-signature-编解码tf-nested_structure_coder--纯-python-signature_codec)
  - [2.4 Checkpoint：`TFRecord` → length-delimited protobuf](#24-checkpointtfrecord--length-delimited-protobuf)
  - [2.5 删除的 TF 集成层](#25-删除的-tf-集成层)
  - [2.6 错误类型：`TimeoutError` → `DeadlineExceededError`](#26-错误类型timeouterror--deadlineexceedederror)
- [3. 不得不做的变更](#3-不得不做的变更)
  - [3.1 fork 精简 proto 到 `third_party/`](#31-fork-精简-proto-到-third_party)
  - [3.2 `TrajectoryWriter` 加本地路径分支](#32-trajectorywriter-加本地路径分支)
  - [3.3 pybind PascalCase 双名别名](#33-pybind-pascalcase-双名别名)
  - [3.4 `LocalClient.sample` 默认 `emit_timesteps=True`](#34-localclientsample-默认-emit_timestepstrue)
  - [3.5 `LocalClient` 无 pickle（`insert`/`writer` 已对齐）](#35-localclient-无-pickleinsertwriter-已对齐)
  - [3.6 旧 checkpoint 不兼容](#36-旧-checkpoint-不兼容)
- [4. 使用样例对比](#4-使用样例对比)
  - [4.1 创建 Server 与 Client](#41-创建-server-与-client)
  - [4.2 写入数据](#42-写入数据)
  - [4.3 采样数据](#43-采样数据)
  - [4.4 StructuredWriter](#44-structuredwriter)
  - [4.5 Checkpoint 保存与恢复](#45-checkpoint-保存与恢复)
- [5. 架构总览](#5-架构总览)
- [6. 附录：本地 Writer 解绑与本地化（D1/D2/D3-a 摘录）](#6-附录本地-writer-解绑与本地化d1d2d3-a-摘录)
  - [6.1 问题溯源](#61-问题溯源)
  - [6.2 决策点](#62-决策点)
  - [6.3 修正目标与交付边界](#63-修正目标与交付边界)
  - [6.4 writer 持 map 的生命周期方式：持拷贝（方案 P）](#64-writer-持-map-的生命周期方式持拷贝方案-p)
  - [6.5 ADR-0001：推翻 A1 砍 Writer 决策](#65-adr-0001推翻-a1-砍-writer-决策)

---

## 1. 为什么这么设计

### 1.1 痛点：TensorFlow 是单机内嵌场景的负担

原版 Reverb 是一个依赖 TensorFlow 的 C++/gRPC 库，专为**分布式**强化学习设计：
训练进程通过 gRPC 与独立的 Reverb server 进程交换数据，数据载体是 `tf::Tensor`，
signature 用 TF 的 `nested_structure_coder` 编解码，checkpoint 用 TFRecord 格式，
还提供 `tf.data.Dataset` 风格的 `ReplayDataset` 供训练循环消费。

这套设计在分布式场景下很合理，但**单机内嵌训练**场景下存在三个突出问题：

1. **TF 依赖过重**：只为用 `tf::Tensor` 做数据载体，就得拉入整个 TensorFlow
   C++ 库。首次 `bazel build` 会下载 ~453MB 的 `@org_tensorflow` 源码树，构建
   缓慢，部署体积大。对于"只想用 Reverb 做经验回放、数据本来就是 numpy"的
   用户，这是不必要的负担。

2. **无进程内直连**：原版唯一的本地交互方式是 `Client('localhost:port')`，
   即便 server 和 client 在同一进程，数据也得经过 gRPC 序列化→网络栈→反序列化
   的完整往返。零开销内嵌使用无从谈起。

3. **TF 与 Reverb 的 gRPC 冲突**：TF 自带 gRPC，Reverb 也静态链接 gRPC。
   在同一进程里 `import tensorflow` 会触发重复 flag 注册冲突。原版通过
   `tf_client.py` 的 op 机制规避，但纯 numpy 用户根本不想碰 TF。

### 1.2 目标：保留 C++ 核心，去掉 TF 依赖，支持进程内直连

本 fork 的目标很明确：

- **保留** Reverb 的 C++ 算法核心：`Table` / `ItemSelector`（6 种采样/淘汰策略）
  / `RateLimiter` / `chunker` / 异步 worker 线程（零 GIL）。这些是性能基础，
  不该重写。
- **去掉** TensorFlow 依赖：数据载体从 `tf::Tensor` 换成自研的 `TensorBuffer`
  （numpy bytes + spec），proto 自研 fork，checkpoint 换格式，删掉所有 TF 集成层。
- **新增** 进程内直连：`InProcessClient` 直接持有 `shared_ptr<Table>`，
  `TrajectoryWriter` 加本地路径分支，绕过 gRPC，零网络开销。
- **保留** gRPC 模式可用：`Server(in_process=False)` 仍是完整的分布式 server，
  `Client`/`Writer`/`StructuredWriter` 走 gRPC 路径（也已去 TF，数据流为 numpy）。

### 1.3 设计决策汇总

| 编号 | 决策 | 理由 |
| ------ | ------ | ------ |
| A1 | 内嵌只做 `TrajectoryWriter` + `StructuredWriter`(旧 `Writer`/`StreamingTrajectoryWriter` 本地路径砍掉) | 这两个是现代推荐 API；旧 writer 本地路径价值低，砍掉省复杂度。**`Writer` 部分被 [ADR-0001](adr/0001-embedded-writer-local-path.md) 推翻**(`StreamingTrajectoryWriter` 部分保留) |
| A2 | 保留 gRPC 层 | 一套 C++ 代码两种模式（内嵌 + 分布式），不为内嵌牺牲分布式能力 |
| A3 | 彻底去 TF，fork 精简 proto 到 `third_party/` | TF proto 链式拉入一堆传递依赖，精简 fork 切断 `@org_tensorflow` |
| B1 | `TensorBuffer` 用拷贝 bytes（`std::string`） | worker 线程零 GIL，简单正确；预留零拷贝升级路径（`ponytail:` 标注） |
| B2 | 自研 `TensorSpec` + `TensorBuffer` | 不依赖 TF，对齐 numpy dtype |
| C1 | pybind 暴露 `timeout` 参数，超时抛 `DeadlineExceededError` | 原版 `TimeoutError` 与 Python 内置同名，易混淆 |
| C2 | 保留 `table_worker_` + `extension_worker_` 异步线程 | 零 GIL 是性能基础，不能丢 |
| D1 | 本地 `TrajectoryWriter`/`Writer` 持 `tables_` map，按 `item.table()` 分发 | 修正“绑定单一 table”的历史偏懒选择，签名对齐 gRPC `Client`(不收 table 参数)，消除三层 API 债。详见 [unbind-local-writer-plan.md](unbind-local-writer-plan.md) |
| D2 | writer 级 backpressure(任意表满则 writer 停) | 对齐 gRPC writer 级单一 stream 的语义；单表场景行为不变 |
| D3 | checkpoint 用 length-delimited protobuf | TFRecord 去掉 CRC32 就是标准 length-delimited protobuf，~20 行实现。**注**：与下方 §3.5 的 D3-a(Writer 本地化) 不同编号语境，此处 D3 为 checkpoint 决策 |

---

## 2. 与原有版本的差异

### 2.1 数据载体：`tf::Tensor` → `TensorBuffer`

原版 C++ 全栈用 `tensorflow::Tensor` 承载数据。chunker 的 `buffer_`、sampler 的
输出、table 的 chunk 存储都是 `Tensor`。所有 `Concat`/`SubSlice`/`CopyFrom` 等
操作都调 TF API。

本 fork 引入 `TensorBuffer`（`reverb/cc/support/tensor_proxy.h`）：

```cpp
struct TensorSpec {
  DataType dtype;              // 自研枚举，对齐 numpy dtype
  std::vector<int64_t> shape;
};

class TensorBuffer {
 public:
  static absl::StatusOr<TensorBuffer> FromNdArray(py::object ndarray);
  py::object ToNdArray() const;
  absl::string_view bytes() const;   // worker 线程零 GIL 安全
  static absl::StatusOr<TensorBuffer> Concat(...);
  TensorBuffer SubSlice(int64_t offset) const;
  // ... InsertBatchDim / RemoveBatchDim / CopyReshaped / SerializeToProto ...
};
```

`DataType` 枚举**对齐 numpy 而非 TF**（避免概念残留）：

```cpp
enum class DataType : uint8_t {
  Invalid = 0, Float32, Float64,
  Int8, Int16, Int32, Int64,
  Uint8, Uint16, Uint32, Uint64,
  Bool, Complex64, Complex128, String,
};
```

`FromNdArray` 强制 C-contiguous（`py::array::c_style`），`memcpy` 到 `std::string`。
worker 线程只读 `bytes()`，不持 GIL。pybind11 的 `type_caster<TensorBuffer>`
实现 ndarray ↔ TensorBuffer 自动互转，Python 侧完全无感。

> **拷贝 vs 零拷贝**：当前用拷贝 bytes 语义（简单、零 GIL）。若 profile 显示写入
> memcpy 成为瓶颈，升级路径是裸指针 + `DeferredFreeQueue`（`FromNdArray` 时
> `Py_INCREF` 持 `PyObject*`，worker 读裸指针，释放入队主线程统一 `Py_DECREF`）。
> 代码里用 `ponytail:` 注释标注了这个升级点。

### 2.2 客户端：只有 gRPC `Client` → 新增 `InProcessClient` / `LocalClient`

原版只有一个 `Client`，通过 gRPC 连 server。本 fork 在 C++ 层新增 `InProcessClient`
（`reverb/cc/in_process_client.h`），直接持有
`vector<shared_ptr<Table>>`，所有方法零网络：

```cpp
class InProcessClient {
 public:
  explicit InProcessClient(std::vector<std::shared_ptr<Table>> tables,
                           std::shared_ptr<Checkpointer> checkpointer = nullptr);
  // 不收 table 参数，writer 持 tables_ map 按 item.table() 分发（D1）
  absl::Status NewTrajectoryWriter(const TrajectoryWriter::Options& options,
                                  std::unique_ptr<TrajectoryWriter>* writer);
  absl::Status NewStructuredWriter(std::vector<StructuredWriterConfig> configs,
                                  std::unique_ptr<StructuredWriter>* writer);
  absl::Status NewWriter(int chunk_length, int max_timesteps, bool delta_encoded,
                         int max_in_flight_items, std::unique_ptr<Writer>* writer);
  absl::Status NewSampler(const std::string& table_name, ...);
  absl::Status MutatePriorities(...);
  absl::Status Reset(...);
  absl::Status Checkpoint(std::string* path);
  absl::Status LoadLatest();
  absl::Status Load(absl::string_view path);
  absl::Status ServerInfo(std::vector<TableInfo>* table_info);
};
```

Python 层有两个客户端类，共享 `_BaseClient` 的 `sample`/`insert`/`writer`/
`mutate_priorities`/`reset`/`server_info`/`checkpoint` 逻辑：

- **`Client`**（gRPC）：构造为 `Client('localhost:port')`，走 gRPC。保留原版的
  `insert`/`writer`/`trajectory_writer`/`structured_writer` 全部工厂方法。
- **`LocalClient`**（内嵌）：由 `Server(in_process=True).in_process_client` 返回，
  包装 C++ `InProcessClient`。与 `Client` API 严格镜像——`insert`/`writer` 上提到
  `_BaseClient` 共享单一实现，`trajectory_writer`/`structured_writer` 不收 table 参数
  （D1）。唯一缺失是 pickle（见 [3.5](#35-localclient-无-pickleinsertwriter-已对齐)）。

两者通过两个 hook 区分 C++ 调用差异：`_fetch_server_info_proto`（gRPC 传 timeout，
内嵌忽略）和 `_new_sampler`（gRPC 无 rate-limiter timeout，内嵌有）。

### 2.3 Signature 编解码：TF `nested_structure_coder` → 纯 Python `signature_codec`

原版用 TF 的 `nested_structure_coder` 把 `tf.TypeSpec`（`TensorSpec`/
`BoundedTensorSpec`）编码成 `StructuredValue` proto。这强制依赖 TF。

本 fork 写了纯 Python 的 `signature_codec`（`reverb/signature_codec.py`）：

```python
class TensorSpec:
  def __init__(self, shape, dtype, name=None): ...

def encode_signature(structure: Any) -> bytes: ...
def decode_signature(data: bytes) -> Any: ...
```

`Table` 构造时 signature 的叶子必须是 `signature_codec.TensorSpec`（不是 TF 的）。
编码后传给 C++ `Table` 的是序列化后的 `SignatureProto` 字符串。C++ 侧只透传、
不解析（signature 校验在 Python `TrajectoryWriter` 里做，或本地路径跳过）。

### 2.4 Checkpoint：`TFRecord` → length-delimited protobuf

原版 `TfRecordCheckpointer` 用 `tensorflow::io::RecordWriter`。本 fork 用
`SimpleCheckpointer`（`reverb/cc/platform/default/simple_checkpointer.cc`），
基于自研的 length-delimited protobuf IO（`reverb/cc/support/length_delimited_io.cc`）。

TFRecord 本质 = varint 长度 + 数据 + CRC32。去掉 CRC32 就是标准 length-delimited
protobuf，~20 行实现，流式、无 2GB 限制。一个 checkpoint 目录含：
`tables.ckpt` / `items.ckpt` / `chunks.ckpt` / `DONE`。

### 2.5 删除的 TF 集成层

以下原版组件被**整体删除**：

| 删除内容 | 原因 |
| --------- | ------ |
| `reverb/cc/ops/*` | `tf.data` dataset op，纯 TF 集成层 |
| `reverb/cc/platform/tfrecord_checkpointer.*` | 用 `SimpleCheckpointer` 替代 |
| `reverb/tf_client.py`（`TFClient`/`ReplayDataset`） | TF 集成层，内嵌模式不需要 |
| `reverb/distributions.py` | TF 分布相关，已不需要 |
| `@org_tensorflow` / `@local_xla` / `tf_workspace*` | WORKSPACE 彻底移除，首次构建不再下载 453MB |

`__init__.py` 不再导出 `TFClient`/`ReplayDataset`/`PriorityTable`（后者是 `Table`
的旧名，已统一为 `Table`）。

### 2.6 错误类型：`TimeoutError` → `DeadlineExceededError`

原版 `errors.TimeoutError` 与 Python 内置 `TimeoutError` 同名，且不继承它，容易
混淆。本 fork 改为 `DeadlineExceededError(ReverbError)`，并通过 pybind 的
`MaybeRaiseFromStatus` 把 C++ 的 `absl::StatusCode::kDeadlineExceeded` 映射成它。

```python
class ReverbError(Exception): ...
class DeadlineExceededError(ReverbError): ...
```

C++ 侧 `Sampler::Options.rate_limiter_timeout` 和 `TrajectoryWriter::Flush`/
`EndEpisode` 的 timeout 在超时时返回 `kDeadlineExceeded`，pybind 统一抛
`DeadlineExceededError`。

---

## 3. 不得不做的变更

这些变更不是"想做"，而是"去 TF + 加内嵌"的必然结果，有些带来了 API 不对称。

### 3.1 fork 精简 proto 到 `third_party/`

原版 `schema.proto`/`patterns.proto`/`checkpoint.proto` 直接 import TF 的 proto
（`TensorProto`/`StructuredValue`），TF proto 链式拉入 `resource_handle.proto`→
`device_attribute.proto` 等一大堆。

本 fork 在 `third_party/reverb_tensor/reverb_tensor.proto` 自定义了精简版：
`DataType` 枚举、`TensorShapeProto`、`TensorProto`、`SignatureProto`（只含 Reverb
用到的 6 种节点：TensorSpec/BoundedTensorSpec/List/Tuple/Dict/NamedTuple）。所有
原 import `@org_tensorflow` 的 proto 改为 import 这个 fork，彻底切断传递依赖。

### 3.2 `TrajectoryWriter` 加本地路径分支

原版 `TrajectoryWriter` 只走 gRPC：`RunStreamWorker` 把 chunk+item 打包进
`InsertStreamRequest` 发出。

本 fork 给 `TrajectoryWriter` 加了本地路径分支（`is_local_` 标志）：本地路径调
`Table::InsertOrAssignAsync`，用 callback 替代 gRPC 的 `OnReadDone`。

> **决策**：原计划新建独立的 `LocalTrajectoryWriter` 类，实际改为直接在
> `TrajectoryWriter` 内加分支——复用全部 chunker/column 逻辑，避免代码重复。
>
> **修正(D1)**：早期实现让本地 writer 构造时绑定**单一** `table_`，逼出三层 API 债
> （`LocalClient.trajectory_writer(table=...)` 必须显式传 table、§3.4 的默认值偏差、
> §3.5 的 `insert`/`writer` 缺失）。现改为 writer 持 `tables_` map（拷贝，方案 P），
> worker 按 `item.table()` 查表分发，未知表报 `kNotFound`；`InProcessClient::
> NewTrajectoryWriter` 去掉 table 参数，完全对齐 gRPC `Client` 签名。backpressure
> 为 writer 级单一 flag（D2：任意表满则 writer 停，对齐 gRPC writer 级 stream），
> 单表场景行为不变。详见 [unbind-local-writer-plan.md](unbind-local-writer-plan.md)。

### 3.3 pybind PascalCase 双名别名

`InProcessClient` 的 pybind 绑定同时暴露了 `snake_case`（`new_sampler`/
`reset`/`checkpoint`/`server_info`）和 `PascalCase`（`NewSampler`/`Reset`/
`Checkpoint`/`ServerInfo`）两套方法名，指向同一 C++ 函数。

**原因**：`reverb/client.py` 的 `_BaseClient` 想在 gRPC `Client` 和内嵌
`LocalClient` 间共享一套调用代码。gRPC `Client` 绑定的是 PascalCase（历史遗留），
内嵌原生是 snake_case。为避免在 Python 层做 if-else 分发，给内嵌也绑了 PascalCase
别名，让两个 client 走同一路径。

```cpp
.def("new_sampler", new_sampler_fn, ...)
.def("NewSampler", new_sampler_fn, ...)   // 别名,与 gRPC Client 对齐
```

这是为了代码收敛而引入的少量 pybind 重复，权衡后可接受。

### 3.4 `LocalClient.sample` 默认 `emit_timesteps=True`

`sample(emit_timesteps=...)` 控制返回整条 trajectory 还是按 timestep 拆分。

早期内嵌 `LocalClient` 默认 `False`（内嵌场景几乎总是想要整条 trajectory），与
 gRPC `Client` 的 `True` 不一致。该偏差是“本地 writer 绑定单一 table”连带逼出的
 三层 API 债之一（见 §3.2 修正）。

**修正(D1)**：writer 解绑 + 补 `insert`/`writer` 后，两端 API 严格镜像，
`_default_emit_timesteps` 统一为 `True`（gRPC 的历史行为）。调用方可显式传
`emit_timesteps=False` 取整条 trajectory。

### 3.5 `LocalClient` 无 pickle（`insert`/`writer` 已对齐）

gRPC `Client` 有 `insert(data, priorities)`、`writer(max_sequence_length)`、
`__reduce__`（pickle 支持，因为 Client 只存 server 地址，可跨进程重建）。

早期 `LocalClient` 没有 `insert`/`writer`（受“绑定单一 table”连带影响，见 §3.2
修正），与 gRPC `Client` 不对称。

**修正(D1/D3-a)**：writer 解绑后，`insert`/`writer` 上提到 `_BaseClient`，
`LocalClient` 与 `Client` 共享单一实现（靠 `self.writer`/`self._client.NewWriter`
鸭子类型分派），签名与语义完全一致。`LocalClient.writer` 创建本地 `Writer`
（持 `tables_` map，按 `item.table()` 分发，`InsertCallback` 递减
`num_items_in_flight_` 复刻 gRPC `ConfirmItems` 语义）。

唯一保留的真实物理约束：**无 `__reduce__`（pickle）**——`LocalClient` 持进程内
Table 指针，不可跨进程序列化。这是物理约束，非 API 债。

### 3.6 旧 checkpoint 不兼容

数据载体从 `tf::Tensor` 换成 `TensorBuffer`，proto 从 TF 的 `TensorProto` 换成
自研 `reverb.tensor.TensorProto`，checkpoint 格式从 TFRecord 换成
length-delimited protobuf。**三重变化使旧 checkpoint 无法读取**，需重新生成。

---

## 4. 使用样例对比

### 4.1 创建 Server 与 Client

**原版（TF + gRPC，仅分布式）**：

```python
import reverb
server = reverb.Server(
    tables=[reverb.PriorityTable(name='t', ...)],
    port=8000)
client = reverb.Client('localhost:8000')   # gRPC,同进程也得走网络
```

**本 fork（内嵌模式，零网络）**：

```python
import reverb
server = reverb.Server(
    tables=[reverb.Table(
        name='t',
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=1000,
        rate_limiter=reverb.rate_limiters.MinSize(100))],
    in_process=True)                # 关键:进程内直连,不起 gRPC
client = server.in_process_client   # LocalClient,零网络
```

**本 fork（gRPC 模式，仍可用）**：

```python
server = reverb.Server(tables=[...], port=8000, in_process=False)
client = reverb.Client(f'localhost:{server.port}')
```

> 说明：`in_process=True` 时不起 gRPC 服务、不分配端口（`server.port` 为 `None`），
> 只能通过 `server.in_process_client` 访问；`localhost_client()` 会抛错。
> `in_process=False` 时反之，`in_process_client` 抛错，用 `localhost_client()`。

### 4.2 写入数据

**原版（gRPC `Client.insert`）**：

```python
client.insert(some_trajectory, {'t': 1.0})
```

**本 fork 内嵌（`trajectory_writer`，推荐）**：

```python
with client.trajectory_writer(num_keep_alive_refs=10) as w:
    w.append({'obs': np.zeros(4, dtype=np.float32)})   # 纯 numpy
    w.create_item(
        table='t', priority=1.0,
        trajectory={'obs': w.history['obs'][:]})       # 引用最近数据
    w.flush()
```

> 说明：修正后（D1）本地 `trajectory_writer` 与 gRPC `Client.trajectory_writer`
> 签名一致，都不传 `table`——writer 持 client 全部表，每个 item 在 `create_item`
> 时按 `table` 参数路由。内嵌 `LocalClient` 也可用 `client.insert(...)`/
> `client.writer(...)`，与 gRPC 完全镜像（见 [3.5](#35-localclient-无-pickleinsertwriter-已对齐)）。
> `num_keep_alive_refs` 是循环缓冲区大小，即 trajectory 最大跨度。
> `w.history['col'][:]` 返回 `TrajectoryColumn`，`create_item` 的
> `trajectory` 是一个结构与期望采样结构一致嵌套 dict/list。

### 4.3 采样数据

**原版（返回 timestep 列表）**：

```python
for sequence in client.sample('t', num_samples=4):
    for step in sequence:          # 每个 sequence 是 timestep 列表
        print(step.data)
```

**本 fork 内嵌（传 `emit_timesteps=False` 取整条 trajectory）**：

```python
for sample in client.sample('t', num_samples=4, emit_timesteps=False):
    # sample 是单个 ReplaySample(传 emit_timesteps=False 取整条 trajectory)
    print(sample.info.key, sample.info.priority)
    print(np.asarray(sample.data[0]))   # flat list of columns
```

> 说明：`sample.data` 是 flat list（除非 `unpack_as_table_signature=True` 且表
> 声明了 signature，则按 signature 结构 unflatten 成 dict）。`sample.info` 是
> `SampleInfo(key, probability, table_size, priority, times_sampled)`。
> 带超时采样：`client.sample('t', num_samples=1, timeout_ms=500)`，超时抛
> `reverb.errors.DeadlineExceededError`。

### 4.4 StructuredWriter

StructuredWriter 用静态 pattern 把 append 流转成表插入，适合固定窗口采样。

**本 fork 内嵌**：

```python
from reverb import structured_writer

step_spec = {'a': np.zeros([], np.float32)}
ref_step = structured_writer.create_reference_step(step_spec)
pattern = {'x': ref_step['a'][-3:]}    # 最近 3 步
config = structured_writer.create_config(pattern=pattern, table='t')

writer = client.structured_writer(configs=[config])
for i in range(5):
    writer.append(np.asarray(float(i), dtype=np.float32))
writer.end_episode()

for sample in client.sample('t', num_samples=3, emit_timesteps=False):
    print(np.asarray(sample.data[0]))   # [0., 1., 2.], [1., 2., 3.], ...
```

> 说明：修正后（D1）内嵌 `structured_writer` 与 gRPC `Client.structured_writer`
> 签名一致，都不传 `table`——每个 config 的 `table` 字段指定目标表，writer 按
> `item.table()` 路由，支持多表写入。

### 4.5 Checkpoint 保存与恢复

**本 fork 内嵌**：

```python
import tempfile
from reverb.platform.default import checkpointers

root = tempfile.mkdtemp()
server = reverb.Server(
    tables=[reverb.Table(name='c', ...)],
    in_process=True,
    checkpointer=checkpointers.DefaultCheckpointer(path=root))
client = server.in_process_client

# 写入后保存
client.checkpoint()   # 返回 checkpoint 目录路径

# 新进程/新 server 启动时,自动从 root 恢复最新 checkpoint
server2 = reverb.Server(
    tables=[reverb.Table(name='c', ...)],
    in_process=True,
    checkpointer=checkpointers.DefaultCheckpointer(path=root))
# server2 构造时已调 load_latest(),恢复完成
```

> 说明：`Server(in_process=True)` 构造时会自动调 `InProcessClient.LoadLatest()`
> 恢复最新 checkpoint。首次启动（空目录）是 `FileNotFoundError`，被当作正常情况
> 警告并跳过；checkpoint 损坏则抛异常（不静默吞错）。旧版 TF checkpoint
> **不兼容**，无法恢复。

---

## 5. 架构总览

```
Python API
  server.py            client.py            trajectory_writer.py
  (Server+Table)       (Client/LocalClient  (TrajectoryWriter/
                       共享 _BaseClient)     TrajectoryColumn)
       │ pybind11 (type_caster<TensorBuffer> 自动 ndarray 互转)
       ▼
  ┌─────────────┐   gRPC    ┌──────────────┐
  │InProcessClient│────────▶│   Client     │  (gRPC 路径,分布式)
  │ (零网络直连)  │         │ (localhost:port)│
  └──────┬──────┘          └──────┬───────┘
         │ 直接持有               │ gRPC stream
         ▼                        ▼
  ┌──────────────────────────────────┐
  │             Table                │
  │  mutex + table_worker + ext_worker (零 GIL)
  │  ├── ItemSelector (Fifo/Lifo/Uniform/Prioritized/Heap x2)
  │  ├── RateLimiter
  │  └── ChunkStore (持有 TensorBuffer bytes)
  └──────────┬───────────────────────┘
             │ InsertOrAssignAsync
             ▼
  TrajectoryWriter(table)   ← 本地路径分支 (is_local_=true)
    ├── Chunker (TensorBuffer)
    └── 列式 append → create_item → flush → Table

数据载体: TensorBuffer (std::string bytes + TensorSpec{DataType, shape})
  - FromNdArray: 强制 C-contiguous, memcpy
  - worker 线程读 bytes(), 零 GIL
  - 拷贝语义 (ponytail: 预留零拷贝升级路径)

proto: third_party/reverb_tensor (自研 fork, 切断 @org_tensorflow)
checkpoint: SimpleCheckpointer (length-delimited protobuf, 非 TFRecord)
```

**两种模式的数据流**：

- **内嵌**：`LocalClient` → `InProcessClient` → `Table`（直接指针，零序列化）
- **gRPC**：`Client` → gRPC stream → `Server` → `Table`（序列化 numpy bytes）

C++ 闭包零 TF（`bazel query deps(//reverb:pybind)` 无 `org_tensorflow`/`local_xla`），
Python 闭包零 TF（无 `tf_nightly`/`keras`）。首次 `bazel build` 不再下载 TF 源码树。

---

## 6. 附录：本地 Writer 解绑与本地化（D1/D2/D3-a 摘录）

> 本节摘录自 [unbind-local-writer-plan.md](unbind-local-writer-plan.md) 与
> [adr/0001-embedded-writer-local-path.md](adr/0001-embedded-writer-local-path.md)，
> 是对上文 §3.2/§3.4/§3.5 所述修正的完整背景与决策记录。

### 6.1 问题溯源

设计文档 §3.2 把“本地 `TrajectoryWriter` 绑定单一 table”当作类结构折中的一部分，
§3.4（`LocalClient.sample` 默认 `emit_timesteps=False`）和 §3.5（`LocalClient` 无
`insert`/`writer`）被列为两个平行的“不得不做的变更”。

代码核实后发现这三者其实是**同一个根因的三面**：

1. 本地 writer 构造函数收 `shared_ptr<Table> table_`，worker `RunLocalWorker`
   硬编码 `table_->InsertOrAssignAsync(...)`。
2. `CreateItem(table, ...)` 的 `table` 参数在本地路径**完全不用于路由**——
   只写进 `item.set_table()` 字段，该字段本地路径从不读取；signature 校验默认
   跳过（`options.flat_signature_map = nullopt`）。
3. 绑定单一 table → `LocalClient.trajectory_writer(table=...)` 必须显式传 table
   → 无法对齐 gRPC `Client.trajectory_writer(num_keep_alive_refs)` 的无 table 签名
   → 砍掉 `insert`/`writer`（§3.5）→ 只剩多步 trajectory 写入 → 翻默认值（§3.4）。

**判断**：当时选择绑定单一 table 是偏懒。`InProcessClient` 本就持有
`flat_hash_map<string, shared_ptr<Table>> tables_`，把 writer 改成持 map、worker
按 `item.table()` 查表分发，是顺着现有结构就能做的事。省了几行 C++，付了三层
API 债。修正把债还掉。

### 6.2 决策点

#### D1：`NewTrajectoryWriter`/`NewStructuredWriter` 去掉 `table` 参数

核实 gRPC `Client::NewTrajectoryWriter` 签名**不收 table 参数**。gRPC 路径的表名
校验延迟到 `CreateItem` 时，由 `ItemAndRefs::Validate` 用 `flat_signature_map`（从
`ServerInfo` 缓存里拿的全部表 signature）查表做。

**决策：去掉 `table` 参数，完全对齐 gRPC 签名。** 理由：保留 `table` 作 early 校验
会引入新的签名不对称（gRPC 不收、内嵌收），陷入“看起来对齐、实际契约不同”的陷阱——
正是本次修正要消灭的东西。对齐比 early 校验更重要；gRPC 的延迟校验本就是 proven
设计，内嵌沿用即可。

落地：

- `InProcessClient::NewTrajectoryWriter(const Options& options, ...)` 不收 table，
  writer 构造时持完整 `tables_` map。
- `RunLocalWorker` 按 `item.table()` 查 `tables_` 分发，找不到报 `kNotFound`。
- signature 校验对齐 gRPC：用现成的 `internal::FlatSignatureFromSignatureProto`（gRPC
  侧同样走的成熟函数）把各 Table 的 signature 转成 `FlatSignatureMap` 塞进 options。
  `ItemAndRefs::Validate` 走 gRPC 同一路径，无边界 bug 风险。
- `NewStructuredWriter` 同步去掉 `table` 参数（内部调 `NewTrajectoryWriter`）。

#### D2：backpressure 多表化——writer 级，对齐 gRPC

核实 gRPC 路径的 backpressure 是 **writer 级单一通道**：所有表的数据共用一个 gRPC
`InsertStream`，`write_inflight_` 是 writer 级单一 bool，`WriteIfNotEmpty` 在 write
在飞期间 `mu_.Await` 阻塞。不管 item 要写哪个表，只要 stream 上有一个 write 未完成，
整个 writer 就停。

**决策：writer 级 backpressure，任意表满则 writer 停。** 理由：与 gRPC 语义完全
同构——gRPC 是 stream 级阻塞，本地是 `local_can_insert_more_` 单一 flag，两者语义
等价。改动最小（沿用单一 flag，只是 `InsertOrAssignAsync` 的表从固定变为按 item
查），单表场景行为不变。

代价：A 满 B 没满时 writer 也停，但这是可接受的（与 gRPC 一致，且一个 writer 交替
写多表本就是 `insert` 多表场景，停顿不会放大）。

#### D3：Python `LocalClient.insert` / `writer`——给 `Writer` 类加本地路径

核实 `insert` 与 `writer` 的绑定关系及 gRPC `Client` 用法：

- `Client.insert` **硬依赖** `writer`：`insert` 实现就是
  `with self.writer(max_sequence_length=1) as w: w.append(data); w.create_item(...)`。
  `insert` 是 gRPC `Client` 高频主力 API（tests 26 处用法），内部依赖 `writer`，
  两者都是活 API。
- `Writer` 类**没有真正废弃**：仍在 `__init__.py` 正式导出，docstring 是
  "will eventually be deprecated"（将来某天），非已废弃。本 fork 还主动恢复了它
  （去掉 `NotImplementedError`）。
- `Writer` 类是**独立实现**，非 `TrajectoryWriter` 的封装：自带
  `buffer_`/`chunks_`/`pending_items_`/gRPC `stream_`/`item_confirmation_worker_` 线程，
  不用 chunker/column 抽象，是更早期的同步实现。

**决策：给 `Writer` 类加本地路径，实现 `LocalClient.writer`/`insert`，与 gRPC `Client`
完全对齐。**

##### gRPC 链路调研：server 侧“远端处理”只有三件事，且全不依赖 gRPC

`ProcessIncomingRequest` 的全部工作：

1. **`SaveChunks`**——把请求里的 `ChunkData` 存进 reactor 的 `chunks_` map，纯内存操作。
2. **`GetItemWithChunks`**——按 item 的 `flat_trajectory` 引用的 chunk key，从
   `chunks_` 找出对应 `shared_ptr<Chunk>`，构造 `Table::Item(item, chunks)`，纯内存操作。
3. **`table->InsertOrAssignAsync`**——按表名查表，调 `InsertOrAssignAsync` 插入，
   `insert_completed_` callback 在插入完成后触发。

C++ `Writer` 本就持有与 reactor 一一对应的成员（`chunks_` 对应 reactor 的 `chunks_`，
`tables_[table_name]` 对应 `server_->TableByName`，callback 对应 reactor 的
`insert_completed_`）。因此本地路径几乎是把 reactor 的 `ProcessIncomingRequest`
平移进 `Writer::WritePendingData`——变量名都一一对应。

实际工作量：绝大部分原样复用（`Append`/`AppendSequence`/`CreateItem`/`Finish`/
`Flush`/`Close`/`ConfirmItems` 同步骨架不动），只改 `WritePendingData`——把
`stream_->Write(request)` 替换为内联的 `SaveChunks` + `GetItemWithChunks` +
`tables_[name]->InsertOrAssignAsync`。

##### 确认机制：保留 `Writer` 同步骨架，callback 递减计数

`Writer` 是同步模型：调用线程直接 `Append`→`Finish`→`WritePendingData`，
`ConfirmItems(limit)` 在调用线程同步阻塞等 `num_items_in_flight_` 降下来。这与
`TrajectoryWriter` 的异步模型（write_queue + worker 线程）根本不同，不能套用
`TrajectoryWriter` 的 `local_can_insert_more_` flag 模式。

本地化方案：保留同步骨架，只换信号源——每 insert 建 per-item `InsertCallback`，
捕获 `this`，在表 worker 线程触发时持 `mu_` 递减 `num_items_in_flight_` 并 signal。
`ConfirmItems(limit)`/`num_items_in_flight_` 不动，唤醒源从“gRPC READ 响应”换成
“callback signal”。`Close()` 本地路径：`ConfirmItems(0)` drain → 释放 callback，
无遗留 in-flight，析构安全。

`ConfirmItems` 的 `done` 条件本地路径改为 `num_items_in_flight_ <= limit || closed_`
（加 `closed_` 逃生，防表 worker 出错致 callback 永不触发时永久阻塞，对齐
`TrajectoryWriter` 的 `!closed_ && !stream_ok_` 等待模式）。

### 6.3 修正目标与交付边界

**本次修正是一个原子交付单元**——`TrajectoryWriter` 解绑与 `Writer` 本地化（D3-a）
**一起上，不拆分**。理由：简化用户认知（一次性对齐），避免中间态（只解绑不补 Writer
时，`LocalClient` 有 `trajectory_writer` 但无 `insert`/`writer`，仍不对称）。

- **C++ 层**：本地 `TrajectoryWriter`/`Writer` 持 `tables_` map 而非单一 `table_`，
  worker 按 `item.table()` 分发。`CreateItem` 指向未知表时报错（原静默）。
- **Python 层**：`LocalClient.trajectory_writer`/`structured_writer` 去掉 `table` 参数，
  签名对齐 gRPC `Client`。补 `insert`/`writer`，语义对齐。
- 消解 §3.4：`_default_emit_timesteps` 统一回 `True`。
- 保留 §3.5 中唯一真实的物理约束：无 `__reduce__`（pickle），因为 `LocalClient`
  持进程内指针不可跨进程序列化。

### 6.4 writer 持 map 的生命周期方式：持拷贝（方案 P）

两个类（`TrajectoryWriter`/`Writer`）解绑后都要持 `tables_` map。决策：**writer 持
map 拷贝**（复制 `flat_hash_map<string, shared_ptr<Table>>`，每个 `shared_ptr<Table>`
引用计数 +1），不改 `InProcessClient::tables_` 为 `shared_ptr<map>`。理由：

1. **checkpoint 恢复语义正确**：`LoadLatest` 原地改写 `Table` 对象内容
   （`InitializeFromCheckpoint` 重建 sampler/remover/rate_limiter，
   `InsertCheckpointItem` 灌回 item），**不替换 `shared_ptr<Table>` 指向**。writer 持
   的 shared_ptr 副本与 client 的指向同一 Table 对象，`LoadLatest` 后 writer 自动看到
   恢复状态。
2. **`LoadLatest` 前提保证安全**：`LoadLatest` 要求各 table 为空，即仅新建 server 时
   调用一次。有 writer 在写的表不空，不会与 `LoadLatest` 竞态。
3. **性能可忽略**：writer 创建是 per-context（per-episode），非 per-step 热路径。
   map 拷贝 = 几个 `shared_ptr` 原子递增（典型 1-4 个表，十几纳秒），相对 writer
   构造本身的微秒/毫秒开销是噪声。
4. **ripple 最小**：不改 `InProcessClient` 成员类型，`LoadLatest`/`Load`/`GetTable`
   逻辑全不动。

### 6.5 ADR-0001：推翻 A1 砍 Writer 决策

**背景**：设计文档 §1.3 决策 A1 当初决定：内嵌模式只做 `TrajectoryWriter` +
`StructuredWriter`，旧 `Writer`/`StreamingTrajectoryWriter` 的本地路径“价值低，
砍掉省复杂度”。这导致 `LocalClient` 无 `insert`/`writer`，与 gRPC `Client` 的 API
不对称（§3.5），并连带逼出 `LocalClient.sample` 默认 `emit_timesteps=False` 的偏差
（§3.4）。

**触发推翻的事实**：

- `Writer` **没有真正废弃**：仍在 `__init__.py` 正式导出，docstring 是 "will
  eventually be deprecated"（将来某天），非已废弃；本 fork 还主动恢复了它。
- `insert` **硬依赖** `writer`：`insert` 实现就是
  `with self.writer(max_sequence_length=1) as w: w.append(data); w.create_item(...)`。
  `insert` 是 gRPC `Client` 高频主力 API（tests 26 处用法），A1 砍 `writer` 等于内嵌
  用户也丢了 `insert` 这个便捷入口。
- gRPC `Client` 三套写入 API（`insert`/`writer`/`trajectory_writer`）并存且都活，
  `trajectory_writer` 是推荐项但不替代另两者。

**决策：推翻 A1 关于 Writer 的部分。** 给 `Writer` 类加本地路径（基于 server 侧
`ProcessIncomingRequest` 的三件事全不依赖 gRPC，可平移进 `Writer::WritePendingData`）。
`LocalClient` 补 `writer`/`insert`，与 gRPC `Client` API 严格镜像。A1 的“省复杂度”
理由被“API 严格镜像”取代。

**后果**：

- `LocalClient` 具备 `insert`/`writer`/`trajectory_writer`/`structured_writer` 全套，
  与 gRPC `Client` 签名一致，代码可直接迁移。
- `_default_emit_timesteps` 两端统一为 `True`（消解 §3.4）。
- §3.5 缩减为仅“无 pickle”（`__reduce__`），这是 `LocalClient` 持进程内指针的真实
  物理约束，非 API 债。
- `StreamingTrajectoryWriter` 仍不在内嵌范围（A1 该部分保留，本决策只推翻 Writer 部分）。

**状态**：accepted。supersedes §1.3 决策 A1 中“旧 `Writer` 本地路径砍掉”的部分。
