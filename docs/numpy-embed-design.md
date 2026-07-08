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
  - [3.4 `LocalClient.sample` 默认 `emit_timesteps=False`](#34-localclientsample-默认-emit_timestepsfalse)
  - [3.5 `LocalClient` 无 `insert` / `writer` / pickle](#35-localclient-无-insert--writer--pickle)
  - [3.6 旧 checkpoint 不兼容](#36-旧-checkpoint-不兼容)
- [4. 使用样例对比](#4-使用样例对比)
  - [4.1 创建 Server 与 Client](#41-创建-server-与-client)
  - [4.2 写入数据](#42-写入数据)
  - [4.3 采样数据](#43-采样数据)
  - [4.4 StructuredWriter](#44-structuredwriter)
  - [4.5 Checkpoint 保存与恢复](#45-checkpoint-保存与恢复)
- [5. 架构总览](#5-架构总览)

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
| A1 | 内嵌只做 `TrajectoryWriter` + `StructuredWriter` | 这两个是现代推荐 API；旧 `Writer`/`StreamingTrajectoryWriter` 的本地路径价值低，砍掉省复杂度 |
| A2 | 保留 gRPC 层 | 一套 C++ 代码两种模式（内嵌 + 分布式），不为内嵌牺牲分布式能力 |
| A3 | 彻底去 TF，fork 精简 proto 到 `third_party/` | TF proto 链式拉入一堆传递依赖，精简 fork 切断 `@org_tensorflow` |
| B1 | `TensorBuffer` 用拷贝 bytes（`std::string`） | worker 线程零 GIL，简单正确；预留零拷贝升级路径（`ponytail:` 标注） |
| B2 | 自研 `TensorSpec` + `TensorBuffer` | 不依赖 TF，对齐 numpy dtype |
| C1 | pybind 暴露 `timeout` 参数，超时抛 `DeadlineExceededError` | 原版 `TimeoutError` 与 Python 内置同名，易混淆 |
| C2 | 保留 `table_worker_` + `extension_worker_` 异步线程 | 零 GIL 是性能基础，不能丢 |
| D3 | checkpoint 用 length-delimited protobuf | TFRecord 去掉 CRC32 就是标准 length-delimited protobuf，~20 行实现 |

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
  absl::Status NewTrajectoryWriter(const std::string& table,
                                   const TrajectoryWriter::Options& options,
                                   std::unique_ptr<TrajectoryWriter>* writer);
  absl::Status NewStructuredWriter(const std::string& table, ...);
  absl::Status NewSampler(const std::string& table_name, ...);
  absl::Status MutatePriorities(...);
  absl::Status Reset(...);
  absl::Status Checkpoint(std::string* path);
  absl::Status LoadLatest();
  absl::Status Load(absl::string_view path);
  absl::Status ServerInfo(std::vector<TableInfo>* table_info);
};
```

Python 层有两个客户端类，共享 `_BaseClient` 的 `sample`/`mutate_priorities`/
`reset`/`server_info`/`checkpoint` 逻辑：

- **`Client`**（gRPC）：构造为 `Client('localhost:port')`，走 gRPC。保留原版的
  `insert`/`writer`/`trajectory_writer`/`structured_writer` 全部工厂方法。
- **`LocalClient`**（内嵌）：由 `Server(in_process=True).in_process_client` 返回，
  包装 C++ `InProcessClient`。只提供 `trajectory_writer`/`structured_writer`/
  `new_sampler`/`sample` 等方法（无 `insert`/`writer`，见 [3.5](#35-localclient-无-insert--writer--pickle)）。

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

本 fork 给 `TrajectoryWriter` 加了 `shared_ptr<Table>` 构造函数和本地路径分支
（`is_local_` 标志）：本地路径调 `Table::InsertOrAssignAsync`，用 callback 替代
gRPC 的 `OnReadDone`。

> **决策**：原计划新建独立的 `LocalTrajectoryWriter` 类，实际改为直接在
> `TrajectoryWriter` 内加分支——复用全部 chunker/column 逻辑，避免代码重复。
> `InProcessClient::NewTrajectoryWriter` 构造绑定了单一 table 的本地 writer。

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

### 3.4 `LocalClient.sample` 默认 `emit_timesteps=False`

`sample(emit_timesteps=...)` 控制返回整条 trajectory 还是按 timestep 拆分。

- gRPC `Client` 默认 `True`（向后兼容原版行为）。
- 内嵌 `LocalClient` 默认 `False`（内嵌场景几乎总是想要整条 trajectory，不是
  timestep 拆分列表）。

```python
class LocalClient(_BaseClient):
  _default_emit_timesteps = False   # gRPC Client 是 True
```

调用方可显式覆盖。这个默认值差异是**有意为之**，匹配两种模式的典型用法。

### 3.5 `LocalClient` 无 `insert` / `writer` / pickle

gRPC `Client` 有 `insert(data, priorities)`（便捷插入完整 trajectory）、
`writer(max_sequence_length)`（旧版流式 Writer）、`__reduce__`（pickle 支持，
因为 Client 只存 server 地址，可跨进程重建）。

`LocalClient` **没有**这些：

- `insert`/`writer` 是旧 API，内嵌场景推荐用 `trajectory_writer`。`LocalClient`
  的 `trajectory_writer` 需要显式指定 `table`（因为本地 writer 绑定单一 table）。
- pickle 无意义：`LocalClient` 持有进程内 Table 指针，不可跨进程序列化。

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
with client.trajectory_writer(table='t', num_keep_alive_refs=10) as w:
    w.append({'obs': np.zeros(4, dtype=np.float32)})   # 纯 numpy
    w.create_item(
        table='t', priority=1.0,
        trajectory={'obs': w.history['obs'][:]})       # 引用最近数据
    w.flush()
```

> 说明：内嵌 `trajectory_writer` 需显式传 `table`（本地 writer 绑定单一表），
> 而 gRPC `Client.trajectory_writer(num_keep_alive_refs)` 不传 table（每个 item
> 在 `create_item` 时指定表）。`num_keep_alive_refs` 是循环缓冲区大小，即 trajectory
> 最大跨度。`w.history['col'][:]` 返回 `TrajectoryColumn`，`create_item` 的
> `trajectory` 是一个结构与期望采样结构一致嵌套 dict/list。

### 4.3 采样数据

**原版（返回 timestep 列表）**：

```python
for sequence in client.sample('t', num_samples=4):
    for step in sequence:          # 每个 sequence 是 timestep 列表
        print(step.data)
```

**本 fork 内嵌（默认返回整条 trajectory）**：

```python
for sample in client.sample('t', num_samples=4):
    # sample 是单个 ReplaySample(因 LocalClient 默认 emit_timesteps=False)
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

writer = client.structured_writer(table='t', configs=[config])
for i in range(5):
    writer.append(np.asarray(float(i), dtype=np.float32))
writer.end_episode()

for sample in client.sample('t', num_samples=3):
    print(np.asarray(sample.data[0]))   # [0., 1., 2.], [1., 2., 3.], ...
```

> 说明：内嵌 `structured_writer` 需显式传 `table`（与 `trajectory_writer` 一致），
> 所有 config 的 item 都写入该表。gRPC `Client.structured_writer(configs)` 不传
> table（每个 config 的 `table` 字段指定目标表）。

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
