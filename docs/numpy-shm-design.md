# Reverb 同机分进程共享内存传输设计文档

> 本文档阐述 Reverb 在"同机、Client 与 Server 分属不同进程"场景下，使用
> POSIX 共享内存（SHM）替代 gRPC loopback 进行数据传输的方案。面向已熟悉
> [numpy-embed-design.md](numpy-embed-design.md) 的开发者。

## 0. TL;DR

- **痛点**：同机分进程下，gRPC loopback 对每个 tensor 做 proto 序列化 +
  snappy 压缩/解压，CPU 与内存带宽是瓶颈。sample 路径尤甚——server 存的
  chunk 是压缩的，**每次采样都解压一次**，同一 chunk 解压 N 次。
- **方案**：新增独立 SHM 传输层，与 gRPC / in_process 三路并存。Table 逻辑
  留在 server 进程，SHM 只承载 chunk 字节，控制面走 per-client 双向 SPSC
  ring buffer + server 轮询，bootstrap 用 Unix domain socket。
- **API**：新增 `ShmClient`，镜像 `_BaseClient`（sample / insert /
  trajectory_writer / structured_writer），语义与 gRPC / LocalClient 完全一致，
  用户代码三路可无缝迁移。

## 1. 动机与边界

### 1.1 要解决的

gRPC loopback（`Client('localhost:port')` 连本机 server）在同机分进程下可用，
但数据路径是：

```
client ndarray
  → CompressTensorAsProto (snappy 压缩 + proto 序列化)   [client CPU]
  → gRPC 序列化 ChunkData                                  [双方 CPU]
  → server 解析 → ChunkStore 存压缩 proto
  → sample 时: DecompressTensorFromProto (每次采样解压)   [server CPU, N 次]
  → gRPC 序列化回传 → client 解析 → DecompressTensorFromProto [双方 CPU]
```

核心浪费：proto 序列化 + snappy 压缩/解压的 CPU 与内存带宽开销，sample 路径
重复解压尤甚。SHM 直接共享未压缩字节，跳过全部序列化与压缩。

### 1.2 不解决的

- **跨机**：SHM 不跨机。跨机仍走 gRPC。
- **Table 算法核心**：selector / rate limiter / chunker / worker 线程不动，
  仍在 server 进程。
- **现有 gRPC / in_process 路径**：保留，不改动。

### 1.3 与内嵌模式的差异

| 维度 | in_process（已实现） | shm（本文档） |
| --- | --- | --- |
| 进程关系 | 同进程 | 同机不同进程 |
| Table 访问 | `shared_ptr<Table>` 直接指针 | server 进程独占，client 经控制 ring 请求 |
| chunk 字节 | `shared_ptr<ChunkStore::Chunk>` 共享 | SHM 段共享字节 + server 集中引用计数 |
| 序列化 | 零 | 零（控制消息除外） |
| 压缩 | 零（in_process 走 `AsSample(SampledItem)` 直接读 `shared_ptr`） | 零数据面；server 内部仍压缩存 ChunkStore |

> 注：in_process 的 sample 走 `sampler.cc::AsSample(const Table::SampledItem&)`
> 路径，直接 `shared_ptr<Chunk>` 零拷贝读，本就不压缩。SHM 复刻这套"零压缩读"
> 语义，但字节在 SHM 段而非进程内指针。

## 2. 设计决策汇总

| 编号 | 决策 | 理由 |
| --- | --- | --- |
| S1 | 新增独立 SHM 传输层，与 gRPC / in_process 三路并存 | 不侵入现有 C++ 核心；用户显式选传输 |
| S2 | SHM 只放 chunk 字节，Table 逻辑留 server 进程 | 复杂度量级最低；避免跨进程 shared_ptr / 互斥锁 / worker 调度 |
| S3 | 控制面走 per-client 双向 SPSC ring buffer + server 轮询 | SPSC 无锁成熟；崩溃隔离好（某 client ring 坏不影响其他） |
| S4 | bootstrap 用 Unix domain socket | POSIX 标准、同机专用、可检测断连 |
| S5 | 字节池用固定档位 slab 分配 | 碎片少、回收简单（归位即可） |
| S6 | server 集中管理 SHM 字节引用计数，client 只读不释放 | 语义清晰；client 崩溃时 server 集中释放 |
| S7 | SHM 池与现有 ChunkStore 并存（双份内存：压缩 proto + 未压缩字节） | 现有 Table / ChunkStore / sampler 零改动 |
| S8 | insert：client 送原始字节进 SHM，server 压缩存档 | client 端零压缩（动机核心） |
| S9 | sample：server 预切片成成品字节进 SHM，client 直接读 | client 侧零计算；每次独立分配不复用 |
| S10 | rate limiter / backpressure / checkpoint 语义完全对齐现有 | 用户代码三路无缝迁移 |
| S11 | Python 新增 `ShmClient`，镜像 `_BaseClient` | API 一致；三路并列 |
| S12 | 支持 trajectory_writer / structured_writer；insert 后补 | writer backpressure 经反向 ring confirm |
| S13 | 崩溃恢复：udsocket 断连检测 + 集中释放该 client SHM 偏移 | 简单可靠 |
| S14 | 字节池满：阻塞等待，对齐全语义 | 不报错、不丢数据 |
| S15 | SHM 用 POSIX `shm_open` | 跨平台、与 udsocket 配合自然 |

## 3. 架构总览

```
┌─────────────── Client 进程 ───────────────┐   ┌──────── Server 进程 ────────┐
│                                             │   │                              │
│  ShmClient (pybind)                         │   │  Table (selector/RL/worker)  │
│   ├── trajectory_writer (chunker 在 client) │   │  ChunkStore (压缩 proto)     │
│   ├── structured_writer                     │   │  ShmBytePool (未压缩字节)    │
│   └── sample / insert                       │   │  ShmServer (dispatch 线程)   │
│        │                                    │   │        ▲                     │
│        ▼                                    │   │        │ 控制消息             │
│  ShmConnection                              │   │        │                     │
│   ├── udsocket (bootstrap + 断连检测) ◄─────┼───┼────────┼─────────────────────│
│   ├── ctrl ring C→S (SPSC, client 写)       │   │        │                     │
│   ├── ctrl ring S→C (SPSC, server 写)       │   │        │                     │
│   └── byte pool (POSIX shm, 共享) ◄─────────┼───┼────────┴─────────────────────│
│        (client 只读字节)                    │   │  (server 分配/回收/引用计数) │
└─────────────────────────────────────────────┘   └──────────────────────────────┘
```

**数据流（insert）**：

```
client ndarray → memcpy 进 SHM 字节池档位块
  → ctrl ring C→S 发 insert 请求(含 SHM 偏移+spec+table+priority+flat_trajectory)
  → server ShmServer 线程读 ring
  → 用 SHM 字节构造压缩 ChunkData → InsertOrAssignAsync 进 Table/ChunkStore
  → 反向 ring S→C 发 confirm(item key)
  → client writer 递减 num_items_in_flight(backpressure)
```

**数据流（sample）**：

```
client → ctrl ring C→S 发 sample 请求(table, num_samples, timeout)
  → server ShmServer → Table::Sample(走现有 rate limiter/selector)
  → 对每个 SampledItem: UnpackChunkColumnAndSlice 解压+切片(现有逻辑)
    → 成品 TensorBuffer 字节 memcpy 进 SHM 字节池
  → 反向 ring S→C 发 sample 响应(各列的 SHM 偏移+spec+shape+SampleInfo)
  → client 按偏移从 SHM 读字节 → ToNdArray → 组装 Sample
  → client 读完发 release 请求(C→S ring) → server 回收偏移
```

## 4. 组件设计

### 4.1 SHM 字节池 `ShmBytePool`

- **技术**：POSIX `shm_open` + `ftruncate` + `mmap`，固定路径名（由 server 生成，
  bootstrap 时传 client）。
- **布局**：固定档位 slab。档位如 64B / 1KB / 16KB / 256KB / 4MB（可配）。每
  档位一个 free list（偏移链表）。server 单线程分配/回收（无需跨进程锁）。
- **分配**：按请求大小选最小够用档位，从该档位 free list 取一块返回偏移。无空闲
  则向上取更大档位或报池满。
- **回收**：偏移归还所属档位 free list。
- **引用计数**：server 维护 `map<偏移, refcount>`。sample 发 N 列则各偏移 +1；
  client release 则 -1；归零回收。
- **池满**：阻塞等待（S14）。server 端 `ShmServer` 线程在池满时阻塞，对应的
  client 请求排队（控制 ring 自然背压）。
- **容量**：默认 2x 所有表 max_size × 平均 chunk 字节，可配。

> `ponytail:` slab 档位是经验值，profile 后可调。跨进程分配单线程化是简化——
> 若 server dispatch 成瓶颈，升级为 per-档位 spinlock。

### 4.2 控制面 `ShmCtrlRing`（per-client 双向 SPSC）

- **结构**：每个 client 连接时，server 为其创建两条 SPSC ring（POSIX shm）：
  - `C→S`：client 写，server 读（insert / sample / release / reset 等请求）
  - `S→C`：server 写，client 读（sample 响应 / confirm / 错误）
- **SPSC 实现**：经典无锁单生产者单消费者 ring，`head`/`tail` 原子变量，cache line
  对齐防 false sharing。容量固定（如 1024 槽），每槽定长（放控制消息，大数据用
  SHM 偏移引用）。
- **消息类型**：定长 header（type + length + seq）+ 变长 body（proto 或自定义
  二进制）。body 超单槽时跨槽拼接。
- **server 轮询**：`ShmServer` 单线程轮询所有 client 的 C→S ring，dispatch 到
  Table。响应写回对应 client 的 S→C ring。

> `ponytail:` SPSC ring 不用信号量（避免崩溃泄漏），靠 server 主动轮询。延迟
> 由轮询间隔决定（默认忙等或微秒级 sleep）。若延迟敏感，升级为 eventfd 通知。

### 4.3 Bootstrap 与连接 `ShmBootstrap`

- **server 侧**：启动时创建主 SHM 字节池段（POSIX shm，生成唯一路径名如
  `/reverb_shm_pool_<pid>`），并在一个 Unix domain socket 路径上 listen。
- **client 侧**：连 udsocket，发送 `Hello{client_id}`。server 回
  `Welcome{pool_shm_name}`，并为该 client 创建 C→S / S→C 两条 ring 段，回
  `{c2s_shm_name, s2c_shm_name}`。client `mmap` 三段（pool + 两条 ring）。
- **断连检测**：server 监听 udsocket EOF，判定 client 断连，集中释放该 client
  所有未 release 的 SHM 偏移，销毁其 ring 段。

### 4.4 server 侧 `ShmServer`

- **线程模型**：一个 dispatch 线程轮询所有 client C→S ring + 一个表 worker 池
  （复用现有 `Table::table_worker_`）。
- **insert 处理**：读 ring 请求 → 用 SHM 字节构造压缩 `ChunkData`（调现有
  `CompressTensorAsProto`）→ `Table::InsertOrAssignAsync`（带 `InsertCallback`）
  → callback 触发时往该 client S→C ring 写 confirm。
- **sample 处理**：读 ring 请求 → `Table::Sample`（现有 rate limiter/selector）
  → 对每个 `SampledItem` 调 `UnpackChunkColumnAndSlice`（现有逻辑）解压切片 →
  成品字节进 SHM 池 → S→C ring 写响应（偏移列表 + SampleInfo）。
- **与 ChunkStore 关系**：SHM 池是 ChunkStore 之外的并行数据面。insert 时 server
  既存压缩 proto 进 ChunkStore，也保留 SHM 字节供 sample 解压来源（或直接用
  insert 时 client 送来的原始 SHM 字节做 sample 切片源，省一次解压——见 §6）。

### 4.5 client 侧 `ShmClient`（C++ + pybind）

- **C++ `ShmClient`**：持 `ShmConnection`（udsocket + 三段 mmap 指针 + C→S/S→C
  ring 读写器）。方法镜像 `InProcessClient`：`Sample` / `Insert` /
  `NewTrajectoryWriter` / `NewStructuredWriter` / `MutatePriorities` / `Reset` /
  `Checkpoint` / `ServerInfo`。
- **trajectory_writer**：chunker/column 逻辑在 client 进程（复用现有
  `TrajectoryWriter` 的大部分，只把 `RunLocalWorker` 的 `InsertOrAssignAsync`
  换成"字节进 SHM 池 + C→S ring 发 insert + 等 S→C confirm"）。backpressure：
  `local_can_insert_more_` 等 confirm 信号（对齐 D2）。
- **sample**：发请求 → 轮询 S→C ring → 按偏移读 SHM 字节 → `TensorBuffer`
  → `ToNdArray` → 组装 `Sample`（复用 `sampler.cc::AsSample` 的 `Sample` 构造）。
- **pybind**：暴露 `ShmClient(udsocket_path)`，snake_case + PascalCase 双名别名
  （对齐 §3.3 的 gRPC/LocalClient 共享路径策略）。

### 4.6 Python 层

- `reverb/client.py`：新增 `ShmClient(_BaseClient)`。`_BaseClient` 的
  `sample`/`insert`/`mutate_priorities`/`reset`/`server_info`/`checkpoint` 共享
  逻辑靠 `self._client`（C++ `ShmClient`）鸭子类型分派。`trajectory_writer`/
  `structured_writer` 不收 table 参数（对齐 gRPC/LocalClient）。
- `reverb/server.py`：`Server` 新增 `shm=True` / `shm_socket_path=...` 参数。
  `shm=True` 时起 `ShmServer`（含 udsocket + 字节池），`server.shm_socket_path`
  供 client 连。可与 `in_process` / `port` 组合或互斥（见 §5）。
- 无 pickle（同 LocalClient，持 SHM mmap 指针不可序列化）。

## 5. 模式组合矩阵

| Server 构造 | gRPC | in_process | shm |
| --- | --- | --- | --- |
| `Server(tables, port=8000)` | ✓ | ✗ | ✗ |
| `Server(tables, in_process=True)` | ✗ | ✓ | ✗ |
| `Server(tables, shm=True)` | ✗ | ✗ | ✓ |
| `Server(tables, port=8000, shm=True)` | ✓ | ✗ | ✓ |
| `Server(tables, in_process=True, shm=True)` | ✗ | ✓ | ✓ |

- gRPC、in_process、shm 三者可按需组合（同进程内嵌 + 同机跨进程并存）。
- client 侧按需用 `Client(addr)` / `server.in_process_client` / `ShmClient(path)`。

## 6. 未决 / 后续

- **insert 字节复用于 sample 切片源**：client insert 时送的原始 SHM 字节，sample
  时 server 可直接基于它切片，省去"压缩进 ChunkStore 再解压"的一次往返。需 server
  维持 `chunk_key → SHM 偏移` 索引（在 ChunkStore 之外）。这是 S7"并存"的优化变
  体，可作为 v2。v1 先按"server 从压缩 ChunkStore 解压到 SHM"实现，行为正确后再
  优化。
- **insert（单步）API**：依赖 writer，v2 补。
- **多 server 进程**：本文档不涉及（一个 server 进程，多 client）。
- **SHM 段权限**：POSIX shm 默认仅同用户。跨用户场景需 `chmod`/`chown`，后续按需。

## 7. 与 numpy-embed-design.md 的关系

- 本方案是 numpy-embed-design 的**同机跨进程扩展**，复用其 `TensorBuffer` /
  `ChunkData` / `SimpleCheckpointer` / `_BaseClient` 抽象。
- 不改动 numpy-embed-design 已定的任何决策（A1–D3），仅在传输层新增第三条路径。
- `TensorBuffer` 的"拷贝 bytes 语义"在 SHM 下自然延伸：SHM 段就是一块共享的
  bytes 缓冲，`TensorBuffer::bytes()` 可直接指向 SHM 偏移（零进程内拷贝）。
- ponytail 升级路径一致：SHM 字节池本身已是"零拷贝共享"，numpy-embed-design 里
  `TensorBuffer` 的零拷贝升级（裸指针 + DeferredFreeQueue）是正交的进程内优化。

## 8. SPSC ring 协议规范（可实现粒度）

本节是 §4.2 的展开。定义字节级布局、消息类型、流控、错误处理，供 dispatch
线程、pool、bootstrap 直接照写。

### 8.1 内存布局

每条 ring 是一个独立 POSIX shm 段，固定大小，`mmap` 后按以下 C 结构体布局
（POD，无指针，跨进程安全）：

```c
// 段头(64 字节, 单 cache line)
struct RingHeader {
  uint64_t magic;          // 0x524556524253484D ("REVRBSHM")
  uint32_t version;        // 协议版本, 当前 1
  uint32_t capacity;       // 槽位数(2 的幂)
  uint32_t slot_size;      // 每槽字节数(含 SlotHeader)
  uint32_t reserved;       // 对齐填充
  // 生产者/消费者各占独立 cache line, 防 false sharing
  // ponytail: 这两个必须是 atomic, SPSC 无锁靠 release/acquire 配对
  std::atomic<uint64_t> producer_seq  // 下一个要写的槽的 seq(从 1 开始)  __attribute__((aligned(64)));
  std::atomic<uint64_t> consumer_seq; // 下一个要读的槽的 seq              __attribute__((aligned(64)));
  uint64_t capacity_mask;  // capacity - 1, 用于 seq % capacity
  uint64_t padding[5];     // 填满 cache line
};
// 紧随其后的 slots 区: capacity 个 slot_size 槽
// 注: SlotHeader.seq 是普通 uint64_t(非 atomic), 靠 producer_seq/consumer_seq
//     的 release/acquire 建立可见性; 但写入时先置 0 再置 seq 的顺序需保证,
//     见 §8.4 说明
```

- **seq 而非 index**：用单调递增的 `seq`（从 1 开始），槽位 = `seq & capacity_mask`。
  比裸 index 多一位冗余，能区分"空"与"满"（满: producer_seq - consumer_seq ==
  capacity）。避免"满空不可区分需留一槽"的常见 ring 陷阱。
- **slot 占用判定**：每槽首 8 字节存"写入该槽的 producer_seq"（见 `SlotHeader`）。
  消费者读到 `slot.seq == consumer_seq` 表示槽有效；`slot.seq < consumer_seq`
  表示未写入或已被越过（跳过）。
- **容量**：`slot_size` 默认 256 字节，`capacity` 默认 1024（= 256KB/段）。控制
  消息多在单槽内；超长消息走 §8.4 的跨槽拼接。两值 2 的幂，位运算取模。

### 8.2 槽位结构

```c
struct SlotHeader {
  uint64_t seq;            // 写入方的 producer_seq(消费者据此判有效)
  uint16_t msg_type;       // 见 §8.3 消息类型表
  uint16_t flags;          // bit0: HAS continuation(跨槽); bit1: IS continuation
  uint32_t body_len;       // 本槽 body 字节数(<= slot_size - sizeof(SlotHeader))
};
// 紧随 body: slot_size - sizeof(SlotHeader) 字节
```

- **单槽消息**：`flags == 0`，`body_len` 为实际长度，body 即完整消息。
- **跨槽消息**（body 超单槽容量）：首槽 `flags |= HAS_CONT`，后续槽
  `flags |= IS_CONT`，末槽无 `HAS_CONT`。消费者按 `msg_type` 知道如何拼接。
  body 内承载的是一条 length-delimited proto（消息类型对应 proto，见 §8.3），
  跨槽即对该 proto 字节流分段。

### 8.3 消息类型（C→S 与 S→C）

消息体统一用 length-delimited protobuf（varint 长度前缀 + proto bytes），与现有
`reverb_service.proto` 字段对齐，便于复用现有 proto 定义。SHM 专有新增 proto 在
`reverb/cc/shm/shm_protocol.proto`。

#### C→S 请求（client 写，server 读）

| type 值 | msg_type 名 | body proto | 对应现有 RPC |
| --- | --- | --- | --- |
| 1 | `HELLO` | `HelloRequest{client_pid, client_protocol_version}` | bootstrap（仅初始一次，见 §8.6） |
| 2 | `INSERT` | `ShmInsertRequest`（见下） | InsertStream |
| 3 | `SAMPLE` | `ShmSampleRequest{table, num_samples, timeout_ms, emit_timesteps}` | SampleStream |
| 4 | `RELEASE` | `ShmReleaseRequest{repeated uint64 offsets}` | SHM 专有（回收字节池偏移） |
| 5 | `MUTATE_PRIORITIES` | `MutatePrioritiesRequest`（复用现有 proto） | MutatePriorities |
| 6 | `RESET` | `ResetRequest{repeated string table_names}`（复用） | Reset |
| 7 | `CHECKPOINT` | `CheckpointRequest`（复用） | Checkpoint |
| 8 | `SERVER_INFO` | `ServerInfoRequest`（复用） | ServerInfo |
| 9 | `CLOSE` | 空 | client 主动关闭 |

```protobuf
// shm_protocol.proto (新增)
message ShmInsertRequest {
  // trajectory/item 元信息(对齐 InsertStreamRequest, 但 chunk 字节走 SHM)
  repeated ShmChunkRef chunks = 1;       // 替代 InsertStreamRequest.chunks
  repeated PrioritizedItem items = 2;   // 复用现有 proto, flat_trajectory 引用 chunk_key
  repeated uint64 keep_chunk_keys = 3;
}
message ShmChunkRef {
  uint64 chunk_key = 1;
  uint64 shm_offset = 2;        // ShmBytePool 偏移
  uint64 length = 3;            // 字节数
  TensorSpecProto spec = 4;     // dtype + shape(每列一个 ref? 见 §8.5)
  SequenceRange sequence_range = 5;
  bool delta_encoded = 6;
  int32 num_columns = 7;
}
message ShmSampleRequest {
  string table = 1;
  int64 num_samples = 2;
  int64 timeout_ms = 3;          // rate_limiter_timeout
  bool emit_timesteps = 4;
}
message ShmReleaseRequest {
  repeated uint64 offsets = 1;   // 读完的 SHM 偏移, server 递减引用计数
}
message HelloRequest {
  int32 client_pid = 1;
  uint32 protocol_version = 2;
}
```

#### S→C 响应（server 写，client 读）

| type 值 | msg_type 名 | body proto | 对应 |
| --- | --- | --- | --- |
| 101 | `WELCOME` | `WelcomeResponse{pool_shm_name, c2s_shm_name, s2c_shm_name, server_info}` | bootstrap 响应 |
| 102 | `INSERT_ACK` | `InsertAck{repeated uint64 keys, repeated uint64 offsets_to_release}` | InsertStreamResponse + SHM 偏移回收 |
| 103 | `SAMPLE_RESP` | `ShmSampleResponse`（见下） | SampleStream |
| 104 | `ERROR` | `ShmError{code, message, request_seq}` | 统一错误 |
| 105 | `MUTATE_ACK` | `MutatePrioritiesResponse`（复用） | |
| 106 | `RESET_ACK` | `ResetResponse`（复用） | |
| 107 | `CHECKPOINT_RESP` | `CheckpointResponse`（复用） | |
| 108 | `SERVER_INFO_RESP` | `ServerInfoResponse`（复用） | |

```protobuf
message ShmSampleResponse {
  repeated ShmSample samples = 1;
}
message ShmSample {
  SampleInfo info = 1;                 // 复用现有 proto(key/priority/probability...)
  repeated ShmColumn columns = 2;      // 每列一个
}
message ShmColumn {
  uint64 shm_offset = 1;       // 成品字节偏移(server 预切片后)
  uint64 length = 2;
  TensorSpecProto spec = 3;    // dtype + shape
  bool squeeze = 4;            // 对齐 FlatTrajectory.Column.squeeze
}
message WelcomeResponse {
  string pool_shm_name = 1;
  string c2s_shm_name = 2;
  string s2c_shm_name = 3;
  ServerInfoResponse server_info = 4;  // 含各表 signature, 供 writer 校验
}
message InsertAck {
  repeated uint64 keys = 1;                 // 已插入 item key
  repeated uint64 offsets_to_release = 2;   // insert 用完的 chunk SHM 偏移, client 可释放
}
message ShmError {
  enum Code {
    UNKNOWN = 0;
    DEADLINE_EXCEEDED = 1;   // 映射 Python DeadlineExceededError
    INVALID_ARGUMENT = 2;
    NOT_FOUND = 3;           // 表/chunk 不存在
    RESOURCE_EXHAUSTED = 4;  // 字节池满(理论上阻塞不报错, 保留)
    INTERNAL = 5;
  }
  Code code = 1;
  string message = 2;
  uint64 request_seq = 3;   // 对应请求的 producer_seq, client 匹配
}
```

### 8.4 读写算法（SPSC，无锁）

**生产者写一条消息**（client 写 C→S / server 写 S→C）：

```c
Status Write(Ring* r, uint16_t msg_type, absl::string_view body) {
  // 1. 算需要几个槽
  size_t slot_body_cap = r->slot_size - sizeof(SlotHeader);
  size_t num_slots = (body.size() + slot_body_cap - 1) / slot_body_cap;
  if (num_slots == 0) num_slots = 1;
  if (num_slots > r->capacity) return RESOURCE_EXHAUSTED;  // 消息超 ring 容量

  // 2. 等待连续 num_slots 空槽(满则阻塞, 见 §8.7 backpressure)
  // SPSC: producer 只读 consumer_seq(acquire), 等其推进腾出空间
  uint64_t first_seq = r->producer_seq.load(std::memory_order_relaxed);
  while (first_seq - r->consumer_seq.load(std::memory_order_acquire)
         > r->capacity - num_slots) {
    // ponytail: 忙等或 sched_yield; 无信号量避免崩溃泄漏(§4.2)
    sched_yield();
  }

  // 3. 逐槽写入
  for (size_t i = 0; i < num_slots; i++) {
    uint64_t seq = first_seq + i;
    Slot* s = &r->slots[seq & r->capacity_mask];
    size_t chunk_len = std::min(slot_body_cap, body.size() - i * slot_body_cap);
    s->header.msg_type = (i == 0) ? msg_type : 0;
    s->header.flags = (i + 1 < num_slots) ? HAS_CONT : 0;
    if (i > 0) s->header.flags |= IS_CONT;
    s->header.body_len = chunk_len;
    memcpy(s->body, body.data() + i * slot_body_cap, chunk_len);
    // 关键: 先写完 body, 最后置 seq 发布(release)。consumer 见 seq==期望值
    // 才读 body, acquire 配对保证 consumer 看到上面所有写入。
    // 不需要先清 0: producer 独占写, consumer 只在 seq==consumer_seq 时读,
    // 此槽上一轮的 seq < consumer_seq, consumer 不会误读旧值。
    std::atomic_store_explicit(
        reinterpret_cast<std::atomic<uint64_t>*>(&s->header.seq),
        seq, std::memory_order_release);
  }
  r->producer_seq.store(first_seq + num_slots, std::memory_order_release);
  return OK;
}
```

**消费者读一条消息**（server 读 C→S / client 读 S→C）：

```c
Status Read(Ring* r, uint16_t* msg_type, std::string* body) {
  uint64_t seq = r->consumer_seq.load(std::memory_order_relaxed);
  Slot* s = &r->slots[seq & r->capacity_mask];
  // consumer 读 seq 判就绪; 但需保证读到 seq 后能看到 producer 对该槽 body 的写入。
  // seq 是普通 uint64_t, 靠 producer_seq 的 release 发布可见性: producer 写完槽后
  // store(producer_seq, release), 但 consumer 不读 producer_seq。
  // 正确性靠 SlotHeader.seq 自身的 release/acquire: 此处用 atomic 读 seq。
  uint64_t slot_seq = std::atomic_load_explicit(
      reinterpret_cast<const std::atomic<uint64_t>*>(&s->header.seq),
      std::memory_order_acquire);
  if (slot_seq != seq) return NOT_READY;  // 生产者未写到

  // 收集跨槽消息
  *msg_type = s->header.msg_type;
  body->clear();
  size_t i = 0;
  while (true) {
    body->append(s->body, s->header.body_len);
    bool has_cont = s->header.flags & HAS_CONT;
    if (!has_cont) break;
    i++;
    s = &r->slots[(seq + i) & r->capacity_mask];
    uint64_t cont_seq = std::atomic_load_explicit(
        reinterpret_cast<const std::atomic<uint64_t>*>(&s->header.seq),
        std::memory_order_acquire);
    // IS_CONT 槽的 seq 应连续; 若不连续说明生产者中断(不应发生), 报错
    if (cont_seq != seq + i) return INTERNAL;
  }
  // 消费完所有槽, 推进 consumer_seq(release), 让 producer 看到空间释放
  r->consumer_seq.store(seq + i + 1, std::memory_order_release);
  return OK;
}
```

- **关键正确性**：写方先填字段最后置 `seq`（release），读方先读 `seq` 确认就绪
  再读字段（acquire）。这是 SPSC 无锁 ring 的标准 release/acquire 配对。
- **无信号量**：阻塞靠忙等/yield。避免进程崩溃后信号量泄漏（§4.2 决策）。

### 8.5 insert 的 chunk 多列处理

一个 `ChunkData` 含多列 tensor（`repeated TensorProto tensors`）。SHM 下一个
chunk 的多列字节布局：

- **方案**：一个 `ShmChunkRef` 对应一个 chunk，但其 `shm_offset/length` 指向的
  SHM 区域是**多列拼接**的连续字节（每列紧跟上一列）。`spec` 字段改为
  `repeated TensorSpecProto specs`（每列一个），各列长度由 `spec.shape + dtype`
  算出。server 端按 spec 拆列，调 `CompressTensorAsProto` 各列压缩，组装
  `ChunkData.data.tensors`。

```protobuf
// 修正 ShmChunkRef
message ShmChunkRef {
  uint64 chunk_key = 1;
  uint64 shm_offset = 2;
  uint64 total_length = 3;          // 多列拼接总长
  repeated TensorSpecProto specs = 4;  // 每列 dtype+shape, 按顺序
  SequenceRange sequence_range = 5;
  bool delta_encoded = 6;
}
```

server 侧伪代码：

```c
Status HandleInsert(ShmInsertRequest req, ShmConnection* conn) {
  for (auto& ref : req.chunks) {
    ChunkData chunk;
    chunk.set_chunk_key(ref.chunk_key());
    *chunk.mutable_sequence_range() = ref.sequence_range();
    uint64 off = ref.shm_offset();
    for (auto& spec : ref.specs()) {
      int64_t bytes = NumElems(spec) * SizeOf(spec.dtype());
      TensorBuffer buf(spec, string_view(pool_ + off, bytes));
      TensorProto* t = chunk.mutable_data()->add_tensors();
      CompressTensorAsProto(buf, t);  // 现有函数
      off += bytes;
    }
    // 走现有路径: chunks 存 ChunkStore, items 走 InsertOrAssignAsync
    ...
  }
}
```

### 8.6 Bootstrap 握手时序（udsocket + 初始 ring）

bootstrap 阶段 client 还没有 C→S/S→C ring（ring 是握手产物）。所以握手本身走
udsocket 字节流，不走 ring：

```
1. server 启动:
   - shm_open 创建 pool 段 /reverb_shm_pool_<pid>, ftruncate 设容量, mmap
   - unix socket bind+listen 在 /tmp/reverb_shm_<pid>.sock (路径传给 client)

2. client 连接:
   - connect udsocket
   - 发 HelloRequest{client_pid, protocol_version} (length-delimited proto)

3. server 收 Hello:
   - 为该 client 创建两条 ring 段: /reverb_shm_c2s_<server_pid>_<client_pid>,
     /reverb_shm_s2c_<server_pid>_<client_pid>
   - ftruncate 各 ring 段(RingHeader + capacity*slot_size)
   - mmap, 初始化 RingHeader
   - 回 WelcomeResponse{pool_shm_name, c2s_shm_name, s2c_shm_name, server_info}
     (含各表 signature, 供 client writer 校验, 对齐 gRPC InitializeConnection)

4. client 收 Welcome:
   - shm_open + mmap 三段(pool, c2s, s2c)
   - 之后所有控制消息走 ring, udsocket 仅保留用于断连检测(§8.8)
```

> `ponytail:` ring 段名含 server_pid + client_pid，避免多 server/多 client 冲突。
> 断连检测靠 udsocket，不靠 ring（ring 无法感知对端进程消失）。

### 8.7 backpressure 与阻塞语义

ring 满时的阻塞行为，对齐 §S10（语义对齐 gRPC）：

- **C→S 满**（client 写太快，server 处理不及）：`Write` 在 client 侧阻塞。
  等价于 gRPC stream 的流控。client writer 的 `local_can_insert_more_` /
  `num_items_in_flight_` 自然受此约束。
- **S→C 满**（server 响应太快，client 读不及）：`Write` 在 server 侧阻塞。
  server dispatch 线程阻塞在此 client 的 S→C，其他 client 仍可处理（除非
  dispatch 单线程——见下）。
- **dispatch 单线程风险**：§4.2 说 server 单线程轮询所有 client。若某 client
  S→C 满导致 dispatch 阻塞，会饿死其他 client。**对策**：dispatch 线程对每个
  client 的 S→C 写采用非阻塞尝试，失败则跳过该 client 继续轮询下一个，待该
  client ring 有空间再回写（响应暂存 per-client outbox 队列）。这样 dispatch
  不被慢 client 阻塞。
- **字节池满**（§S14）：server 在分配 SHM 偏移时阻塞。同样采用非阻塞尝试 +
  暂存：sample 请求若池满，暂存该请求，先处理其他请求，待池有空间再完成。

### 8.8 断连检测与崩溃恢复

- **正常关闭**：client 发 `CLOSE` 消息（C→S），server 收到后释放该 client 所有
  未 release 偏移，销毁 ring 段（shm_unlink），关闭 udsocket。
- **client 崩溃**：udsocket 上 server `recv` 返回 EOF/ECONNRESET。server 判定
  client 断，执行集中释放：遍历该 client 的 `outstanding_offsets_` 集合，逐个
  递减引用计数并回收，shm_unlink 两条 ring 段。
- **server 崩溃**：client 的 udsocket `recv` 返回 EOF。client 侧报
  `ConnectionError`（所有在途请求失败）。client 持有的 SHM mmap 指针失效。
- **outstanding_offsets_ 追踪**：server 为每 client 维护
  `flat_hash_set<uint64>`，每次 sample 响应发出的偏移加入，每次收到 RELEASE
  移除。崩溃时遍历此集合集中释放。

### 8.9 请求-响应匹配

SHM 是异步 ring，不像 gRPC 一请求一响应同步。匹配靠 `request_seq`：

- client 每发一个请求，记录其 `producer_seq` 作为 `request_seq`。
- 响应（`SAMPLE_RESP`/`INSERT_ACK`/`ERROR` 等）的 body proto 里带 `request_seq`
  字段（或在 `ShmError` 里显式带；成功响应可隐式按顺序匹配，因 SPSC 保序）。
- **简化**：SPSC 保序，同一 client 的响应顺序与请求顺序一致。client 可用 FIFO
  队列按序匹配，不必每响应带 seq。仅 `ERROR` 显式带 `request_seq` 便于定位。

> `ponytail:` 靠 SPSC 保序做隐式匹配，省掉每响应的 seq 字段。若未来要支持
> 乱序响应（如多 worker 并发处理），再加显式 seq。
