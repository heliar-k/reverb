# SHM 传输层实现规格说明书

> 本文档由 [numpy-shm-design.md](../design/numpy-shm-design.md) 转化而来，是 SHM
> 传输层的**实现现状规格**：文件结构、BUILD 规则、类声明、proto、pybind、测试策略
> 与边界情况，均以 `reverb/cc/shm/` 当前代码为准。文档经历了 plan → 实现两个阶段：
> 早期"规划"表述（未实现的功能、预留字段、设想架构）已随实现同步删除或改写，若发现
> 与代码不符，以代码为准并修本文档。面向已阅读设计文档的开发者。

## 目录

- [1. 文件布局](#1-文件布局)
- [2. 构建规则（BUILD）](#2-构建规则build)
- [3. C++ 接口与类](#3-c-接口与类)
  - [3.1 Ring 协议](#31-ring-协议)
  - [3.2 字节池](#32-字节池)
  - [3.3 Bootstrap](#33-bootstrap)
  - [3.4 Server 侧](#34-server-侧)
  - [3.5 Client 侧](#35-client-侧)
- [4. Proto 定义](#4-proto-定义)
- [5. pybind 绑定](#5-pybind-绑定)
- [6. Python 层](#6-python-层)
- [7. 线程模型与数据流](#7-线程模型与数据流)
- [8. 测试策略](#8-测试策略)
- [9. 边界情况与错误处理](#9-边界情况与错误处理)
- [附录：关键实现决策说明](#附录关键实现决策说明)

---

## 1. 文件布局

SHM 传输层代码集中在 `reverb/cc/shm/`：

```
reverb/cc/shm/
├── BUILD                       # Bazel 构建规则（10 个测试目标）
├── ring.h / ring.cc            # ShmCtrlRing：SPSC ring buffer（POSIX shm 段）
├── ring_test.cc                # ring 单元测试
├── byte_pool.h / byte_pool.cc  # ShmBytePool：slab 档位 POSIX shm 分配器
├── byte_pool_test.cc           # 字节池单元测试
├── byte_pool_echo_test.cc      # pool+ring 双进程回声（跨进程读写校验）
├── bootstrap.h / bootstrap.cc  # ShmBootstrapServer：udsocket 握手 + 段名生成
├── bootstrap_test.cc           # 握手单测
├── echo_test.cc                # 最简握手 + 回声端到端
├── shm_connection.h / shm_connection.cc  # ShmConnection：五段 mmap 的 RAII 聚合
├── shm_server.h / shm_server.cc          # ShmServer：dispatch 线程 + per-client 状态
├── shm_client.h / shm_client.cc          # ShmClient + ShmSampler（pybind 就绪）
├── shm_sample_test.cc          # sample 路径集成测试
├── shm_insert_test.cc          # insert 路径集成测试
├── shm_crash_test.cc           # 崩溃恢复测试
├── shm_checkpoint_test.cc      # checkpoint 集成测试
├── shm_wakeup_test.cc          # eventfd 唤醒（ticket 03）测试
└── shm_protocol.proto          # SHM 专有 proto 消息
```

> 注意：不存在独立的 `shm_server_test.cc` / `shm_client_test.cc`（plan 时代设想），
> 端到端覆盖由上面 7 个 SHM 集成测试分担（见 §8）。

## 2. 构建规则（BUILD）

`reverb/cc/shm/BUILD` 内的目标（用 `reverb_cc_library` / `reverb_cc_test` /
`reverb_cc_proto_library` 封装）：

| 目标 | 类型 | 说明 |
| --- | --- | --- |
| `shm_protocol_cc_proto` | proto | deps：`//reverb/cc:schema_cc_proto`、`patterns_cc_proto`、`reverb_service_cc_proto`、`//third_party/reverb_tensor:reverb_tensor_cc_proto` |
| `ring` | library | deps：`:shm_protocol_cc_proto` + absl（status/strings/span） |
| `byte_pool` | library | `alwayslink = 1`（libpybind.so 静态链接需要间接符号）；deps：absl flat_hash_map/status/synchronization/span |
| `bootstrap` | library | deps：`:shm_protocol_cc_proto` + absl |
| `shm_connection` | library | 聚合 ring + byte_pool |
| `shm_server` | library | deps：bootstrap/byte_pool/ring/shm_connection/proto + `//reverb/cc:table`、`sampler`、`chunk_store`、`tensor_compression`、`support:task_executor`、`checkpointing` 等 |
| `shm_client` | library | deps：bootstrap/byte_pool/ring/shm_connection/proto + `//reverb/cc:chunker`、`sampler`、`trajectory_writer`、`structured_writer`、`writer` 等 |
| `ring_test` / `byte_pool_test` / `byte_pool_echo_test` / `bootstrap_test` / `echo_test` / `shm_sample_test` / `shm_insert_test` / `shm_crash_test` / `shm_checkpoint_test` / `shm_wakeup_test` | test | 10 个测试目标（见 §8） |

`reverb/pybind.cc` 的 pybind 目标依赖 `//reverb/cc/shm:shm_server` 与
`//reverb/cc/shm:shm_client`（§5）。

## 3. C++ 接口与类

### 3.1 Ring 协议

```cpp
// reverb/cc/shm/ring.h

namespace deepmind {
namespace reverb {
namespace shm {

inline constexpr uint64_t kRingMagic = 0x524556524253484DULL;  // "REVRBSHM"
inline constexpr uint32_t kRingVersion = 1;
inline constexpr size_t kDefaultSlotSize = 256;   // bytes（含 SlotHeader）
inline constexpr size_t kDefaultCapacity = 1024;  // slots（256KB/段）

// 段头，恰好 128 字节 = 2 条 cache line。
// head（生产者）在 line 0，tail（消费者）在 line 1，防 false sharing。
// 均从 1 开始；seq 0 表示槽从未写过。显式字段布局 + pad，不用 alignas(64)
// 成员（那会撑大结构体）。
struct RingHeader {
  // Line 0 (offset 0..63)。
  uint64_t magic = kRingMagic;
  uint32_t version = kRingVersion;
  uint32_t capacity = kDefaultCapacity;   // 槽位数（2 的幂）
  uint32_t slot_size = kDefaultSlotSize;  // 每槽字节数（含 SlotHeader）
  // ticket 03（原 `reserved`）：server-asleep 标志，每个 C→S ring 一份。
  // server dispatch 线程在进入 poll() 阻塞前对每条 c2s ring 置 1（seq_cst）；
  // client 的 WriteBlocking 在成功写入后 load 该标志（seq_cst），仅当置位时
  // 才向 control_fd 发 1 字节唤醒——server 醒着时热路径零新增 syscall。
  // 双端都 seq_cst（非 release/acquire）：这是经典 flag-then-recheck 模式，
  // 发布标志与重读 ring 之间发生 store→load 重排就是 missed-wakeup 漏洞。
  // 混版本退化安全：旧 client 不读标志（server 50ms poll 兜底）、
  // 旧 server 不置位（client 永不发字节）。
  std::atomic<uint32_t> server_asleep{0};
  uint64_t capacity_mask = kDefaultCapacity - 1;
  std::atomic<uint64_t> head{1};  // 生产者：下一个要写的槽的 seq
  uint64_t pad0[3];               // 填满 line 0
  // Line 1 (offset 64..127)。
  std::atomic<uint64_t> tail{1};  // 消费者：下一个要读的槽的 seq
  uint64_t pad1[7];               // 填满 line 1
};
static_assert(sizeof(RingHeader) == 128, "RingHeader 必须恰好 128 字节（2 cache lines）");

// 槽头，恰好 16 字节。seq 是写入时生产者的 head 值；消费者在
// seq == 期望 tail 时判定槽有效。seq 0 = 从未写入。
struct SlotHeader {
  uint64_t seq = 0;      // 经 Ring::Write/Read 的 atomic 操作发布
  uint16_t msg_type = 0; // MsgType 枚举值（线上为裸 uint16）
  uint16_t flags = 0;    // bit0: HAS_CONTINUATION; bit1: IS_CONTINUATION
  uint32_t body_len = 0; // 本槽 body 字节数（<= slot_size - 16）
};
static_assert(sizeof(SlotHeader) == 16, "SlotHeader 必须恰好 16 字节");

inline constexpr uint16_t kFlagHasContinuation = 0x0001;
inline constexpr uint16_t kFlagIsContinuation = 0x0002;

// SPSC ring：一端 Create（server/owner），另一端 Open（client）。
// Write 满时阻塞（忙等 sched_yield）；Read 非阻塞，无消息时返回
// NotFoundError("NOT_READY")。需要阻塞语义的调用方策略在 ShmConnection（R5）。
class Ring {
 public:
  // owner 侧：创建 + ftruncate 新 SHM 段，初始化 header。
  static absl::StatusOr<Ring> Create(std::string shm_name,
                                     uint32_t capacity = kDefaultCapacity,
                                     uint32_t slot_size = kDefaultSlotSize);
  // client 侧：打开已有 SHM 段（读写）。
  static absl::StatusOr<Ring> Open(std::string shm_name);

  // 写一条消息（可能跨多槽）。阻塞（sched_yield）直到有足够连续空槽；
  // 消息比整个 ring 大返回 RESOURCE_EXHAUSTED。
  absl::Status Write(MsgType msg_type, absl::Span<const char> payload);

  // 非阻塞写（ticket ⑥ / design §8.7）：槽位不够时立即返回
  // ResourceExhaustedError("RING_FULL")，不动 head——调用方
  // （ShmServer::EnqueueS2C）把消息暂存 outbox 下轮重试，不阻塞 dispatch。
  absl::Status TryWrite(MsgType msg_type, absl::Span<const char> payload);

  // 读一条消息，重组跨槽片段。非阻塞：无消息时立即返回
  // NotFoundError("NOT_READY")；首槽后缺 continuation 槽是损坏信号，
  // 返回 InternalError。
  absl::Status Read(MsgType* msg_type, std::string* payload);

  // ticket 03：有消息可读（消费者视角：下一槽 seq 已发布）。与 Read 同一
  // acquire 配对；供 ShmServer 进入 poll() 前的 quiescence 检查使用。
  bool HasData() const;

  // ticket 03：共享内存中的 asleep 标志（见 RingHeader）。server 写、client 读，
  // 双端 seq_cst。
  std::atomic<uint32_t>* server_asleep() const;

  uint32_t capacity() const;
  uint32_t slot_size() const;
  const std::string& shm_name() const;

  // 给定几何的整段字节数（header + slots）。
  static size_t TotalBytes(uint32_t capacity, uint32_t slot_size);
};

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
```

**实现要点**（design §8.4）：

- `Write`/`WriteSlots`：先填 body，最后以 release 语义发布槽 `seq`，再以 release
  推进 `head`。消费者 acquire 读槽 `seq` 确认就绪后再读 body——标准 release/acquire
  配对。`WriteSlots` 槽 0 最后发布（ticket「SHM 并发正确性」），消除多槽消息
  "continuation slot missing" 致命错误。
- `Read` 非阻塞：`slot.seq != tail` 即 `NOT_FOUND`。
- 阻塞策略（轮询 + 存活探测 + 60s 硬上限）在 `ShmConnection::WriteBlocking` /
  `ReadBlocking` 等调用点，Ring 本身不实现（R5）。

### 3.2 字节池

```cpp
// reverb/cc/shm/byte_pool.h

namespace deepmind {
namespace reverb {
namespace shm {

// 默认 slab 档位（bytes）：64B..4MB，覆盖单列 tensor 至 ~1M float32。
inline constexpr size_t kDefaultSlabSizes[] = {
    64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304,
};
inline constexpr size_t kDefaultBlocksPerSlab = 256;

inline constexpr uint64_t kPoolMagic = 0x52455652424F4F4CULL;  // "REVRBOOL"
inline constexpr uint32_t kPoolVersion = 1;

// 段布局（单一连续 mmap）：
//   [PoolHeader]                     // 固定，offset 0
//   [SlabMeta * slab_count]          // 每档一份，存于 SHM
//   [slab 0 数据块 ...][slab 1 数据块 ...] ...
// 每档 free list 是单链表：空闲块首 8 字节存下一空闲块偏移；
// free_head_offset == kNullOffset 表示空表。
struct PoolHeader {
  uint64_t magic = kPoolMagic;
  uint32_t version = kPoolVersion;
  uint32_t slab_count = 0;
  uint64_t total_size = 0;  // 段字节数
  uint64_t reserved = 0;
};
static_assert(sizeof(PoolHeader) == 32);

struct SlabMeta {
  uint64_t block_size = 0;        // 档位字节数
  uint64_t block_count = 0;       // 块数
  uint64_t region_offset = 0;     // 数据区起点（相对段基址）
  uint64_t free_head_offset = 0;  // 空闲链表头（kNullOffset = 空）
};
static_assert(sizeof(SlabMeta) == 32);

inline constexpr uint64_t kNullOffset = UINT64_MAX;

// slab 分配 POSIX shm 字节池。server Create（O_CREAT|O_EXCL）且是唯一分配者；
// client Open 为 RW（决策 C4）但只用 At() 读写 server 授予的偏移。
// Allocate 在档位耗尽时快速失败返回 RESOURCE_EXHAUSTED——阻塞会死锁 server
// 的单 dispatch 线程（Allocate 与 Deallocate 的唯一调用方）。
// Allocate/Deallocate/Ref/Unref/ReleaseAll 仅 server 可调。
class ShmBytePool {
 public:
  // server 侧：创建 + ftruncate + 初始化 slab 元数据与 free list。
  // slab_sizes 空时用 kDefaultSlabSizes；自定义须严格升序且每档 >= 8 字节
  // （空闲块首 8 字节存下一偏移）；blocks_per_slab 须 > 0。违规在创建任何
  // 段之前返回 InvalidArgumentError（ticket 02）。
  static absl::StatusOr<ShmBytePool> Create(
      const std::string& shm_name,
      absl::Span<const size_t> slab_sizes = {},
      size_t blocks_per_slab = kDefaultBlocksPerSlab);

  // client 侧：打开已有段（RW，C4），校验 magic/version。只可用 At()。
  static absl::StatusOr<ShmBytePool> Open(const std::string& shm_name);

  // 从最小够用档位分配 `bytes`。档位 free list 空则返回 RESOURCE_EXHAUSTED
  // （从不阻塞，见类注释）。SERVER-ONLY。返回相对段基址的偏移。
  absl::StatusOr<uint64_t> Allocate(size_t bytes);

  // 归还偏移到所属档位 free list。SERVER-ONLY。
  void Deallocate(uint64_t offset);

  // 偏移处的指针（client 读写 / server 读写）。
  void* At(uint64_t offset);
  const void* At(uint64_t offset) const;

  // 偏移所属档位的块大小（测试/诊断用）。
  size_t block_size_at(uint64_t offset) const;

  const std::string& name() const;
  size_t size() const;

  // 引用计数（server 进程内）。Ref 递增；Unref 返回是否归零（归零后调用方
  // Deallocate）。决策 C3：server 为 sample 成品字节 memcpy 进池时设 refcount=1；
  // client 读完发 RELEASE，server Unref，归零即回收。
  void Ref(uint64_t offset);
  bool Unref(uint64_t offset);

  // 崩溃恢复批量释放（ticket ⑥）：逐个递减 refcount，归零的 Deallocate。
  void ReleaseAll(const std::vector<uint64_t>& offsets);

 private:
  // PickSlab：最小够用档位（smallest-fit 扫描，要求档位严格升序）。
  // PopFree/PushFree：free list 头弹出/压入。
  std::string shm_name_;
  void* base_ = nullptr;        // mmap 基址
  size_t pool_size_ = 0;
  PoolHeader* header_ = nullptr;
  SlabMeta* slabs_ = nullptr;   // slab_count 项，段内
  size_t slab_count_ = 0;
  bool owner_ = false;          // true => 析构时 shm_unlink

  // ponytail: refcount 用进程内 flat_hash_map（不在 SHM）——server 是唯一
  // 分配者（C4），refcount 不跨进程。升级路径：第二个分配者出现时换
  // 段内 per-block refcount 数组 + 位图（spec §3.2）。
  absl::flat_hash_map<uint64_t, int> refcounts_;

  // 防 stray 跨线程 Deallocate 的 free list 守卫。SERVER-ONLY。
  absl::Mutex mu_;
};

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
```

**实现要点**：

- server `Create`：`shm_open(O_CREAT|O_RDWR|O_EXCL)` → `ftruncate` → `mmap` →
  初始化 `PoolHeader`/`SlabMeta`/free list。
- client `Open`：`shm_open(O_RDWR)` → `mmap` → 校验 `kPoolMagic`/`kPoolVersion`。
- `At(offset) = base_ + offset`，server/client 两进程映射同一段，偏移语义一致。
- **池满语义**：v1 实现是**快速失败**（`RESOURCE_EXHAUSTED`），不是阻塞等待
  （design §8.7 的阻塞设想未采用）——阻塞在单 dispatch 线程上会死锁。client
  收到档位耗尽错误（瞬时 `ResourceExhausted`）；超档请求是永久
  `InvalidArgument`（ticket 02，客户端可据此区分）。

### 3.3 Bootstrap

```cpp
// reverb/cc/shm/bootstrap.h

namespace deepmind {
namespace reverb {
namespace shm {

inline constexpr uint32_t kProtocolVersion = 1;  // Hello/Welcome 版本校验

// server 侧：监听 udsocket。Create 先 unlink 残留 socket 文件（R7：PID 复用
// 会留陈旧 .sock），再 bind + listen。
class ShmBootstrapServer {
 public:
  static absl::StatusOr<ShmBootstrapServer> Create(const std::string& socket_path);

  // 阻塞至 client 连接。返回 (client_fd, client_pid)；client_pid 经 SO_PEERCRED
  // 读取。调用方持有 fd 并负责 close。
  absl::StatusOr<std::pair<int, int>> Accept();

  // 监听 fd：供 dispatch 线程 poll 新连接（不阻塞 accept）。
  int listen_fd() const;
  const std::string& socket_path() const;
};

// A3/D-格式段名生成。server 独占生成权（spec A3）。`server_token` = udsocket
// 路径 + per-server epoch（PID + 墙上时钟纳秒，ticket #7）；token 消毒
// （非 alnum -> '_'）保证 POSIX shm 名合法。按 socket 路径（而非仅 PID）命名
// 让同进程多 ShmServer 共存（scan #12）；epoch 让同路径崩溃重启的 server 不与
// 旧客户端活段碰撞。client 从 Welcome 学段名，不自算。决策 D 每流一对：
//   /reverb_shm_pool_<token>
//   /reverb_shm_insert_c2s_<token>_<client_pid>
//   /reverb_shm_insert_s2c_<token>_<client_pid>
//   /reverb_shm_sample_c2s_<token>_<client_pid>
//   /reverb_shm_sample_s2c_<token>_<client_pid>
struct ShmSegmentNames {
  std::string pool;
  std::string insert_c2s;
  std::string insert_s2c;
  std::string sample_c2s;
  std::string sample_s2c;
};
ShmSegmentNames MakeShmNames(absl::string_view server_token, int client_pid);
std::string MakePoolShmName(absl::string_view server_token);

// 经 client_fd 发 WelcomeResponse（length-delimited：4 字节大端长度前缀 + proto）。
absl::Status SendWelcome(int client_fd, const WelcomeResponse& welcome);

// 经 client_fd 收 HelloRequest（length-delimited）。`timeout` 约束整个握手
// 等待（per-byte deadline，防 trickling sender）；TryAccept 传小上限——它跑在
// 单 dispatch 线程上，无限读会被 connect-and-stall 客户端卡死全体服务。
absl::StatusOr<HelloRequest> RecvHello(
    int client_fd, absl::Duration timeout = absl::InfiniteDuration());

// 校验客户端协议版本，不匹配返回 InvalidArgumentError。
absl::Status CheckProtocolVersion(uint32_t client_version);

// 握手结果：Welcome + 仍打开的 udsocket fd（调用方持有，连接期内保持打开，
// ticket ⑥：server poll 该 fd 的 POLLHUP/EOF 检测 client 崩溃，§8.8）。
struct ClientBootstrapResult {
  WelcomeResponse welcome;
  int fd = -1;
};

// 连接 + 发 Hello + 收 Welcome + 返回打开的 fd（不关闭）。
// ShmClient::Connect 用（fd 存为 liveness 信号）。
absl::StatusOr<ClientBootstrapResult> ClientBootstrapWithFd(
    const std::string& socket_path, int client_pid);

// 一次性握手：连接、交换 Hello/Welcome、关闭 fd（无需持久 liveness fd 的
// 调用方，如 echo 测试）。ponytail: 保留以不改既有测试/调用方。
absl::StatusOr<WelcomeResponse> ClientBootstrap(const std::string& socket_path,
                                                int client_pid);

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
```

### 3.4 Server 侧

```cpp
// reverb/cc/shm/shm_server.h

namespace deepmind {
namespace reverb {
namespace shm {

// per-client 状态。dispatch 线程是 outstanding_offsets_ 与 outbox 的唯一变更者
// （单线程 dispatch，决策 A1/R11），实际无需加锁；保留 mutex 与 spec §3.4
// 声明一致并为未来 per-client dispatch 拆分留余地。决策 D：outbox 按流分开
// （insert/sample），各自 s2c ring 的重试队列保持同质。
struct ClientState {
  int fd = -1;                       // udsocket fd（断连时关闭）
  int client_pid = 0;
  ShmConnection conn;                // 4 条 ring（insert+sample 的 c2s/s2c）；
                                     // pool 在 server 侧由 ShmServer 独占，不用此句柄
  internal::flat_hash_set<uint64_t> outstanding_offsets_;  // C3：崩溃恢复（ticket ⑥）

  // S→C 暂存队列：ring 满时塞不下、由非阻塞 TryWrite 失败的消息。每轮 dispatch
  // FlushOutbox 重试。决策 D：insert/sample 分开，一条满不挡另一条。
  absl::Mutex insert_outbox_mu;
  std::vector<std::pair<uint16_t, std::string>> insert_outbox;
  absl::Mutex sample_outbox_mu;
  std::vector<std::pair<uint16_t, std::string>> sample_outbox;

  // insert 回调保活（ticket ④）：InsertOrAssignAsync 只存 callback 的 weak_ptr，
  // 表 worker 异步触发（HandleInsert 返回之后）。shared_ptr 必须活到回调触发，
  // 攒在这里、聚合 InsertAck 入队后清空。每项带所属 INSERT 请求 id，完成/失败
  // 请求只清自己的保活项（绝不整向量清空——其他在飞请求会丢 ACK）。
  struct PendingInsertCallback {
    uint64_t request_id;
    std::shared_ptr<Table::InsertCallback> callback;
  };
  std::vector<PendingInsertCallback> pending_insert_callbacks;
  uint64_t next_insert_request_id = 0;  // 仅 dispatch 线程触碰，无需锁

  // ticket ⑩ 死锁修复（方向 A）：异步 sample 完成回调在表 worker 线程触发，
  // 不能碰 pool_/outstanding_offsets_（单线程 dispatch 不变式）。回调把
  // SampledItem + 路由元数据攒进此队列，dispatch 线程每轮 DrainPendingSamples
  // 取出做 unpack + pool + 写 SAMPLE_RESP。
  struct PendingSample {
    ShmSampleRequest req;
    absl::Status status;       // 表 worker 结果（含 DeadlineExceeded）
    Table::SampledItem item;   // 成功时的采样项
  };
  absl::Mutex pending_samples_mu;
  std::vector<PendingSample> pending_samples;
  std::vector<std::shared_ptr<Table::SamplingCallback>> pending_sample_callbacks;

  // ticket ⑥：client 显式 CLOSE 时置位；dispatch 循环的 IsClientDead 检查
  // 下一轮走 HandleDisconnect。atomic：CloseClientFdForTest 在 dispatch 线程外写。
  std::atomic<bool> close_requested{false};
};

// ShmServer 拥有全部表（ticket ⑨：按表名路由）、一个 ShmBytePool（唯一分配者，
// C4）和一个 bootstrap udsocket。单 dispatch 线程轮询 listen socket 收新 client，
// 再非阻塞读每个 client 的 C→S ring：
//   SAMPLE → FindTable(req.table) → Table::EnqueSampleRequest（异步，ticket ⑩：
//     dispatch 不阻塞于 rate limiter）→ 完成回调攒进 pending_samples →
//     DrainPendingSamples 在 dispatch 线程做 UnpackChunkColumnAndSlice →
//     memcpy 成品字节进池（refcount=1, C3）→ S→C 写 SAMPLE_RESP（非阻塞，
//     满则暂存 outbox，§8.7）；
//   RELEASE → Unref 各偏移，归零 Deallocate；
//   INSERT/ALLOCATE → §3.4 下方。
class ShmServer {
 public:
  // `tables` 非空且名字唯一（此处校验；Python `Server` 也校验，C++ 自卫）。
  // `checkpointer` 可选（ticket ⑪）：提供时 HandleCheckpoint 保存全部表并
  // 返回路径；为 null 时返回 FailedPreconditionError（镜像 InProcessClient）。
  // ticket 02：`slab_sizes` / `pool_blocks_per_slab` 调池几何（单连接高水位 =
  // sum(slab × blocks)）；空/0 用默认（与 ticket 02 前逐字节一致）。
  static absl::StatusOr<std::unique_ptr<ShmServer>> Create(
      std::vector<std::shared_ptr<Table>> tables,
      const std::string& socket_path,
      std::shared_ptr<Checkpointer> checkpointer = nullptr,
      absl::Span<const size_t> slab_sizes = {},
      size_t pool_blocks_per_slab = 0);

  // 启动 dispatch 线程。
  absl::Status Start();

  // 停止 dispatch 线程，停全部表（Table::Stop = Close + join worker + drain
  // callback executor——保证此后无表回调再触发，评审 #1），再清理客户端、
  // unlink SHM 段。
  void Stop();

  const std::string& socket_path() const;

  // ticket ⑥ 测试专用：模拟 client 连接消失（server 崩溃/fd 关闭），让
  // CLIENT 的 liveness control_fd 看到 EOF。置 close_requested 标志，dispatch
  // 线程下一轮观察后走正常 HandleDisconnect——关闭动作留在 dispatch 线程，
  // 避免调用线程与 dispatch 循环竞争 double-close。
  void CloseClientFdForTest();

  // ticket 01 测试专用：自 Start() 起收到的 INSERT ring 消息总数。
  int insert_requests_received_for_test() const;

  // ticket 03 测试专用：dispatch 线程进入阻塞 poll() 的次数。空闲 server 约
  // 每秒 20 次（50ms 兜底超时），而非空转 ~20k 轮/秒。
  uint64_t poll_entries_for_test() const;

 private:
  void DispatchLoop();
  bool Quiescent();       // ticket 03：无任何可做工作（c2s 空 + outbox 空 +
                          //   无待 drain sample）时 true；在飞异步回调不算
                          //   （它们 enqueue 后会写 wake_fd_）
  void WakeDispatch();    // ticket 03：eventfd 写 1 字节把 dispatch 从 poll() 拉出。
                          //   Stop() 与每个 off-dispatch 生产者（insert ACK 聚合/
                          //   sample 完成/checkpoint executor）enqueue 之后调用。
  bool TryAccept();       // 非阻塞 accept（poll listen fd）
  void HandleInsertRequests(size_t client_id);   // 排空 insert C→S（ALLOCATE/INSERT/RELEASE）
  void HandleSampleRequests(size_t client_id);   // 排空 sample C→S（SAMPLE/RELEASE）
  void FlushOutbox(ClientState& state);          // 非阻塞写两个 outbox（§8.7）
  absl::Status HandleSample(std::shared_ptr<ClientState> state,
                            const ShmSampleRequest& req);   // 异步入队（评审 #1 持 shared_ptr）
  void DrainPendingSamples(ClientState& state);             // dispatch 线程 unpack+pool+SAMPLE_RESP
  absl::Status HandleRelease(ClientState& state,
                             const ShmReleaseRequest& req); // Unref，归零 Deallocate（C3）
  absl::Status HandleInsert(std::shared_ptr<ClientState> state,
                            const ShmInsertRequest& req);   // ParseFromArray → InsertOrAssignAsync
  absl::Status HandleAllocate(ClientState& state,
                              const ShmAllocateRequest& req);  // C4：授予池偏移
  absl::Status HandleMutatePriorities(ClientState& state,
                                      const MutatePrioritiesRequest& req);
  absl::Status HandleReset(ClientState& state, const ResetRequest& req);
  absl::Status HandleCheckpoint(std::shared_ptr<ClientState> state);
  absl::Status HandleServerInfo(ClientState& state);
  absl::Status EnqueueInsertS2C(ClientState& state, MsgType type,
                                absl::string_view body);  // TryWrite，满则入 insert_outbox
  absl::Status EnqueueSampleS2C(ClientState& state, MsgType type,
                                absl::string_view body);  // 同上，sample_outbox
  bool IsClientDead(const ClientState& state);            // ticket ⑥：poll fd 查 EOF/HUP
  void HandleDisconnect(size_t client_id);                // 集中释放 + unlink + 清 clients_
  void CleanupClient(ClientState& state, bool unlink_rings);
  absl::StatusOr<std::shared_ptr<Table>> FindTable(const std::string& name) const;

  internal::flat_hash_map<std::string, std::shared_ptr<Table>> tables_;
  std::string socket_path_;
  std::string name_token_;   // socket 路径 + per-server epoch（ticket #7）
  ShmBytePool pool_;
  ShmBootstrapServer bootstrap_;
  std::shared_ptr<Checkpointer> checkpointer_;  // ticket ⑪，可选注入
  std::vector<std::shared_ptr<ClientState>> clients_;  // 评审 #1：shared_ptr 所有权

  std::thread dispatch_thread_;
  std::atomic<bool> running_{false};
  std::atomic<int> insert_requests_received_{0};      // ticket 01 测试计数
  int wake_fd_ = -1;                                  // ticket 03：server-local eventfd
  std::atomic<uint64_t> poll_entries_for_test_{0};    // ticket 03 测试计数
  TaskExecutor checkpoint_executor_{1, "ShmCheckpointExecutor"};  // 评审 #3：checkpoint Save 专用
};

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
```

**线程模型与关键语义**（对应原 R3/R4/R11）：

- **单 dispatch 线程**：accept 新 client、排空所有 client 的两条 C→S ring、drain
  pending samples、flush outbox、检测断连，全在一线程。`pool_` 分配/回收/refcount
  因此无锁（R11）。
- **异步化**：`Table::Sample` 经 `EnqueSampleRequest` 异步入队，dispatch 不阻塞于
  rate limiter（ticket ⑩ 方向 A）；`InsertOrAssignAsync` 本就异步。两者完成回调
  都在表 worker 线程触发，只攒队列/保活列表，dispatch 线程负责全部 SHM 侧动作。
- **eventfd 事件驱动（ticket 03）**：dispatch 静止（Quiescent）时阻塞
  `poll(listen_fd + 各 client control_fd + wake_fd_, 50ms 兜底)`，不再忙轮询。
  所有 off-dispatch 生产者 enqueue 后调 `WakeDispatch()` 写 eventfd；
  `RingHeader::server_asleep` 让 client 只在 server 睡着时发唤醒字节。
- **批量 INSERT 聚合 ACK（ticket 01）**：`HandleInsert` 把一批（≤64 items /
  ≤128KB）的每个 `ShmChunkRef` 从池中 `ParseFromArray` 还原 `ChunkData`，逐 item
  `InsertOrAssignAsync`；表 worker 完成回调经 `AckAggregate` 聚合，最后一条完成时
  组装**一条** `INSERT_ACK`（全部 keys + offsets_to_release）入 insert_outbox。
  `EnqueueInsertS2C` 的 FIFO 护栏保证 RESP 定序（跨表回调并发下不乱序）。
- **checkpoint 不在 dispatch 线程跑（评审 #3）**：`HandleCheckpoint` 把
  `checkpointer_->Save`（无界磁盘 I/O）调度到 `checkpoint_executor_`（单线程
  TaskExecutor），完成后经 outbox 回传 `CHECKPOINT_RESP`；`Stop()` 先 drain
  executor 再停表。`~TaskExecutor` 可安全重入 `Close()`。
- **ClientState 生命周期（评审 #1）**：`clients_` 存 `shared_ptr<ClientState>`，
  insert/sample 完成回调按值捕获 shared_ptr——`clients_.erase()/clear()` 只丢
  server 的引用，回调仍可安全访问。`Stop()` 先 `Table::Stop()`（保证回调不再
  触发）再清客户端，确定性关闭窗口。
- **断连**：`IsClientDead` 每轮 poll 各 client fd（`POLLHUP/POLLERR/EOF`）；
  显式 `CLOSE` 消息置 `close_requested`，下一轮同路径处理。`HandleDisconnect` 集中
  释放 `outstanding_offsets_`（C3）、unlink 该 client 的 4 条 ring、关 fd、从
  `clients_` 移除。

### 3.5 Client 侧

```cpp
// reverb/cc/shm/shm_connection.h

namespace deepmind {
namespace reverb {
namespace shm {

// ticket ⑥（design §8.8）：`fd` 对端已关闭（EOF/POLLHUP/POLLERR）？双端共用：
//   - server：探测 ClientState.fd 检 client 崩溃；
//   - client：探测 control_fd 检 server 消失，让等 S→C 响应的 ReadBlocking
//     快速失败而非永久自旋。
// fd < 0（无 fd，如 moved-from / server 侧）=> false。
bool IsPeerClosed(int fd);

// 评审 #2（ring 写侧 liveness）：镜像读侧 ReadBlocking 的阻塞写辅助。轮询
// TryWrite + sched_yield，每轮探测 control_fd 对端死亡（UnavailableError），
// 超时给 DeadlineExceededError。没有它，Ring::Write 的裸 sched_yield 循环在
// server dispatch 被卡（如慢 checkpoint，评审 #3）或死亡时 100% CPU 永转，
// 挂死 TrajectoryWriter::Close()/GC 且无错误浮现。
constexpr absl::Duration kWriteBlockingHardCap = absl::Seconds(60);
absl::Status WriteBlocking(Ring* ring, MsgType msg_type,
                           absl::Span<const char> payload, int control_fd,
                           absl::Duration timeout = kWriteBlockingHardCap);

// 一个 server↔client 之间的 SPSC ring 组 + 共享字节池。决策 D（per-flow ring）：
// 两对 ring——insert 流（TrajectoryWriter 的 RunShmWorker）一对 + sample 流
// （ShmSampler worker）一对。每对严格 SPSC：该流唯一的 client worker 线程是
// 其 c2s 唯一生产者、s2c 唯一消费者。拆对让两条 worker 线程无需 mutex 并发跑
// ——单对设计在两条 writer 都起后台线程后违反 SPSC 不变式（一 c2s head 两生产者、
// 无 CAS => 数据损坏）。
//
// `pool` 句柄仅 client 侧有意义（C4 下 ShmBytePool::Open 的 RW 映射）；server
// 在自己的 ShmServer 里持有 owner/分配器 ShmBytePool，不经过此结构共享。
// 两侧都经 pool.At(offset) 读 sample 字节。
//
// `control_fd`（ticket ⑥）：连接期内保持打开的 udsocket fd，作 liveness 信号。
// server 侧把 accept 的 fd 存 ClientState.fd（这里保持 -1）；client 侧存
// bootstrap fd，~ShmConnection 关闭它。client 进程崩溃或 ~ShmClient 时 fd 关 →
// server poll 见 POLLHUP/EOF → HandleDisconnect（§8.8）。
struct ShmConnection {
  Ring insert_c2s;  // client insert worker -> server（ALLOCATE/INSERT/RELEASE）
  Ring insert_s2c;  // server -> client insert worker（ALLOCATE_RESP/INSERT_ACK）
  Ring sample_c2s;  // client sample worker -> server（SAMPLE/RELEASE）
  Ring sample_s2c;  // server -> client sample worker（SAMPLE_RESP）
  ShmBytePool pool; // client 侧 RW 映射（C4）；server 另有自己的
  std::string pool_shm_name;
  int control_fd = -1;  // client liveness fd（ticket ⑥）；-1 = 无

  // ticket「shm-close-while-in-flight」：Close()/析构在 control_fd 关闭**之前**
  // 置位。在飞读循环（shm_client.cc 的 ReadBlocking、trajectory_writer.cc 的
  // read_blocking）轮询此标志，连接被从下方关闭时立即 UnavailableError，而不是
  // 在无人服务的 ring 上永转。仅靠 fd 探测不够：close 后 fd = -1（或被复用），
  // fd < 0 时探测被跳过——那个洞曾挂死 DisconnectWithInFlightCallbacksDoesNotUaf。
  std::atomic<bool> closed{false};

  void Close();  // 幂等。先置 closed（release），再关 control_fd。

  // ticket ⑩：串行化 insert 流上的 send→read-ACK 往返，让两个生产者永不同时
  // 碰 insert_c2s 的单一 head。RunShmWorker（insert worker 后台线程的
  // ALLOCATE→ALLOCATE_RESP 与 INSERT→INSERT_ACK 往返）与调用线程的
  // MutatePriorities/Reset/ServerInfo/Checkpoint 往返都为此 mutex 覆盖整个
  // send→read 序列。sample 流不碰此锁，保持无锁。
  // （死锁复盘 2026-07-17：此锁曾被怀疑为偶发死锁根因，gdb 抓栈证明无辜——
  //   真根因是 dispatch 在 HandleSample 的 rate-limiter 无限阻塞，已由方向 A
  //   异步化修复。此锁保持原样：串行化整个往返反而是对的。）
  mutable absl::Mutex insert_flow_mu;
};

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
```

```cpp
// reverb/cc/shm/shm_client.h

namespace deepmind {
namespace reverb {
namespace shm {

// ShmSampler 是 SHM 传输的 client 侧 sampler。决策 C5：复用现有 Sampler 架构
// （worker 线程喂 samples_ 队列，GetNextTrajectory 弹出），只把每次取样的
// gRPC/local Table::Sample 换成 SHM ring 往返。
// GIL 纪律与现有 Sampler 一致：worker 线程从池字节构建 TensorBuffer（无 GIL）；
// GetNextTrajectory 在调用线程跑，ToNdArray() 才可能拿 GIL。
class ShmSampler {
 public:
  // `conn` 借用（ShmClient 所有），须活得比 sampler 久。`options` 镜像
  // Sampler::Options（v1 只用 max_samples + rate_limiter_timeout）。
  // `active_flag`（可选）：ShmClient 拥有的 single-sampler permit——NewSampler
  // 先 claim，Close()/析构时释放。
  static absl::StatusOr<std::unique_ptr<ShmSampler>> Create(
      ShmConnection* conn, const std::string& table_name,
      const Sampler::Options& options,
      std::atomic<bool>* active_flag = nullptr);

  // 阻塞至取到完整 sample。`data` 填充完整（展平）trajectory 的 TensorBuffer
  // （每列一个）。调用方可在线程上 ToNdArray()（GIL）。
  absl::Status GetNextTrajectory(std::vector<TensorBuffer>* data,
                                 std::shared_ptr<const SampleInfo>* info = nullptr);

  void Close();  // 取消 worker 并 join 线程
 private:
  void RunWorker();   // worker 主循环：取满 max_samples 或取消
  absl::StatusOr<std::unique_ptr<Sample>> FetchOne();  // 一次 SHM 往返
};

// ShmClient：连 ShmServer 的 udsocket，握手，mmap 五个 SHM 段
// （pool + 两对 ring）。NewSampler 返回 ShmSampler。
class ShmClient {
 public:
  static absl::StatusOr<std::unique_ptr<ShmClient>> Connect(
      const std::string& socket_path);

  absl::Status NewSampler(const std::string& table_name,
                          const Sampler::Options& options,
                          std::unique_ptr<ShmSampler>* sampler);

  // ---- Writer path（ticket ④）----

  // SHM 模式 TrajectoryWriter：chunker/column/backpressure 在 client 侧，
  // insert 经 SHM 到 server 的 Table（附录 A4）。ticket ⑧ step 2：
  // flat_signature_map 由一次实时 SERVER_INFO 往返填充，ItemAndRefs::Validate
  // 校验 trajectory 与表当前签名——每次调用都取新 TableInfo，会话中途
  // Table.replace 改签名在下个 writer 生效。
  absl::Status NewTrajectoryWriter(const TrajectoryWriter::Options& options,
                                   std::unique_ptr<TrajectoryWriter>* writer);

  // 镜像 InProcessClient::NewStructuredWriter：共享主体委托 MakeStructuredWriter，
  // 经 NewTrajectoryWriter 钩子包装 SHM TrajectoryWriter。各 config 的 `table`
  // 字段路由到 server 侧表。
  absl::Status NewStructuredWriter(
      std::vector<StructuredWriterConfig> configs,
      std::unique_ptr<StructuredWriter>* writer);

  // ponytail: plain Writer（writer.h）无 SHM seam——本地构造要 tables map，
  // SHM client 不持表（表在 server）。给 Writer 加 SHM 传输会为 legacy API
  // 复制 RunShmWorker 逻辑。Won't fix：需要 SHM insert 用 NewTrajectoryWriter /
  // NewStructuredWriter。Python 层（ShmClient）覆盖 `writer`/`insert` 在到达
  // 此处前抛 NotImplementedError；此 C++ stub 留给直接 pybind 调用方作防御性
  // UnimplementedError。
  absl::Status NewWriter(int chunk_length, int max_timesteps,
                         bool delta_encoded, int max_in_flight_items,
                         std::unique_ptr<Writer>* writer);

  // 经按需 SERVER_INFO ring 往返返回各表实时 TableInfo。ticket ⑧ step 2：
  // 取代 step-1 bootstrap 快照，server_info() 反映会话中途状态。走 insert 流
  // （insert_flow_mu 下），同其他控制面 op（10/11）。
  absl::Status ServerInfo(std::vector<TableInfo>* table_info);

  // ticket ⑩：控制面 op，镜像 InProcessClient::MutatePriorities / Reset。
  // 走 insert 流（insert_c2s/s2c，insert_flow_mu 下）——与 RunShmWorker 不竞态
  // 双生产者。未知表 -> NotFoundError（server 回 ShmError::NOT_FOUND），Python
  // 层 FileNotFoundError。复用 reverb_service.proto 的请求类型。
  absl::Status MutatePriorities(const std::string& table,
                                const std::vector<KeyWithPriority>& updates,
                                const std::vector<uint64_t>& deletes);
  absl::Status Reset(const std::string& table);

  // ticket ⑪：触发 server 侧 checkpoint 并返回保存路径。走 insert 流
  // （insert_flow_mu 下）。服务端错误（无 checkpointer、Save 失败）以
  // ShmError::INTERNAL -> absl::InternalError 到达。
  absl::Status Checkpoint(std::string* path);

  ShmConnection* connection() { return &conn_; }

 private:
  explicit ShmClient(ShmConnection conn);
  ShmConnection conn_;
  // 扫描 #1：本连接的 single-sampler permit。第二个活 sampler 会让两个生产者
  // worker 共写 sample_c2s（SPSC 无 CAS => head 损坏），且无 request_seq 时可能
  // 消费对方的 SAMPLE_RESP——静默错表数据。NewSampler claim（exchange），
  // ShmSampler::Close 释放。
  std::atomic<bool> sampler_active_{false};
};

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
```

> 注：原 spec §3.5 的 `SendRequest` / `RecvResponse` / `uds_fd_` /
> `table_info_cache_` 成员与"无后台线程"表述均已过时：I/O 统一走
> `ShmConnection` 的 insert/sample 两流（`insert_flow_mu` 串行化 insert 流），
> `ShmSampler` 有独立 worker 线程（决策 C5），`server_info` 是每次实时往返而非
> 缓存快照（ticket ⑧ step 2）。

## 4. Proto 定义

```protobuf
// reverb/cc/shm/shm_protocol.proto

syntax = "proto3";

package deepmind.reverb.shm;

import "reverb/cc/schema.proto";
import "reverb/cc/patterns.proto";
import "reverb/cc/reverb_service.proto";
import "third_party/reverb_tensor/reverb_tensor.proto";

option cc_enable_arenas = true;

// 线上消息类型，存于 SlotHeader.msg_type（uint16）。C→S 用小值，S→C 用 100+。
// 两个 ALLOCATE 类型实现决策 C4：client 写 insert 字节前先向 server（唯一
// 分配者）申请池偏移。
enum MsgType {
  MSG_TYPE_UNSPECIFIED = 0;
  // C→S
  HELLO = 1;
  INSERT = 2;
  SAMPLE = 3;
  RELEASE = 4;
  ALLOCATE = 5;  // C4：client 向 server 申请池偏移
  // ticket ⑩：控制面走 insert 流（insert_c2s/s2c）+ client 侧 mutex，不与
  // RunShmWorker 竞态双生产者。design §8.3 原把 5 预留给 MUTATE_PRIORITIES，
  // v1 把 5 给了 ALLOCATE（C4），故 6/7 是剩余空闲 C→S 值。
  MUTATE_PRIORITIES = 6;
  RESET = 7;
  // ticket ⑪：checkpoint 走 insert 流。复用 reverb_service.proto 的
  // CheckpointRequest/CheckpointResponse。
  CHECKPOINT = 8;
  CLOSE = 9;
  // ticket ⑧ step 2：按需 server_info 往返（取代连接时 bootstrap 快照）。
  // 复用 ServerInfoResponse（空请求，同 CheckpointRequest）。
  SERVER_INFO = 10;
  // S→C
  WELCOME = 101;
  INSERT_ACK = 102;
  SAMPLE_RESP = 103;
  ALLOCATE_RESP = 104;  // C4：server 返回授予的池偏移
  // ticket ⑩：控制面空 ACK（复用 reverb_service.proto 的请求类型）。
  MUTATE_ACK = 106;
  RESET_ACK = 107;
  // ticket ⑪：携带 CheckpointResponse.checkpoint_path 回 client。
  CHECKPOINT_RESP = 108;
  // ticket ⑧ step 2：携带 ServerInfoResponse（repeated TableInfo）。走 insert 流。
  SERVER_INFO_RESP = 110;
  ERROR = 105;
}

// ── C→S ──

message HelloRequest {
  int32 client_pid = 1;
  uint32 protocol_version = 2;
}

message ShmInsertRequest {
  repeated ShmChunkRef chunks = 1;
  repeated .deepmind.reverb.PrioritizedItem items = 2;
  repeated uint64 keep_chunk_keys = 3;
}

message ShmChunkRef {
  uint64 chunk_key = 1;
  uint64 shm_offset = 2;      // ShmBytePool 偏移（client 已 ALLOCATE 获得）
  uint64 total_length = 3;    // 序列化 ChunkData 字节数
  repeated .reverb.tensor.SignatureProto.TensorSpec specs = 4;  // 不填充（见下）
  .deepmind.reverb.SequenceRange sequence_range = 5;            // 不填充
  bool delta_encoded = 6;                                       // 不填充
}

message ShmSampleRequest {
  string table = 1;
  int64 num_samples = 2;
  int64 timeout_ms = 3;  // rate_limiter_timeout，-1 = InfiniteDuration
  bool emit_timesteps = 4;
}

message ShmReleaseRequest {
  repeated uint64 offsets = 1;
}

// C4：client 在 memcpy insert 字节前，向 server（唯一分配者）申请给定大小的池
// 偏移。server 回 ALLOCATE_RESP。
message ShmAllocateRequest {
  uint64 num_bytes = 1;
}

// ── S→C ──

message WelcomeResponse {
  string pool_shm_name = 1;
  // 已废弃的单对别名（留给外部读者）。v2 per-flow 传输（决策 D）拆成下方
  // insert_* / sample_*；这两个现在镜像 SAMPLE 流 ring，legacy client 仍能找到。
  string c2s_shm_name = 2 [deprecated = true];
  string s2c_shm_name = 3 [deprecated = true];
  .deepmind.reverb.ServerInfoResponse server_info = 4;
  // Per-flow ring（决策 D）：insert 流（TrajectoryWriter）与 sample 流
  // （ShmSampler）各一对 SPSC，worker 线程永不共享 ring 的生产者/消费者。
  string insert_c2s_shm_name = 5;
  string insert_s2c_shm_name = 6;
  string sample_c2s_shm_name = 7;
  string sample_s2c_shm_name = 8;
}

message InsertAck {
  repeated uint64 keys = 1;
  repeated uint64 offsets_to_release = 2;  // insert 已消费，client 可释放
}

message ShmSampleResponse {
  repeated ShmSample samples = 1;
}

message ShmSample {
  .deepmind.reverb.SampleInfo info = 1;
  repeated ShmColumn columns = 2;
}

message ShmColumn {
  uint64 shm_offset = 1;  // 成品字节偏移（server 预切片后）
  uint64 length = 2;
  .reverb.tensor.SignatureProto.TensorSpec spec = 3;
  bool squeeze = 4;       // 对齐 FlatTrajectory.Column.squeeze
}

message ShmAllocateResponse {
  uint64 shm_offset = 1;  // 授予的偏移；client 可 memcpy 字节至此
}

message ShmError {
  enum Code {
    UNKNOWN = 0;
    DEADLINE_EXCEEDED = 1;
    INVALID_ARGUMENT = 2;
    NOT_FOUND = 3;
    RESOURCE_EXHAUSTED = 4;
    INTERNAL = 5;
  }
  Code code = 1;
  string message = 2;
  uint64 request_seq = 3;  // 失败请求的 producer_seq（design §8.9）
}
```

**实现说明**：

- `ShmChunkRef` 的 `specs`/`sequence_range`/`delta_encoded` 字段**不填充**——
  传输的是序列化的 `ChunkData` proto 本身（client chunker 压缩完毕），server 直接
  `ParseFromArray` 还原，无需按 spec 拆列（`ponytail:` 冗余标注，proto 保留以对齐
  design §8.5 的多列设想）。
- `ShmError.request_seq` 保留但未填充——完全依赖 SPSC 保序隐式匹配（design §8.9
  实现现状）。
- `ERROR = 105`（早期设计表预留 104 给 ERROR、105 给 MUTATE_ACK；实现把 104 给了
  `ALLOCATE_RESP`，ERROR 后移 105）；`SERVER_INFO_RESP = 110`（与 `SERVER_INFO =
  10` 的 +100 对齐）。

## 5. pybind 绑定

`reverb/pybind.cc` 的 `PYBIND11_MODULE` 中**直接绑定 C++ 类**（无 Python 包装类），
snake_case + PascalCase 双名对齐 `Client`/`InProcessClient` 的既有模式：

```cpp
#include "reverb/cc/shm/shm_client.h"
#include "reverb/cc/shm/shm_server.h"

// ShmServer：由 Python `Server(shm=True)` 创建并持有（决策 C1）。
// ctor 与静态 Create 双 lambda 保持同步（ticket 02 的 slab/blocks 透传）。
py::class_<ShmServer, std::shared_ptr<ShmServer>>(m, "ShmServer")
    .def(py::init([](std::vector<std::shared_ptr<Table>> tables,
                     const std::string& socket_path,
                     std::shared_ptr<Checkpointer> checkpointer,
                     std::vector<size_t> slab_sizes,
                     size_t blocks_per_slab) { ... ShmServer::Create ... }),
         py::arg("tables"), py::arg("socket_path") = "",
         py::arg("checkpointer") = nullptr,
         py::arg("slab_sizes") = std::vector<size_t>{},
         py::arg("blocks_per_slab") = 0)
    .def_static("Create", ...)
    .def("Start", &ShmServer::Start, py::call_guard<py::gil_scoped_release>())
    .def("Stop", &ShmServer::Stop, py::call_guard<py::gil_scoped_release>())
    .def_property_readonly("socket_path", ...);

// ShmClient：`py::init` 即 ShmClient::Connect（bootstrap + mmap）。
py::class_<ShmClient, std::shared_ptr<ShmClient>>(m, "ShmClient")
    .def(py::init(shm_connect_fn), py::arg("socket_path"))
    .def_static("Connect", shm_connect_fn, py::arg("socket_path"))
    // keep_alive<0, 1>：返回的 sampler/writers 借用 client 的连接
    // （ShmSampler 用它做 SAMPLE 往返；writer 的 worker 线程经它 flush），
    // ShmClient 必须活得比它们久。
    .def("new_sampler", shm_new_sampler_fn,
         py::arg("table"), py::arg("max_samples") = 1,
         py::arg("buffer_size") = 1,
         py::arg("rate_limiter_timeout_ms") = -1,
         py::keep_alive<0, 1>())
    .def("NewSampler", shm_new_sampler_fn, ...)
    .def("new_trajectory_writer", shm_new_trajectory_writer_fn,
         py::arg("chunker_options"), py::keep_alive<0, 1>())
    .def("NewTrajectoryWriter", shm_new_trajectory_writer_fn, ...)
    .def("new_structured_writer", shm_new_structured_writer_fn,
         py::arg("configs"), py::keep_alive<0, 1>())
    .def("NewStructuredWriter", shm_new_structured_writer_fn, ...)
    .def("server_info", shm_server_info_fn)
    .def("ServerInfo", shm_server_info_fn)
    .def("mutate_priorities", shm_mutate_priorities_fn,
         py::arg("table"), py::arg("updates"), py::arg("deletes"))
    .def("MutatePriorities", shm_mutate_priorities_fn, ...)
    .def("reset", shm_reset_fn, py::arg("table"))
    .def("Reset", shm_reset_fn, py::arg("table"))
    .def("checkpoint", shm_checkpoint_fn)
    .def("Checkpoint", shm_checkpoint_fn);

// ShmSampler：镜像 Sampler 的 GetNextTrajectory——C++ 调用释放 GIL，
// 返回前重取 GIL 构建 info+data tensor 向量（numpy-backed TensorBuffer →
// ndarray 需 GIL）。布局（kNumInfoTensors info 标量前置 + 列数据）与 Sampler
// 一致，`reverb/client.py` 的 `_BaseClient.sample` 可两者通用。
py::class_<ShmSampler>(m, "ShmSampler")
    .def("GetNextTrajectory", ...);
```

> `pybind.cc` 没有 `PyShmClient` 包装类（plan 时代设想）；numpy `_import_array()`
> 复用模块入口既有调用。

## 6. Python 层

### `reverb/client.py` — `ShmClient(_BaseClient)`

```python
class ShmClient(_BaseClient):
    """SHM (POSIX shared memory) client for same-machine cross-process access.

    Connects to a `Server(shm=True)` over a udsocket and mmaps the five SHM
    segments (pool + two ring pairs: insert C→S/S→C + sample C→S/S→C, one
    pair per flow so each flow keeps its own SPSC producer).
    `sample`/`trajectory_writer`/`structured_writer` match `Client`/`LocalClient`
    exactly, so user code is transport-agnostic.
    """

    def __init__(self, socket_path: str, *, output_format: str = "numpy"):
        super().__init__(output_format=output_format)
        self._socket_path = socket_path
        # pybind.ShmClient.__init__ 调 ShmClient::Connect（bootstrap + mmap）；
        # 连接/握手失败抛异常。
        self._client = pybind.ShmClient(socket_path)

    def __repr__(self):
        return f"ShmClient(socket_path={self._socket_path})"

    def __reduce__(self):
        # ticket ⑫: pickle 只存 socket_path；反序列化经 __init__ 重连
        # （重新 bootstrap + mmap 五个 SHM 段）。原进程的 mmap/ring/fd 状态
        # 留在原进程、随原 ShmClient 析构释放——pickle 边界无泄漏。
        return self.__class__, (self._socket_path,)

    def writer(self, *args, **kwargs):
        # Legacy `Writer`（writer.h）无 SHM 传输 seam（ticket ⑬）：本地构造要
        # tables map，SHM client 不持表；为废弃 API 复刻 RunShmWorker 不值。
        # Python 侧抛清晰错误，不让它落到 C++ UnimplementedError。SHM insert
        # 用 `trajectory_writer` / `structured_writer`。
        raise NotImplementedError(
            "ShmClient does not support the legacy writer/insert; use "
            "trajectory_writer or structured_writer instead."
        )

    def insert(self, data, priorities: Dict[str, float]):
        # `insert` 是 `writer` 的糖（见 _BaseClient.insert）；同上。
        raise NotImplementedError(...)

    def _fetch_server_info_proto(self, timeout: Optional[int]):
        # 每次调用实时往返（ticket ⑧ step 2）；timeout 为对齐 gRPC hook 而
        # 接受但忽略（C++ 侧有 60s 硬上限）。
        return self._client.ServerInfo()

    def _new_sampler(self, table, num_samples, buffer_size, timeout_ms):
        timeout_ms_arg = -1 if timeout_ms is None or timeout_ms < 0 else timeout_ms
        return self._client.NewSampler(table, num_samples, buffer_size, timeout_ms_arg)

    def trajectory_writer(self, num_keep_alive_refs, *, max_chunk_length=None):
        # chunker_options 构造（AutoTuned/Constant）→ new_trajectory_writer →
        # reverb.trajectory_writer.TrajectoryWriter 包装。
        ...

    def structured_writer(self, configs):
        # 序列化 configs → new_structured_writer → StructuredWriter 包装。
        ...
```

要点：

- **`output_format`**（"numpy" / "torch"，torch 为可选依赖 `[torch]` extra）：
  `ShmClient` 与 `Client`/`LocalClient` 同构，采样侧 `output_format='torch'`
  时返回 torch.Tensor（torch 零拷贝路径，见 `torch-tensor-phase1.md`）。
- **pickle**（ticket ⑫）：`__reduce__` 存 `socket_path`，反序列化重连——与
  `LocalClient` 持进程内指针不可序列化不同。
- **writer/insert**（ticket ⑬）：Python 层覆盖抛 `NotImplementedError`。
- **signature 校验**（ticket ⑧-2b）：`trajectory_writer` 构造时调 `ServerInfo()`
  往返拿实时 `TableInfo` 填 `flat_signature_map`，`create_item` 校验 trajectory
  与表签名，未知表拒为 `ValueError`（对齐 InProcessClient/gRPC validate_items=True）。

### `reverb/server.py` — `Server` 的 shm 参数

```python
class Server:
    def __init__(self,
                 tables: Optional[Sequence[Table]] = None,
                 port: Optional[int] = None,
                 checkpointer: Optional[checkpointers.CheckpointerBase] = None,
                 in_process: bool = False,
                 shm: bool = False,
                 shm_socket_path: Optional[str] = None,
                 shm_pool_slab_sizes: Optional[List[int]] = None,
                 shm_pool_blocks_per_slab: Optional[int] = None):
        ...
        if shm:
            # 决策 C1：ShmServer 生命周期挂在 Server 对象上。
            # socket 路径默认 /tmp/reverb_shm_<pid>_<n>.sock（per-process 计数器
            # 防同进程两个 Server(shm=True) 互踩）。
            self._shm_server = pybind.ShmServer(
                [t.internal_table for t in tables],
                socket_path=shm_socket_path,
                checkpointer=checkpointer.internal_checkpointer(),
                # None -> empty/0 保持 C++ 默认路径（kDefaultSlabSizes x
                # kDefaultBlocksPerSlab）。
                slab_sizes=shm_pool_slab_sizes or [],
                blocks_per_slab=shm_pool_blocks_per_slab or 0,
            )
            self._shm_server.Start()
            self._shm_socket_path = self._shm_server.socket_path
```

- `shm=True` 时 `Server.shm_socket_path` 可用（`shm=False` 为 `None`）。
- `shm_pool_slab_sizes` / `shm_pool_blocks_per_slab`（ticket 02）：使用者向调优项，
  控制单连接池高水位（默认 9 档 × 256 块 ≈ 1.4GB）；小档配置可把高水位降到
  档位容量量级。
- `shm=True` 不隐含 `in_process=True`，两开关独立（组合矩阵见 guide）。

## 7. 线程模型与数据流

### 7.1 Server 进程

单 dispatch 线程（`ShmServer::DispatchLoop`）全权处理 SHM 面：

```
DispatchLoop()
  ├── poll(listen_fd + 各 client control_fd + wake_fd_, 50ms 兜底)  // ticket 03
  │     静止（Quiescent）才阻塞；有工作立即处理
  ├── TryAccept()：新 client → 握手（RecvHello 带小超时）→ 建 4 条 ring 段 →
  │     SendWelcome → 创建 ClientState
  ├── 对每个 client：
  │     HandleInsertRequests：排空 insert c2s
  │       ALLOCATE → HandleAllocate（选档位，授予偏移）
  │       INSERT → HandleInsert（ParseFromArray 各 ChunkData → 逐 item
  │                InsertOrAssignAsync；回调聚合 → 一条 INSERT_ACK 入 outbox）
  │       RELEASE → Unref 各偏移，归零 Deallocate
  │     HandleSampleRequests：排空 sample c2s
  │       SAMPLE → HandleSample（FindTable → EnqueSampleRequest 异步）
  │       RELEASE → Unref
  │     DrainPendingSamples：table worker 完成的 SampledItem →
  │       UnpackChunkColumnAndSlice → pool_.Allocate + Ref + memcpy →
  │       SAMPLE_RESP（非阻塞写，满则 sample_outbox）
  │     FlushOutbox：两个 outbox 非阻塞 TryWrite（§8.7）
  │     IsClientDead：fd EOF/HUP 或 close_requested → HandleDisconnect
  ├── off-dispatch 生产者（insert ACK 聚合回调、sample 完成回调、
  │     checkpoint executor）enqueue 后都调 WakeDispatch() 写 eventfd
  └── 控制面（10/11/08）：MUTATE_PRIORITIES/RESET/CHECKPOINT/SERVER_INFO 也
        走 insert c2s → HandleXxx → EnqueueInsertS2C；CHECKPOINT 的 Save 在
        checkpoint_executor_ 上跑（评审 #3）
```

### 7.2 Client 进程

- **`ShmSampler` worker 线程**（决策 C5）：`RunWorker` 循环 `FetchOne()`——
  发 SAMPLE 到 sample c2s → 轮询 sample s2c 收 SAMPLE_RESP（或 ERROR）→ 按
  `ShmColumn.shm_offset` 从池读字节构建 `TensorBuffer`（批次列：server 已预拼接
  batched tensor，`squeeze_columns` 解压）→ 发 RELEASE → 组装 `Sample` 推进
  `samples_` 队列。`GetNextTrajectory` 从队列取。`max_samples` 或取消时退出。
- **`RunShmWorker`**（TrajectoryWriter 的 SHM worker 线程，ticket ④/01）：从
  `write_queue_` 取 ready item → **凑批**（连续 already-ready item，≤64 个 /
  ≤128KB，不引入等待）→ `insert_flow_mu` 锁内：每 unique chunk 一条 ALLOCATE
  流水线突发（N 发 N 收 FIFO 配对）→ memcpy 序列化 `ChunkData` → 一条 INSERT →
  等一条聚合 INSERT_ACK → 清 in-flight、`local_can_insert_more_`、RELEASE
  `offsets_to_release`。一批 ~2 次 ring 往返（旧每 item 2 次）。
- **调用线程控制面**：`MutatePriorities`/`Reset`/`ServerInfo`/`Checkpoint` 在
  `insert_flow_mu` 下走 insert 流往返，与 RunShmWorker 串行。
- **阻塞策略**：Ring 本身非阻塞；调用点用 `ReadBlocking`/`WriteBlocking`
  （poll + sched_yield + control_fd 存活探测 + `closed` 标志 + 60s 硬上限）——
  server 死亡/被卡时快速失败（UnavailableError/DeadlineExceededError），不永转。

## 8. 测试策略

`bazel test //reverb/cc/shm/...`（10 个目标；无 plan 时代设想的
`shm_server_test.cc`/`shm_client_test.cc`）：

| 目标 | 覆盖 |
| --- | --- |
| `ring_test` | 单槽/跨槽消息、capacity 满背压、多轮循环、seq 绕回、HasData、server_asleep 唤醒 |
| `byte_pool_test` | slab 选择（最小够用）、分配/回收/再分配、refcount inc/dec→0 回收、几何校验（严格升序、≥8B）、档位耗尽 RESOURCE_EXHAUSTED |
| `bootstrap_test` | 握手往返（Hello→Welcome）、协议版本不匹配拒绝、段名格式 |
| `byte_pool_echo_test` | 双进程 pool+ring 回声（跨进程读写校验） |
| `echo_test` | 最简握手 + 回声端到端 |
| `shm_sample_test` | ShmSampler 完整 sample 路径（含批次列、timeout→DeadlineExceeded、未知表 NOT_FOUND） |
| `shm_insert_test` | RunShmWorker insert 路径（批量聚合 ACK 往返计数、ALLOCATE 配对、多表路由、signature 校验） |
| `shm_crash_test` | 崩溃恢复：client SIGKILL → server 集中释放无泄漏、新 client 可连；server 消失 → client 快速失败；close-while-in-flight（ticket「shm-close-while-in-flight」） |
| `shm_checkpoint_test` | checkpoint 保存 + 恢复（经 gRPC/InProcess 的 LoadLatest） |
| `shm_wakeup_test` | eventfd 唤醒：空闲 server poll 计数、唤醒后即时处理（ticket 03） |

双进程场景用 fork + `ShmServer`/`ShmClient` 真跨进程验证（echo/crash 测试）。

## 9. 边界情况与错误处理

| 场景 | 行为 | 参考 |
| --- | --- | --- |
| server 启动时旧 `.sock` 文件残留 | `ShmBootstrapServer::Create` 先 `unlink` 再 `bind` | R7 |
| server 重启时旧 SHM `/dev/shm/` 残留 | 段名含 server epoch（PID+墙上时钟纳秒），同 socket 重启不与活段碰撞；残留仅 tmpfs 少量，重启自清 | ticket #7 / A3 |
| client 写时 ring 满 | `WriteBlocking` 轮询 TryWrite（sched_yield），每轮探测 control_fd + `closed`，60s 硬上限 | 评审 #2 / ticket「shm-close-while-in-flight」 |
| server 写 S→C ring 满 | `TryWrite` 失败 → 暂存 per-flow outbox，下轮 FlushOutbox 重试；insert/sample outbox 互不阻塞 | §8.7 |
| byte pool 档位耗尽 | `Allocate` 快速失败 `RESOURCE_EXHAUSTED`（不阻塞，防 dispatch 死锁）；超档请求永久 `InvalidArgument` | ticket 02 |
| `Table::Sample` 返回 0 个 sample | 异步完成回调携带 status（含 DeadlineExceeded）→ SAMPLE_RESP 空 / ERROR | ticket ⑩ |
| client 崩溃（SIGKILL） | udsocket EOF → `HandleDisconnect` 集中释放 `outstanding_offsets_`、unlink 4 条 ring | §8.8 / ticket ⑥ |
| server 崩溃 | client 读侧探测 control_fd EOF/`closed` → `UnavailableError`（不永转） | §8.8 / ticket ⑥ |
| client 发送 `CLOSE` | `close_requested` 置位 → 下一轮 HandleDisconnect | ticket ⑥ |
| 多列 chunk spec 与字节不匹配 | 不存在——传输的是自描述 `ChunkData` proto，`ParseFromArray` 还原 | §8.5 |
| 协议版本不匹配 | server 回 ERROR + 关闭连接 | §8.2 |
| `Insert` items 引用未知 chunk_key | 对应 item 报错，已插入的照常 ACK | — |
| `Sample` 超时 | ERROR(DEADLINE_EXCEEDED) → Python `DeadlineExceededError` | §8.3 |
| 同一 SHM 偏移被多个 sample 引用 | 引用计数避免提前回收（C3） | §4.1 |
| 同连接第二个活 sampler | `NewSampler` 拒绝（single-sampler permit，SPSC 不变式） | 扫描 #1 |
| 会话中途 `Table.replace` 改签名 | 旧 writer 持旧签名；下个 `trajectory_writer` 构造时实时往返拿新签名 | ticket ⑧ step 2 |

## 附录：关键实现决策说明

### A0. 已确认决策（2026-07-09 用户确认）

| 编号 | 决策 | 内容 |
| --- | --- | --- |
| **C1** | ShmServer 生命周期挂在 `Server` Python 对象上 | 不独立成 `ShmServer` Python 对象。`Server(shm=True)` 构造时创建并持有 `ShmServer`（C++ dispatch 线程随之启动），`Server.stop()` / `__del__` 时销毁。用户经 `Server.shm_socket_path` 拿路径去连 `ShmClient`。 |
| **C2** | insert 的 SHM 偏移在收到 `INSERT_ACK.offsets_to_release` 前，client 不得重用 | client 发完 INSERT 请求后，对应的 SHM 字节区域视为"占用中"，writer 的 backpressure（`num_items_in_flight`）正是靠"等 ACK"来约束。server 压缩完进 ChunkStore 后回 ACK，client 才允许把该偏移归还字节池复用。 |
| **C3** | sample 的 SHM 偏移引用计数起点 = 1 | server 在 `ShmBytePool::Allocate` 成品字节 memcpy 进池时即设 refcount=1；client 读完该 sample 的所有列后发 RELEASE，server `Unref`，归零则 `Deallocate`。client 崩溃时 server 遍历 `outstanding_offsets_` 集中释放。 |
| **C4** | pool 读写权限 + 分配权归属 | client 以 `PROT_READ|PROT_WRITE` mmap pool；**但分配权仍在 server 单线程**。insert 路径：client 先经 insert ring C→S 向 server 申请偏移（`ALLOCATE` 子请求），server `Allocate` 返回偏移，client `memcpy` chunk 字节进该偏移，再发 `INSERT`。sample 路径：server 分配 + memcpy 成品字节，client 只读。即"谁生产谁写、server 独占分配器"。 |
| **C5** | SHM Sampler 复用现有 worker 线程架构 | 现有 `Sampler` 类基于后台 worker 线程 + `samples_` 队列，SHM sampler 保留这套（`ShmSampler::RunWorker`），只把 worker 内的 gRPC `SampleStream` 换成 SHM ring 往返。 |

### A1. 为什么 sample 路径下 server 用 dispatch 线程同步做 `UnpackChunkColumnAndSlice`

设计文档 §4.1 说分配单线程化。若把解压委派到 worker 池，而分配仍在 dispatch
线程，需跨线程传 buffer 和同步，增加复杂度。v1 选择在 dispatch 线程同步执行
解压+切片的简化方案：table worker 只做采样，成品 `SampledItem` 攒进
`pending_samples`，dispatch 线程 `DrainPendingSamples` 做 unpack + pool 分配 +
memcpy + 写 SAMPLE_RESP（`ponytail:` 瓶颈后升级为 per-client dispatch + 带锁
分配器）。

### A2. pool 读写权限与分配权归属（见决策 C4）

client 以 `PROT_READ|PROT_WRITE` mmap pool，但**不自行分配**——分配器是 server
单线程独占的 slab free list，无跨进程锁。两条路径的读写角色：

- **insert**（client 生产字节）：client 先经 insert ring C→S 向 server 申请偏移，
  server `Allocate` 返回偏移，client `memcpy` chunk 字节进该偏移，再发 `INSERT`。
  server 读这些字节 `ParseFromArray` 进 ChunkStore，回 `INSERT_ACK.offsets_to_release`，
  client 方可释放该偏移（C2）。
- **sample**（server 生产字节）：server `Allocate` + `memcpy` 成品字节，client 只读。

### A3. SHM 段名如何生成

权威定义在 `reverb/cc/shm/bootstrap.h`（`MakeShmNames`）。server 为每个连接生成
`/reverb_shm_pool_<token>` 与四条 ring（决策 D 每流一对）：
`/reverb_shm_{insert,sample}_{c2s,s2c}_<token>_<client_pid>`，全部经 Welcome 下发，
client 不自算。`<token>` = 消毒后的 server socket 路径 + server epoch（PID + 墙上
时钟纳秒）：socket 路径使同机/同进程多 server 实例不互删段（scan #12），epoch 使
同 socket 路径崩溃重启的 server 不与旧客户端的活段碰撞（ticket #7——旧段不再被
unlink-and-retry 误删，仅 tmpfs 少量残留，重启自清）。

### A4. `ShmClient.NewTrajectoryWriter` 为什么走 SHM 路径而不是直接持 Table

与 `InProcessClient` 不同，`ShmClient` 与 server 在不同进程，不能直接持有
`shared_ptr<Table>`。writer 的逻辑（chunker、column refs、backpressure）在 client
侧运行，但最终的 `InsertOrAssignAsync` 必须经 SHM 路径发往 server。具体做法
（`RunShmWorker`，ticket ④/01）：

1. Writer 的 `RunShmWorker` 线程在 client 进程跑，从 `write_queue_` 取 ready item
2. 凑批（≤64 items / ≤128KB）后，为每个 unique chunk 先发 `ALLOCATE` 申请池偏移
   （流水线突发，N 发 N 收 FIFO 配对）
3. chunker 产出的 `ChunkData`（已压缩）`SerializeToString` memcpy 进该偏移
4. 发一条 `INSERT` 请求（含全部 chunk 偏移 + items）到 insert ring C→S
5. server 读 ring，`ParseFromArray` 还原 `ChunkData` 入 ChunkStore
6. server 回一条聚合 `INSERT_ACK`（含 `offsets_to_release`）
7. client 收到后清 in-flight、允许 writer 重用该批偏移的 SHM 空间

### A5. `Sampler` 的 SHM 路径实现（见决策 C5）

SHM `Sampler` **复用现有 worker 线程 + `samples_` 队列架构**（`ShmSampler`）。
worker 线程内把原 gRPC `SampleStream` 往返换成 SHM ring 往返：发 `SAMPLE` 请求到
sample ring C→S → 轮询 sample ring S→C 取 `SAMPLE_RESP` → 按 `ShmColumn.shm_offset`
从 pool 读字节构建 `TensorBuffer` → 发 `RELEASE` → 把 `Sample` 推进 `samples_`
队列。`GetNextTrajectory` 仍从队列取，与 gRPC/local 路径一致。

> 注意：`TensorBuffer` 指向的 SHM 区域在 RELEASE 前必须保持有效。worker 读完
> 一次 sample 的所有列、组装完 `Sample` 入队后再发 RELEASE，避免 sample 未组装
> 完就被回收。
