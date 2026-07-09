# SHM 传输层实现规格说明书

> 本文档由 [numpy-shm-design.md](numpy-shm-design.md) 转化而来，补充了文件结构、
> BUILD 规则、类声明、测试策略和集成细节，供直接开发使用。面向已阅读设计文档
> 的开发者。

## 目录

- [0. 审阅意见摘要](#0-审阅意见摘要)
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
- [6. Python 层变更](#6-python-层变更)
- [7. 实现顺序](#7-实现顺序)
  - [Phase 1：基础设施](#phase-1基础设施)
  - [Phase 2：控制面](#phase-2控制面)
  - [Phase 3：数据面](#phase-3数据面)
  - [Phase 4：Python 集成](#phase-4python-集成)
  - [Phase 5：端到端验证](#phase-5端到端验证)
- [8. 测试策略](#8-测试策略)
- [9. 边界情况与错误处理](#9-边界情况与错误处理)
- [附录：关键实现决策说明](#附录关键实现决策说明)

---

## 0. 审阅意见摘要

设计文档质量较高，§8 的 ring 协议提供了可直接编码的伪代码。
以下为开发前需注意的遗漏/不明确之处：

| # | 问题 | 说明 |
| --- | ------ | ------ |
| R1 | **C++ 类声明缺失** | 无完整头文件，仅见伪代码。需补全 `ShmBytePool`、`ShmCtrlRing`、`ShmConnection`、`ShmServer`、`ShmClient` 的类声明。 |
| R2 | **BUILD 规则未定义** | 未提供 Bazel 构建目标。需为 `reverb/cc/shm/` 子目录新建 BUILD，并在 `reverb/cc/BUILD` 更新相关依赖。 |
| R3 | **线程模型细节不足** | §4.4 说 dispatch 单线程 + 表 worker 池，但未说明 dispatch 线程与 `Table` worker 线程池如何交互（dispatch 直接调 `Table::InsertOrAssignAsync` / `Table::Sample`？dispatch 调完后 callback 写回 S→C ring？）。需明确 callback 分发路径。 |
| R4 | **ShmServer 的生命周期管理** | 谁持有 `ShmServer`？`Server` Python 对象创建时启动，`Server.stop()` 时停止。需在 pybind `Server` 类中加 `shm` 参数，创建 `ShmServer` 线程。 |
| R5 | **client 侧 `ShmConnection` 线程模型** | client 读 S→C ring 是轮询还是阻塞？设计文档说 client 异步（等待 confirm），但未说明轮询在哪条线程（调用者线程？后台线程？）。建议统一：**写（C→S）+ 读（S→C）都在调用者线程同步完成**，读不到时忙等/有限退避。避免引入后台线程。 |
| R6 | **shm_unlink 与进程崩溃** | POSIX `shm_open` 创建的段在**最后一个 `munmap` + `shm_unlink`** 时销毁。server 进程异常退出时不自动清理 SHM 段，重启时 `/dev/shm/` 残留旧段。需 server 启动时尝试 `shm_unlink` 旧段（同名 PID 不同则跳过），或使用 `O_EXCL` 创建避免冲突。 |
| R7 | **udsocket 路径冲突** | `/tmp/reverb_shm_<pid>.sock` 在 server 崩溃重启后 PID 可能不变（若 fork 或 container restart），导致 `bind` 失败。需 `unlink` 旧 socket + 重新 bind。 |
| R8 | **缺失 slab 大小配置** | §4.1 说档位"可配"但未给出默认值。建议默认：`{64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304}`（即 64B–4MB，约 5.5MB/档位的元数据开销）。 |
| R9 | **release 时序问题** | §4.1 说 client 读完发 release 请求。但 insert 的 chunk 字节何时 release？设计文档说 insert 完成后 server 回 `INSERT_ACK` 带 `offsets_to_release`。但 insert 路径下，client 发完数据后需要等 server 确认两件事：① chunk 已被 server 读入 SHM（即压缩到 ChunkStore 完成）；② 偏移可以释放。这个时序需明确：`INSERT` 请求的 SHM 偏移在收到 `INSERT_ACK` 的 `offsets_to_release` 前，client 必须保持数据有效。**实际上 client 发完 INSERT 后不应立即重用该偏移**——需等 server 确认 server 侧已完成压缩。 |
| R10 | **sample 路径下 SHM 偏移的引用计数起点** | §4.1 说 sample 发 N 列则各偏移 +1，但 SHM 偏移是在 server 侧分配（memcpy 进池）时就设 refcount=1，还是发 S→C 响应时才设？建议：server `sample` 分配池空间 + memcpy 时设 refcount=1，响应发出后 client 读到再做 release。client 崩溃时 server 集中释放所有 outstanding 偏移（refcount 递减到 0 即回收）。 |
| R11 | **多线程安全：ShmBytePool 分配** | §4.1 说 server 单线程分配/回收。但 §4.4 说 dispatch 单线程 + 表 worker 池。如果 sample 的 `UnpackChunkColumnAndSlice` 在 worker 池执行，而该操作需分配 SHM 字节，则存在竞争。建议：SHM 分配只在 dispatch 线程进行。worker 解压后传回 `TensorBuffer`，由 dispatch 线程 memcpy 进池。这样可以保持分配单线程化。 |
| R12 | **信号与清理** | server 收到 SIGTERM/SIGINT 时需清理 SHM 段和 udsocket。当前 Server 的 `__del__`/`stop()` 路径需扩展到 ShmServer。 |
| R13 | **init 中 `_import_array()` 的依赖** | pybind.cc 的 `PYBIND11_MODULE` 入口调用了 `_import_array()`。在新增的 shm/ 模块中如果使用 `py::array_t` 需确保 numpy C-API 已初始化。建议统一在 pybind.cc 中注册所有 shm 绑定，复用已存在的 `_import_array()` 调用。 |

---

## 1. 文件布局

新增 `reverb/cc/shm/` 目录，所有 SHM 传输层代码集中于此：

```
reverb/cc/shm/
├── BUILD                       # Bazel 构建规则
├── ring.h                      # ShmCtrlRing (SPSC ring buffer)
├── ring.cc
├── ring_test.cc
├── byte_pool.h                 # ShmBytePool (slab-based POSIX shm allocator)
├── byte_pool.cc
├── byte_pool_test.cc
├── bootstrap.h                 # ShmBootstrap (udsocket bootstrap + health)
├── bootstrap.cc
├── bootstrap_test.cc
├── shm_server.h                # ShmServer (dispatch thread + per-client state)
├── shm_server.cc
├── shm_server_test.cc
├── shm_client.h                # ShmClient (C++ side, pybind ready)
├── shm_client.cc
├── shm_client_test.cc
├── shm_protocol.proto           # SHM 专有 proto 消息
└── shm_connection.h            # ShmConnection (持有三组 mmap 指针)
```

---

## 2. 构建规则（BUILD）

```python
# reverb/cc/shm/BUILD

load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_cc_library",
    "reverb_cc_test",
    "reverb_cc_proto_library",
)

package(default_visibility = ["//reverb:__subpackages__"])

# ---- Proto ----

reverb_cc_proto_library(
    name = "shm_protocol_cc_proto",
    srcs = ["shm_protocol.proto"],
    deps = [
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:patterns_cc_proto",
        "//third_party/reverb_tensor:reverb_tensor_cc_proto",
    ],
)

# ---- Ring ----

reverb_cc_library(
    name = "ring",
    srcs = ["ring.cc"],
    hdrs = ["ring.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

reverb_cc_test(
    name = "ring_test",
    srcs = ["ring_test.cc"],
    deps = [
        ":ring",
        "//reverb/cc/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

# ---- Byte Pool ----

reverb_cc_library(
    name = "byte_pool",
    srcs = ["byte_pool.cc"],
    hdrs = ["byte_pool.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

reverb_cc_test(
    name = "byte_pool_test",
    srcs = ["byte_pool_test.cc"],
    deps = [
        ":byte_pool",
        "//reverb/cc/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

# ---- Bootstrap (udsocket) ----

reverb_cc_library(
    name = "bootstrap",
    srcs = ["bootstrap.cc"],
    hdrs = ["bootstrap.h"],
    deps = [
        ":ring",
        "//reverb/cc:schema_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

reverb_cc_test(
    name = "bootstrap_test",
    srcs = ["bootstrap_test.cc"],
    deps = [
        ":bootstrap",
        "//reverb/cc/platform:status_matchers",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

# ---- ShmConnection ----

reverb_cc_library(
    name = "shm_connection",
    hdrs = ["shm_connection.h"],
    deps = [
        ":ring",
        ":byte_pool",
    ],
)

# ---- ShmServer ----

reverb_cc_library(
    name = "shm_server",
    srcs = ["shm_server.cc"],
    hdrs = ["shm_server.h"],
    deps = [
        ":bootstrap",
        ":byte_pool",
        ":ring",
        ":shm_connection",
        ":shm_protocol_cc_proto",
        "//reverb/cc:table",
        "//reverb/cc:sampler",
        "//reverb/cc:chunk_store",
        "//reverb/cc:tensor_compression",
        "//reverb/cc/platform:hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

reverb_cc_test(
    name = "shm_server_test",
    srcs = ["shm_server_test.cc"],
    deps = [
        ":shm_server",
        ":shm_client",
        "//reverb/cc/chunker",
        "//reverb/cc:rate_limiter",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/selectors:uniform",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/support:tensor_proxy",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

# ---- ShmClient ----

reverb_cc_library(
    name = "shm_client",
    srcs = ["shm_client.cc"],
    hdrs = ["shm_client.h"],
    deps = [
        ":bootstrap",
        ":byte_pool",
        ":ring",
        ":shm_connection",
        ":shm_protocol_cc_proto",
        "//reverb/cc:chunker",
        "//reverb/cc:sampler",
        "//reverb/cc:writer",
        "//reverb/cc:schema_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

reverb_cc_test(
    name = "shm_client_test",
    srcs = ["shm_client_test.cc"],
    deps = [
        ":shm_client",
        ":shm_server",
        "//reverb/cc/platform:status_matchers",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
```

并在 `reverb/cc/BUILD` 中将 `//reverb/cc/shm:shm_server` / `//reverb/cc/shm:shm_client`
加入 pybind 目标依赖。

---

## 3. C++ 接口与类

### 3.1 Ring 协议 `ShmCtrlRing`

```cpp
// reverb/cc/shm/ring.h

#ifndef REVERB_CC_SHM_RING_H_
#define REVERB_CC_SHM_RING_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "absl/base/internal/endian.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

namespace deepmind {
namespace reverb {
namespace shm {

// 常量
inline constexpr uint64_t kRingMagic = 0x524556524253484DULL;  // "REVRBSHM"
inline constexpr uint32_t kRingVersion = 1;
inline constexpr size_t kDefaultSlotSize = 256;      // bytes
inline constexpr size_t kDefaultCapacity = 1024;      // slots (256KB/ring)

// 段头 (64字节, 单 cache line)
struct RingHeader {
  uint64_t magic = kRingMagic;
  uint32_t version = kRingVersion;
  uint32_t capacity = kDefaultCapacity;       // 槽位数 (2的幂)
  uint32_t slot_size = kDefaultSlotSize;      // 每槽字节数 (含 SlotHeader)
  uint32_t reserved = 0;
  // 以下各占独立 cache line
  alignas(64) std::atomic<uint64_t> producer_seq{1};   // 下一个要写的槽的 seq
  alignas(64) std::atomic<uint64_t> consumer_seq{1};   // 下一个要读的槽的 seq
  uint64_t capacity_mask = capacity - 1;      // capacity - 1
  uint64_t padding[5];                        // 填满 cache line (64 字节对齐)
};
static_assert(sizeof(RingHeader) == 128,
              "RingHeader must be exactly 128 bytes (2 cache lines)");

// 槽头
struct SlotHeader {
  uint64_t seq;            // 写入方的 producer_seq (consumer 据此判有效)
  uint16_t msg_type;       // 消息类型 (见 shm_protocol.proto 的 msg_type 枚举)
  uint16_t flags;          // bit0: HAS_CONTINUATION; bit1: IS_CONTINUATION
  uint32_t body_len;       // 本槽 body 字节数 (<= slot_size - sizeof(SlotHeader))
};
static_assert(sizeof(SlotHeader) == 16, "SlotHeader must be exactly 16 bytes");

// 消息类型枚举 (对应 §8.3 `numpy-shm-design.md`)
enum class MsgType : uint16_t {
  kHello = 1,
  kInsert = 2,
  kSample = 3,
  kRelease = 4,
  kMutatePriorities = 5,
  kReset = 6,
  kCheckpoint = 7,
  kServerInfo = 8,
  kClose = 9,
  // S→C
  kWelcome = 101,
  kInsertAck = 102,
  kSampleResp = 103,
  kError = 104,
  kMutateAck = 105,
  kResetAck = 106,
  kCheckpointResp = 107,
  kServerInfoResp = 108,
};

inline constexpr uint16_t kFlagHasContinuation = 0x0001;
inline constexpr uint16_t kFlagIsContinuation = 0x0002;

// 一条 SPSC ring (POD, 可 mmap 共享)
struct Ring {
  RingHeader* header;                 // 指向 mmap 后的段头
  SlotHeader* slots;                  // 指向槽数组 (紧跟 header 之后)

  // 写入一条消息 (blocking via spinlock/yield)
  // `body` 为序列化后的 protobuf 字节
  absl::Status Write(uint16_t msg_type, const std::string& body);

  // 读取一条消息 (非阻塞)
  // 返回 NOT_FOUND 表示无数据
  absl::Status Read(uint16_t& msg_type, std::string& body);

  // 可写的槽位数
  size_t AvailableSlots() const;

  // 计算槽数组的总字节数
  static size_t SlotsBytes(size_t capacity, size_t slot_size);

  // 计算整个段的总字节数 (header + slots)
  static size_t TotalBytes(size_t capacity, size_t slot_size);
};

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SHM_RING_H_
```

**实现要点**（见设计文档 §8.4）：

- `Write` 中等待可用空间用 `sched_yield()` 忙等；若 `/proc/sys/kernel/sched_yield` 不可用，可用 `std::this_thread::yield()`。
- `Read` 非阻塞：检查 `slot.seq != consumer_seq` 时直接返回 `NotFoundError("NOT_READY")`。
- 关键正确性：写方先填 body 最后 `release(seq)`，读方 `acquire(seq)` 确认就绪再读 body。

### 3.2 字节池 `ShmBytePool`

```cpp
// reverb/cc/shm/byte_pool.h

#ifndef REVERB_CC_SHM_BYTE_POOL_H_
#define REVERB_CC_SHM_BYTE_POOL_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace deepmind {
namespace reverb {
namespace shm {

// 默认 slab 档位 (bytes)
inline constexpr size_t kDefaultSlabSizes[] = {
    64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304
};
inline constexpr int kDefaultSlabCount =
    sizeof(kDefaultSlabSizes) / sizeof(kDefaultSlabSizes[0]);

// 默认池容量：每档位 256 块 (约 1.3GB 总容量)
inline constexpr size_t kDefaultBlocksPerSlab = 256;

class ShmBytePool {
 public:
  // 创建并 mmap 一个新的 SHM 段
  // `shm_name` 如 "/reverb_shm_pool_<pid>"
  static absl::StatusOr<ShmBytePool> Create(
      const std::string& shm_name,
      absl::Span<const size_t> slab_sizes = {},
      size_t blocks_per_slab = kDefaultBlocksPerSlab);

  // 打开一个已有的 SHM 段 (client 侧只读)
  static absl::StatusOr<ShmBytePool> Open(
      const std::string& shm_name);

  ~ShmBytePool();

  // 不可拷贝，可移动
  ShmBytePool(const ShmBytePool&) = delete;
  ShmBytePool& operator=(const ShmBytePool&) = delete;
  ShmBytePool(ShmBytePool&&) noexcept;
  ShmBytePool& operator=(ShmBytePool&&) noexcept;

  // 分配一块。返回偏移量。阻塞等待 (调用者负责：仅 server dispatch 线程调)
  absl::StatusOr<uint64_t> Allocate(size_t bytes);

  // 回收一块
  void Deallocate(uint64_t offset);

  // 获取某偏移处的指针 (client 只读 / server 读写)
  void* At(uint64_t offset);
  const void* At(uint64_t offset) const;

  // 段名
  const std::string& name() const { return shm_name_; }

  // 段大小
  size_t size() const { return pool_size_; }

  // 引用计数操作 (server 侧)
  void Ref(uint64_t offset);
  bool Unref(uint64_t offset);  // returns true if zero (ready for dealloc)

  // 释放该 client 所有 outstanding 偏移 (崩溃恢复)
  void ReleaseAll(const std::vector<uint64_t>& offsets);

 private:
  struct Slab {
    size_t block_size;           // 档位字节数
    size_t block_count;          // 块数
    uint64_t region_offset;      // 数据区起点 (相对 pool 基址)
    uint64_t free_head_offset;   // free list 链表头偏移 (空闲链)
  };

  std::string shm_name_;
  int fd_ = -1;                  // shm fd (server 创建用)
  void* base_ = nullptr;         // mmap 基址
  size_t pool_size_ = 0;
  Slab* slabs_ = nullptr;        // 指向 mmap 内 slab 元数据区
  int slab_count_ = 0;

  // 引用计数表 (server 侧独占, 非 SHM)
  // ponytail: 用 absl::flat_hash_map, 后续可换为数组 + 位图
  struct RefCountEntry {
    uint64_t offset;
    int count;
  };
  // 由于 map 需要大小可变, 存在进程内, 不在 SHM 中
  //   -> 因此 refcount 是进程内独占, 不是跨进程共享的
  internal::flat_hash_map<uint64_t, int> refcounts_;

  // 池满时阻塞用的条件变量 (server 侧)
  absl::Mutex mu_;
  absl::CondVar cv_;
  bool pool_full_ = false;
};

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SHM_BYTE_POOL_H_
```

**实现要点**：

- server 侧的 `Create`：`shm_open(name, O_CREAT|O_RDWR|O_EXCL, 0600)` → `ftruncate` → `mmap` → 初始化 slab 元数据 + free list。
- client 侧的 `Open`：`shm_open(name, O_RDWR, 0)` → `mmap` → 读 header 校验 magic/version。
- Slab 内部布局：`[SlabHeader x N][free list 元数据][数据块 0][数据块 1]...`。free list 是单链表（偏移量存储在 free 块的前 8 字节）。
- `At(offset)`：`static_cast<char*>(base_) + offset`。
- 引用计数只在 server 进程中维护，client 不持有 refcounts_ map。

### 3.3 Bootstrap

```cpp
// reverb/cc/shm/bootstrap.h

#ifndef REVERB_CC_SHM_BOOTSTRAP_H_
#define REVERB_CC_SHM_BOOTSTRAP_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace deepmind {
namespace reverb {
namespace shm {

// Server 侧：创建 udsocket listener，接受 client 连接，分发 Welcome
class ShmBootstrapServer {
 public:
  // bind + listen 在 `socket_path`
  // cleanup: 先 unlink 旧 socket 再 bind (防 PID 复用冲突, R7)
  static absl::StatusOr<ShmBootstrapServer> Create(
      const std::string& socket_path);

  // 接受一个 client 连接
  // 返回 (client_fd, client_pid)
  // 阻塞直到有 client connect
  absl::StatusOr<std::pair<int, int>> Accept();

  const std::string& socket_path() const { return socket_path_; }

 private:
  ShmBootstrapServer(int listen_fd, std::string socket_path);
  int listen_fd_;
  std::string socket_path_;
};

// Server 侧：单次握手
// 接收 HelloRequest → 发 WelcomeResponse (含三段 shm 段名)
absl::Status SendWelcome(
    int client_fd,
    const std::string& pool_shm_name,
    const std::string& c2s_shm_name,
    const std::string& s2c_shm_name,
    const std::string& server_info_proto);

absl::StatusOr<HelloRequest> RecvHello(int client_fd);

// Client 侧：连接 server udsocket，执行握手
// 接收返回的 pool / ring shm 段名
struct BootstrapResult {
  std::string pool_shm_name;
  std::string c2s_shm_name;
  std::string s2c_shm_name;
  std::string server_info_proto;
};
absl::StatusOr<BootstrapResult> ClientBootstrap(
    const std::string& socket_path, int client_pid);

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SHM_BOOTSTRAP_H_
```

### 3.4 Server 侧 `ShmServer`

```cpp
// reverb/cc/shm/shm_server.h

#ifndef REVERB_CC_SHM_SHM_SERVER_H_
#define REVERB_CC_SHM_SHM_SERVER_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/shm/bootstrap.h"
#include "reverb/cc/shm/byte_pool.h"
#include "reverb/cc/shm/ring.h"
#include "reverb/cc/shm/shm_connection.h"
#include "reverb/cc/table.h"

namespace deepmind {
namespace reverb {
namespace shm {

// 每个已连接的 client 的状态
struct ClientState {
  int fd = -1;                    // udsocket
  int client_pid = 0;
  ShmConnection conn;             // 三段 mmap
  // 该 client 所有 outstanding SHM 偏移 (用于崩溃集中释放)
  internal::flat_hash_set<uint64_t> outstanding_offsets_;

  // 暂存的非阻塞 S→C 写失败消息 (outbox)
  // ponytail: vector+mutex, 升 ring-buffer 按需
  absl::Mutex outbox_mu;
  std::vector<std::pair<uint16_t, std::string>> outbox ABSL_GUARDED_BY(outbox_mu);
};

class ShmServer {
 public:
  // `socket_path`: udsocket 路径。空则自动生成 "/tmp/reverb_shm_<pid>.sock"
  static absl::StatusOr<std::shared_ptr<ShmServer>> Create(
      const std::vector<std::shared_ptr<Table>>& tables,
      const std::string& socket_path = "",
      size_t pool_bytes = 0);   // 0 = default (2x sum(max_size × avg_chunk))

  ~ShmServer();

  // 不可拷贝/移动
  ShmServer(const ShmServer&) = delete;
  ShmServer& operator=(const ShmServer&) = delete;

  // 启动 dispatch 线程 (在创建后调用)
  absl::Status Start();

  // 停止 dispatch 线程，清理 client 连接，unlink SHM 段
  void Stop();

  const std::string& socket_path() const { return socket_path_; }

 private:
  // dispatch 线程主循环
  void DispatchLoop();

  // 处理一个 client 的 C→S ring
  void HandleClientRequests(int client_id, ClientState& state);

  // 处理 S→C outbox (非阻塞写)
  void FlushOutbox(int client_id, ClientState& state);

  // insert 请求处理
  absl::Status HandleInsert(int client_id, ClientState& state,
                            const ShmInsertRequest& req);

  // sample 请求处理
  absl::Status HandleSample(int client_id, ClientState& state,
                            const ShmSampleRequest& req);

  // 断连处理
  void HandleDisconnect(int client_id, ClientState& state);

  std::vector<std::shared_ptr<Table>> tables_;
  internal::flat_hash_map<std::string, std::shared_ptr<Table>> tables_by_name_;
  std::string socket_path_;
  std::unique_ptr<ShmBytePool> pool_;

  std::unique_ptr<ShmBootstrapServer> bootstrap_;
  std::vector<std::unique_ptr<ClientState>> clients_;  // index = client_id

  // dispatch 线程控制
  std::thread dispatch_thread_;
  std::atomic<bool> running_{false};

  // ChunkStore 引用 (用于 Load checkpoint 等)
  // ponytail: v1 暂不维护 chunk_key→SHM_offset 索引
};

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SHM_SHM_SERVER_H_
```

**线程模型细化**（对应 R3）：

```
DispatchLoop()
  ├── 轮询 accept 是否有新 client → 创建 ClientState + 两条 ring
  ├── 轮询每个 client 的 C→S ring → 解析请求类型
  │   ├── INSERT → 读 SHM 字节 → CompressTensorAsProto → Table::InsertOrAssignAsync
  │   │              → callback: 写 S→C INSERT_ACK (含 offsets_to_release)
  │   ├── SAMPLE → Table::Sample → UnpackChunkColumnAndSlice (在 dispatch 线程同步执行)
  │   │              → memcpy 成品字节进 SHM pool
  │   │              → 写 S→C SAMPLE_RESP
  │   │              → client 读完发 RELEASE → server 递减 refcount
  │   ├── RELEASE → 遍历 offsets, Unref, 归零则 Deallocate
  │   ├── MUTATE_PRIORITIES → Table::MutateItems → S→C MUTATE_ACK
  │   ├── RESET → Table::Reset → S→C RESET_ACK
  │   ├── CHECKPOINT → checkpointer_->Save → S→C CHECKPOINT_RESP
  │   └── CLOSE → HandleDisconnect
  ├── 对每个 client, 尝试 FlushOutbox (非阻塞写 S→C)
  ├── 检查 udsocket EOF (断连检测)
  └── sleep(0) or sched_yield()
```

> **注意 R11**：`UnpackChunkColumnAndSlice` 在 dispatch 线程内同步完成，
> 不委派到 worker 池。因为 sample 成品字节需立即进 SHM 池（分配操作非线程安全）。
> 若 dispatch 线程成为瓶颈，后续升级为 per-client dispatch 线程 + 带锁的 SHM 分配。

### 3.5 Client 侧 `ShmClient`

```cpp
// reverb/cc/shm/shm_client.h

#ifndef REVERB_CC_SHM_SHM_CLIENT_H_
#define REVERB_CC_SHM_SHM_CLIENT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/sampler.h"
#include "reverb/cc/shm/bootstrap.h"
#include "reverb/cc/shm/byte_pool.h"
#include "reverb/cc/shm/ring.h"
#include "reverb/cc/shm/shm_connection.h"
#include "reverb/cc/structured_writer.h"
#include "reverb/cc/trajectory_writer.h"
#include "reverb/cc/writer.h"

namespace deepmind {
namespace reverb {
namespace shm {

// 镜像 InProcessClient 的 API, 供 pybind 暴露为 ShmClient
class ShmClient {
 public:
  // 连接 server udsocket + 握手 + mmap 三段 SHM
  static absl::StatusOr<std::unique_ptr<ShmClient>> Connect(
      const std::string& socket_path);

  ~ShmClient();

  // ---- Writer Path (对齐 InProcessClient) ----

  absl::Status NewTrajectoryWriter(
      const TrajectoryWriter::Options& options,
      std::unique_ptr<TrajectoryWriter>* writer);

  absl::Status NewStructuredWriter(
      std::vector<StructuredWriterConfig> configs,
      std::unique_ptr<StructuredWriter>* writer);

  absl::Status NewWriter(
      int chunk_length, int max_timesteps, bool delta_encoded,
      int max_in_flight_items,
      std::unique_ptr<Writer>* writer);

  // ---- Sampler Path ----

  absl::Status NewSampler(
      const std::string& table_name,
      const Sampler::Options& options,
      std::unique_ptr<Sampler>* sampler);

  // ---- Direct operations (对齐 InProcessClient) ----

  absl::Status MutatePriorities(
      const std::string& table,
      const std::vector<KeyWithPriority>& updates,
      const std::vector<uint64_t>& deletes);

  absl::Status Reset(const std::string& table);

  absl::Status Checkpoint(std::string* path);

  absl::Status ServerInfo(std::vector<TableInfo>* table_info);

  // ---- SHM 特有 ----

  // 获取底层 ShmConnection (供 Sampler/Writer 读取 SHM 字节用)
  ShmConnection* connection() { return &conn_; }

 private:
  ShmClient(std::string socket_path);

  // 写入 C→S ring (阻塞)
  absl::Status SendRequest(uint16_t msg_type, const std::string& body);

  // 从 S→C ring 读取响应 (非阻塞 + 有限重试)
  absl::StatusOr<std::pair<uint16_t, std::string>> RecvResponse(
      absl::Duration timeout = absl::InfiniteDuration());

  std::string socket_path_;
  int uds_fd_ = -1;
  ShmConnection conn_;

  // ServerInfo 缓存
  internal::flat_hash_map<std::string, TableInfo> table_info_cache_;
};

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SHM_SHM_CLIENT_H_
```

**Client 侧线程模型澄清**（对应 R5）：

所有 I/O 在调用者线程同步完成：`NewSampler` 发请求到 C→S ring，轮询 S→C 等响应。`Sampler::GetNextTrajectory` 同理——在 client 进程中，Sampler 的每次调用都会（通过 `ShmConnection`）发 SAMPLE 请求、等响应、读 SHM 字节、发 RELEASE。无后台线程。

`TrajectoryWriter` 的 insert 路径：chunk 在 client 侧组装完毕 → memcpy 进 SHM 池 → SendRequest（INSERT）→ 等待 INSERT_ACK（backpressure）。这与设计文档 §D2 一致。

---

## 4. Proto 定义

```protobuf
// reverb/cc/shm/shm_protocol.proto

syntax = "proto3";

package deepmind.reverb.shm;

import "reverb/cc/schema.proto";
import "reverb/cc/patterns.proto";

option cc_enable_arenas = true;

// ── C→S ──

message HelloRequest {
  int32 client_pid = 1;
  uint32 protocol_version = 2;
}

message ShmInsertRequest {
  repeated ShmChunkRef chunks = 1;
  repeated PrioritizedItem items = 2;
  repeated uint64 keep_chunk_keys = 3;
}

message ShmChunkRef {
  uint64 chunk_key = 1;
  uint64 shm_offset = 2;
  uint64 total_length = 3;
  repeated TensorSpecProto specs = 4;
  SequenceRange sequence_range = 5;
  bool delta_encoded = 6;
}

message ShmSampleRequest {
  string table = 1;
  int64 num_samples = 2;
  int64 timeout_ms = 3;          // rate_limiter_timeout, -1 = InfiniteDuration
  bool emit_timesteps = 4;
}

message ShmReleaseRequest {
  repeated uint64 offsets = 1;
}

// ── S→C ──

message WelcomeResponse {
  string pool_shm_name = 1;
  string c2s_shm_name = 2;
  string s2c_shm_name = 3;
  ServerInfoResponse server_info = 4;  // 复用现有的 server_info proto
}

message InsertAck {
  repeated uint64 keys = 1;
  repeated uint64 offsets_to_release = 2;   // insert 用完了, client 可释放
}

message ShmSampleResponse {
  repeated ShmSample samples = 1;
}

message ShmSample {
  SampleInfo info = 1;
  repeated ShmColumn columns = 2;
}

message ShmColumn {
  uint64 shm_offset = 1;
  uint64 length = 2;
  TensorSpecProto spec = 3;
  bool squeeze = 4;
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
  uint64 request_seq = 3;
}
```

---

## 5. pybind 绑定

在 `reverb/pybind.cc` 的 `PYBIND11_MODULE` 中添加：

```cpp
#include "reverb/cc/shm/shm_client.h"
#include "reverb/cc/shm/shm_protocol.pb.h"

// ── 在已有模块内 ──
namespace {

// 在现有 `MaybeRaiseFromStatus` 之后新增 ShmClient 的包装类
// 这样可利用已有的 numpy _import_array() 初始化

class PyShmClient {
 public:
  explicit PyShmClient(const std::string& socket_path) {
    auto result = deepmind::reverb::shm::ShmClient::Connect(socket_path);
    MaybeRaiseFromStatus(result.status());
    client_ = *std::move(result);
  }

  // 镜像 InProcessClient 的 pybind 方法
  py::object NewSampler(const std::string& table, int num_samples,
                        int buffer_size, int timeout_ms) { ... }

  py::object NewWriter(int chunk_length, int max_timesteps,
                       bool delta_encoded, int max_in_flight_items) { ... }

  py::object NewTrajectoryWriter(py::handle chunker_options) { ... }

  py::object NewStructuredWriter(const std::vector<std::string>& configs) { ... }

  void MutatePriorities(const std::string& table,
                        const std::vector<std::pair<uint64_t, double>>& updates,
                        const std::vector<uint64_t>& deletes) { ... }

  void Reset(const std::string& table) { ... }

  std::string Checkpoint() { ... }

  std::vector<std::string> ServerInfo() { ... }

 private:
  std::unique_ptr<deepmind::reverb::shm::ShmClient> client_;
};

}  // namespace

// 在 PYBIND11_MODULE 的现有类注册之后添加:
py::class_<PyShmClient>(m, "ShmClient")
    .def(py::init<const std::string&>(), py::arg("socket_path"))
    .def("NewSampler", &PyShmClient::NewSampler,
         py::arg("table"), py::arg("num_samples"),
         py::arg("buffer_size"), py::arg("timeout_ms") = -1)
    .def("NewWriter", &PyShmClient::NewWriter,
         py::arg("chunk_length"), py::arg("max_timesteps"),
         py::arg("delta_encoded"), py::arg("max_in_flight_items"))
    .def("NewTrajectoryWriter", &PyShmClient::NewTrajectoryWriter,
         py::arg("chunker_options"))
    .def("NewStructuredWriter", &PyShmClient::NewStructuredWriter,
         py::arg("configs"))
    .def("MutatePriorities", &PyShmClient::MutatePriorities)
    .def("Reset", &PyShmClient::Reset)
    .def("Checkpoint", &PyShmClient::Checkpoint)
    .def("ServerInfo", &PyShmClient::ServerInfo)
    // snake_case 别名 (对齐 InProcessClient)
    .def("new_sampler", &PyShmClient::NewSampler)
    .def("new_writer", &PyShmClient::NewWriter)
    .def("new_trajectory_writer", &PyShmClient::NewTrajectoryWriter)
    .def("new_structured_writer", &PyShmClient::NewStructuredWriter);
```

**PascalCase + snake_case 双名**策略——对齐 `InProcessClient` 的现有模式
（`pybind.cc` 中对 `InProcessClient` 已有双名绑定）。

> 注意：`ShmClient` 不可 pickle（类似 `LocalClient`），因为持有 SHM mmap 指针。

---

## 6. Python 层变更

### `reverb/client.py` — 新增 `ShmClient`

```python
class ShmClient(_BaseClient):
    """SHM-based client for same-machine cross-process access.

    Uses POSIX shared memory for zero-serialization data transfer.
    API matches `Client` and `LocalClient` exactly.
    """

    def __init__(self, socket_path: str):
        super().__init__()
        self._socket_path = socket_path
        self._client = pybind.ShmClient(socket_path)

    def __repr__(self):
        return f"ShmClient(socket_path={self._socket_path})"

    def _fetch_server_info_proto(self, timeout: Optional[int]):
        return self._client.ServerInfo()

    def _new_sampler(
        self, table: str, num_samples: int, buffer_size: int,
        timeout_ms: Optional[int],
    ):
        timeout_ms_arg = -1 if timeout_ms is None or timeout_ms < 0 else timeout_ms
        return self._client.NewSampler(table, num_samples, buffer_size, timeout_ms_arg)

    def trajectory_writer(
        self, num_keep_alive_refs: int, *,
        max_chunk_length: Optional[int] = None,
    ):
        if num_keep_alive_refs < 1:
            raise ValueError(...)
        if max_chunk_length is None:
            chunker_options = pybind.AutoTunedChunkerOptions(num_keep_alive_refs, 1.0)
        else:
            chunker_options = pybind.ConstantChunkerOptions(
                max_chunk_length=max_chunk_length,
                num_keep_alive_refs=num_keep_alive_refs,
            )
        cpp_writer = self._client.new_trajectory_writer(chunker_options)
        from reverb import trajectory_writer as trajectory_writer_lib
        return trajectory_writer_lib.TrajectoryWriter(cpp_writer)

    def structured_writer(self, configs):
        if not configs:
            raise ValueError(...)
        serialized_configs = [config.SerializeToString() for config in configs]
        cpp_writer = self._client.new_structured_writer(serialized_configs)
        from reverb import structured_writer as structured_writer_lib
        return structured_writer_lib.StructuredWriter(cpp_writer)
```

### `reverb/server.py` — `Server` 新增 `shm` 参数

```python
class Server:
    def __init__(self,
                 tables: Optional[Sequence[Table]] = None,
                 port: Optional[int] = None,
                 checkpointer: Optional[checkpointers.CheckpointerBase] = None,
                 in_process: bool = False,
                 shm: bool = False,                          # ← 新增
                 shm_socket_path: Optional[str] = None):     # ← 新增
        ...
        if shm:
            from reverb.shm_server import ShmServer as PyShmServer
            self._shm_server = PyShmServer(
                [t.table for t in tables],
                socket_path=shm_socket_path,
            )
            self._shm_socket_path = self._shm_server.socket_path()
        ...
```

`reverb/shm_server.py` 是新增的 `pybind.ShmServer` 的 Python 包装。

---

## 7. 实现顺序

### Phase 1：基础设施

| 步骤 | 文件 | 产出 |
| ------ | ------ | ------ |
| 1.1 | `shm_protocol.proto` | proto 定义 + `reverb_cc_proto_library` 构建 |
| 1.2 | `ring.h`, `ring.cc`, `ring_test.cc` | SPSC ring 读写 + 单元测试（无 mmap，纯内存） |
| 1.3 | `byte_pool.h`, `byte_pool.cc`, `byte_pool_test.cc` | Slab 分配/回收 + 引用计数 + 单元测试 |

### Phase 2：控制面

| 步骤 | 文件 | 产出 |
|------|------|------|
| 2.1 | `bootstrap.h`, `bootstrap.cc`, `bootstrap_test.cc` | udsocket 握手 + 段名交换 + 单元测试（双进程测试） |
| 2.2 | `shm_connection.h` | 三组 mmap 的 RAII 包装 |

### Phase 3：数据面

| 步骤 | 文件 | 产出 |
|------|------|------|
| 3.1 | `shm_server.h`, `shm_server.cc`, `shm_server_test.cc` | Dispatch 循环 + insert/sample/release 处理 + 断连恢复 |
| 3.2 | `shm_client.h`, `shm_client.cc`, `shm_client_test.cc` | Client 连接 + insert/sample writer/sampler 完整路径 |

### Phase 4：Python 集成

| 步骤 | 文件 | 产出 |
| ------ | ------ | ------ |
| 4.1 | `reverb/pybind.cc` | `PyShmClient` 的 pybind 绑定 + `PascalCase`/`snake_case` 双名 |
| 4.2 | `reverb/client.py` | `ShmClient(_BaseClient)` 类 |
| 4.3 | `reverb/server.py` | `Server` 的 `shm=True` 参数 + `ShmServer` Python 包装 |

### Phase 5：端到端验证

| 步骤 | 内容 |
| ------ | ------ |
| 5.1 | 单进程双线程测试（模拟 client/server 双进程，用 udsocket + SHM 连通） |
| 5.2 | 真双进程测试（fork + `ShmClient` 连子进程 `ShmServer`） |
| 5.3 | 崩溃恢复测试（kill client → server 清理不泄漏） |
| 5.4 | 性能对比基准（gRPC loopback vs SHM sample throughput + latency） |

---

## 8. 测试策略

### 单元测试

| 模块 | 测试项 |
| ------ | -------- |
| `Ring` | ✅ 单槽消息写读<br>✅ 多槽跨消息（> slot_size）写读<br>✅ capacity 满时阻塞/背压<br>✅ 多轮循环（写满→读空→写满→...）<br>✅ seq 绕回（seq > capacity 多轮） |
| `BytePool` | ✅ slab 选择（选最小够用档位）<br>✅ 分配/回收/再分配<br>✅ 引用计数 inc/dec → 0 时回收<br>✅ 池满时阻塞 behavior（靠超时检测）<br>✅ 多块同档位分配复用 |
| `Bootstrap` | ✅ 握手往返（Hello → Welcome）<br>✅ 段名格式校验<br>✅ 协议版本不匹配拒绝 |

### 集成测试

| 测试 | 内容 |
| ------ | ------ |
| `ShmServer + ShmClient` | insert → sample 完整路径 (同进程双线程) |
| `ShmClient + TrajectoryWriter` | writer append → create_item → sample 读出 |
| `ShmClient::MutatePriorities` | 更新优先级后 sample 概率变化 |
| `ShmClient::Reset` | 重置后 table 为空 |
| `ShmClient::Checkpoint + Load` | checkpoint → 新建 server load → sample 恢复数据 |

### 双进程测试

```cpp
// pseudo-code for process-level test
TEST(ShmIntegration, TwoProcessSample) {
  pid_t pid = fork();
  if (pid == 0) {  // child: server
    auto server = ShmServer::Create(tables, socket_path);
    server->Start();
    sleep(10 秒);  // 等 client 做操作
    server->Stop();
    _exit(0);
  } else {  // parent: client
    sleep(1);  // 等 server 就绪
    auto client = ShmClient::Connect(socket_path);
    // ... 正常 insert / sample ...
    // kill child
  }
}
```

### 崩溃恢复测试

```cpp
// client 进程崩溃，server 正确清理
TEST(ShmIntegration, ClientCrashCleanup) {
  // 1. 创建 server
  // 2. client 连接，insert 数据
  // 3. kill client (SIGKILL)
  // 4. 验证 server 无泄漏:
  //    - 该 client 的 outstanding_offsets_ 为空
  //    - ring 和 pool 段可被 shm_unlink
  //    - 其他 client 不受影响
  // 5. 新 client 可正常连接使用
}
```

---

## 9. 边界情况与错误处理

| 场景 | 行为 | 参考 |
| ------ | ------ | ------ |
| server 启动时旧 `.sock` 文件残留 | `ShmBootstrapServer::Create` 先 `unlink(socket_path)` 再 `bind` | R7 |
| server 重启时旧 SHM `/dev/shm/` 残留 | 使用 `O_EXCL` 创建，失败则 `shm_unlink` 旧名重试（仅同名 PID）；或清所有 `/reverb_shm_*` 前缀 | R6 |
| client 发送消息时 ring 满 | `Ring::Write` 忙等（`sched_yield`），无超时（对齐全语义） | S14 |
| server 写 S→C ring 满 | 暂存到 `ClientState.outbox`，跳过该 client 继续轮询 | §8.7 |
| byte pool 满 | 分配阻塞，暂存 sample 请求，等释放后继续 | S14 |
| server dispatch 线程内 `Table::Sample` 返回 0 个 sample | 写空 `SAMPLE_RESP`（`samples` 字段空列表） | — |
| client 崩溃（SIGKILL） | udsocket EOF → server 调用 `HandleDisconnect` | §8.8 |
| server 崩溃 | client 读到 udsocket EOF → 报 `ConnectionError` | §8.8 |
| client 发送 `CLOSE` 消息 | server 正常释放该 client 全部资源 | §8.8 |
| 多列 chunk 的 spec 数量与 bytes 长度不匹配 | server 在校验时返回 `INVALID_ARGUMENT` | — |
| client 协议版本与 server 不匹配 | server 返回 `ERROR(INVALID_ARGUMENT)` + 关闭连接 | §8.2 |
| `Insert` 请求中 `items` 引用的 `chunk_key` 不在 `chunks` 中 | server 返回 `INSERT_ACK` 只含已插入的 items，缺失的 key 报错 | — |
| `Sample` 超时 | server 返回 `ERROR(DEADLINE_EXCEEDED)` | §8.3 ShmError |
| 同一个 SHM 偏移同时被多个 sample 引用 | 引用计数避免提前回收 | §4.1 |

---

## 附录：关键实现决策说明

### A0. 已确认决策（2026-07-09 用户确认）

以下三项在审阅时由实现方单方面拟定，现已由用户确认锁定，开发时直接遵循：

| 编号 | 决策 | 内容 |
| --- | --- | --- |
| **C1** | ShmServer 生命周期挂在 `Server` Python 对象上 | 不独立成 `ShmServer` Python 对象。`Server(shm=True)` 构造时创建并持有 `ShmServer`（C++ dispatch 线程随之启动），`Server.stop()` / `__del__` 时销毁。用户经 `Server.shm_socket_path` 拿路径去连 `ShmClient`。 |
| **C2** | insert 的 SHM 偏移在收到 `INSERT_ACK.offsets_to_release` 前，client 不得重用 | client 发完 INSERT 请求后，对应的 SHM 字节区域视为"占用中"，writer 的 backpressure（`num_items_in_flight`）正是靠"等 ACK"来约束。server 压缩完进 ChunkStore 后回 ACK，client 才允许把该偏移归还字节池复用。 |
| **C3** | sample 的 SHM 偏移引用计数起点 = 1 | server 在 `ShmBytePool::Allocate` 成品字节 memcpy 进池时即设 refcount=1；client 读完该 sample 的所有列后发 RELEASE，server `Unref`，归零则 `Deallocate`。client 崩溃时 server 遍历 `outstanding_offsets_` 集中释放。 |

> 注：设计文档 §6 的 v2 优化（insert 字节复用于 sample 切片源，省压缩往返）
> 仍延后，v1 按"server 从压缩 ChunkStore 解压到 SHM"实现。

| **C4** | pool 读写权限 + 分配权归属 | client 以 `PROT_READ|PROT_WRITE` mmap pool；**但分配权仍在 server 单线程**。insert 路径：client 先经 C→S ring 向 server 申请偏移（`ALLOCATE` 子请求），server `Allocate` 返回偏移，client `memcpy` chunk 字节进该偏移，再发 `INSERT`。sample 路径：server 分配 + memcpy 成品字节，client 只读。即"谁生产谁写、server 独占分配器"。修正原 A2 的"client 只读"表述——只读仅对 sample 响应字节成立。 |
| **C5** | SHM Sampler 复用现有 worker 线程架构 | 不按原 A5 的"无 worker、调用者线程内联"实现。现有 `Sampler` 类基于后台 worker 线程 + `samples_` 队列，SHM sampler 保留这套，只把 worker 内的 gRPC `SampleStream` 换成 SHM ring 往返。diff 更小、复用已测机制。原 A5 的"无 worker"属理想化，作废。 |

### A1. 为什么 sample 路径下 server 用 dispatch 线程同步做 `UnpackChunkColumnAndSlice`

设计文档 §4.1 说分配单线程化。若把解压委派到 worker 池，而分配仍在 dispatch 线程，
需跨线程传 buffer 和同步，增加复杂度。v1 选择在 dispatch 线程同步执行解压+切片的
简化方案（`ponytail:` 瓶颈后升级为 per-client dispatch + 带锁分配器）。

### A2. pool 读写权限与分配权归属（见决策 C4）

client 以 `PROT_READ|PROT_WRITE` mmap pool，但**不自行分配**——分配器是 server
单线程独占的 slab free list，无跨进程锁（spec §4.1）。两条路径的读写角色：

- **insert**（client 生产字节）：client 先经 C→S ring 向 server 申请偏移，server
  `Allocate` 返回偏移，client `memcpy` chunk 字节进该偏移，再发 `INSERT`。
  server 读这些字节压缩进 ChunkStore，回 `INSERT_ACK.offsets_to_release`，client
  方可释放该偏移（C2）。
- **sample**（server 生产字节）：server `Allocate` + `memcpy` 成品字节，client 只读。

原"client 只读"的表述仅对 sample 响应字节成立，已由 C4 修正。

### A3. 为什么 SHM 段名含 PID

`/reverb_shm_pool_<server_pid>`、`/reverb_shm_c2s_<server_pid>_<client_pid>`、
`/reverb_shm_s2c_<server_pid>_<client_pid>`。PID 后缀避免同机多个 server 实例冲突。
即使一个 server 异常退出重启后 PID 不同，新段名与旧段不冲突（旧段需 `shm_unlink`）。

### A4. `ShmClient.NewTrajectoryWriter` 为什么走 SHM 路径而不是直接持 Table

与 `InProcessClient` 不同，`ShmClient` 与 server 在不同进程，不能直接持有
`shared_ptr<Table>`。writer 的逻辑（chunker、column refs、backpressure）在 client
侧运行，但最终的 `InsertOrAssignAsync` 必须经 SHM 路径发往 server。
具体做法：

1. Writer 的 `RunLocalWorker` 或等效 dispatch loop 在 client 线程运行
2. chunk 数据组装完毕后 memcpy 到 client 侧共享的 SHM 字节池的一块区域
3. 发 INSERT 请求（含 SHM 偏移）到 C→S ring
4. server 读 ring，压缩该 SHM 字节到 ChunkStore
5. server 回复 INSERT_ACK（含 `offsets_to_release`）
6. client 收到后允许 writer 重用该偏移的 SHM 空间

### A5. `Sampler` 的 SHM 路径实现（见决策 C5）

SHM `Sampler` **复用现有 worker 线程 + `samples_` 队列架构**，不按"调用者线程内联"
实现。worker 线程内把原 gRPC `SampleStream` 往返换成 SHM ring 往返：发 `SAMPLE`
请求到 C→S ring → 轮询 S→C ring 取 `SAMPLE_RESP` → 按 `ShmColumn.shm_offset` 从
pool 读字节构建 `TensorBuffer` → 发 `RELEASE` → 把 `Sample` 推进 `samples_` 队列。
`GetNextTrajectory` 仍从队列取，与 gRPC/local 路径一致。

> 注意：`TensorBuffer` 指向的 SHM 区域在 RELEASE 前必须保持有效。worker 读完
> 一次 sample 的所有列、组装完 `Sample` 入队后再发 RELEASE，避免 sample 未组装
> 完就被回收。
