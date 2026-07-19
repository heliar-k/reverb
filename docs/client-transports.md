# Reverb 客户端对比：gRPC `Client` / `LocalClient` / `ShmClient`

> 面向**使用者**的选型与避坑参考。三种客户端都继承 `_BaseClient`，共享
> `sample`/`insert`/`writer`/`mutate_priorities`/`reset`/`server_info`/
> `checkpoint` 的实现，但**传输路径、API 覆盖面、行为细节有真实差异**。
> 设计背景见 [numpy-embed-design.md](numpy-embed-design.md)（gRPC/Local）与
> [numpy-shm-design.md](numpy-shm-design.md)（SHM）。

## 0. TL;DR — 怎么选

| 场景 | 选哪个 | 构造 |
| --- | --- | --- |
| 同进程内嵌训练，无序列化/无网络 | `LocalClient` | `Server(in_process=True).in_process_client` |
| 同一台机器跨进程，要最快采样 | `ShmClient` | `reverb.ShmClient(server.shm_socket_path)` |
| 跨机器 / 分布式 / 需要全套控制面 | gRPC `Client` | `reverb.Client('host:port')` |
| 需 pickle 客户端（如多进程 worker 持有） | gRPC `Client` 或 `ShmClient` | — |
| 需要 `checkpoint` | gRPC 或 `LocalClient` 或 `ShmClient` | — |
| 需要实时 `server_info`（反映 `Table.replace`） | gRPC 或 `LocalClient` | — |

> ⚠️ **关键约束**：`Server` 构造时 `shm=True` **不会自动打开** `in_process=True`，
> 两者需分别显式指定。想同时使用 `LocalClient` 和 `ShmClient` 时必须写
> `Server(in_process=True, shm=True)`。

一句话：**能内嵌就 `LocalClient`；要跨进程就要么 gRPC（图省事/要全套 API）要么
`ShmClient`（图采样性能）；要 pickle 用 gRPC 或 `ShmClient`（`ShmClient` 限同一台
机器）。**

## 1. 总览对比

| 维度 | `Client`（gRPC） | `LocalClient`（in_process） | `ShmClient`（shm） |
| --- | --- | --- | --- |
| 传输 / 拓扑 | 跨进程 / 跨机器，gRPC stream，序列化 numpy bytes | **同进程**，直接持 `shared_ptr<Table>`，无序列化 | **同一台机器跨进程**，POSIX shm + Unix Domain Socket，mmap 零拷贝 |
| 性能 | 基线 | 最快（无网络 / 无序列化） | ~9–11× gRPC 回环（见 [shm-benchmark.md](shm-benchmark.md)） |
| 构造 | `Client('localhost:port')` | `server.in_process_client` | `reverb.ShmClient(server.shm_socket_path)` |
| 内部持有资源 | gRPC channel | 进程内 Table 指针 | SHM mmap + ring 状态 |
| `pickle` | ✅ 支持（存 `server_address`） | ❌ 不可（持进程内指针） | ✅ 支持（存 `socket_path`，反序列化重连） |
| 多表 | ✅ 全部表 | ✅ 全部表 | ✅ 全部表（按表名路由） |
| `sample` 的 `timeout_ms` | ⚠️ **静默忽略**（gRPC `NewSampler` 无 timeout 参数） | ✅ 生效（超时抛 `DeadlineExceededError`） | ✅ 生效（超时抛 `DeadlineExceededError`） |
| `sample` 默认 `emit_timesteps` | `True` | `True` | `True`（三者统一） |
| `server_info` | ✅ 真实，带 timeout | ✅ 真实，忽略 timeout | ✅ 真实（每次调用均实时往返） |
| `mutate_priorities` / `reset` | ✅ | ✅ | ✅（走 insert 流 + 客户端互斥锁） |
| `checkpoint` / 恢复 | ✅ | ✅（`Server(in_process=True)` 构造时自动 `LoadLatest`） | ✅（走 insert 流；恢复时搭车 gRPC/InProcess 的 `LoadLatest`，见 §5.8） |
| `trajectory_writer` / `structured_writer` | ✅ | ✅ | ✅（chunker/column 在 client 侧，insert 走 SHM；`validate_items` 总是开） |
| `writer`（legacy）/ `insert` | ✅ | ✅ | ❌ **Python 层抛 `NotImplementedError`** |

## 2. API 覆盖面

三者对 `_BaseClient` 方法的支持：

```
                    Client      LocalClient     ShmClient
sample              ✅          ✅              ✅
trajectory_writer   ✅          ✅              ✅
structured_writer   ✅          ✅              ✅
writer (legacy)     ✅          ✅              ❌ NotImplementedError (Python 层)
insert              ✅          ✅              ❌ NotImplementedError (Python 层)
mutate_priorities   ✅          ✅              ✅
reset               ✅          ✅              ✅
server_info         ✅ 真实      ✅ 真实          ✅ 真实（实时往返）
checkpoint          ✅          ✅              ✅
```

**`ShmClient` 不提供 `insert`/`writer`**：legacy `Writer`（writer.h）没有
SHM seam（其本地构造需要 tables map，SHM client 不持表），为其复刻
`RunShmWorker` 不划算。`ShmClient` 在 Python 层直接覆盖两者抛清晰的
`NotImplementedError("...use trajectory_writer or structured_writer")`，
不再走到底层 C++ 的 `UnimplementedError`。SHM 写入只能走
`trajectory_writer` / `structured_writer`。

## 3. 三种使用样例

三者写数据的核心 API 完全一致，只有构造和传输不同。下面各给一个最小可运行样例。

### 3.1 gRPC `Client`（跨进程 / 跨机器）

```python
import numpy as np
import reverb

server = reverb.Server(
    tables=[reverb.Table(
        name='t', sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(), max_size=100,
        rate_limiter=reverb.rate_limiters.MinSize(1))],
    in_process=False)                  # 起 gRPC，分配端口
try:
    client = reverb.Client(f'localhost:{server.port}')

    with client.trajectory_writer(num_keep_alive_refs=3) as w:
        for i in range(3):
            w.append({'obs': np.zeros(4, np.float32) + i})
        w.create_item(table='t', priority=1.0,
                      trajectory={'obs': w.history['obs'][:]})
        w.flush()

    for s in client.sample('t', num_samples=1, emit_timesteps=False):
        print(np.asarray(s.data[0]))
finally:
    server.stop()
```

### 3.2 `LocalClient`（同进程内嵌，无序列化/无网络）

```python
server = reverb.Server(tables=[...], in_process=True)   # 不起 gRPC，无端口
client = server.in_process_client                        # LocalClient

# 后续 trajectory_writer / sample 用法与 gRPC 完全一致
```

### 3.3 `ShmClient`（同一台机器跨进程，零拷贝）

```python
server = reverb.Server(tables=[...], in_process=True, shm=True)
#   注意：shm=True 不隐含 in_process=True，两者需分别显式指定。
#   in_process=True 持有 Table；shm=True 在其上叠加 SHM 传输层。
#   SHM 支持全部表（按表名路由）。
try:
    client = reverb.ShmClient(server.shm_socket_path)

    with client.trajectory_writer(num_keep_alive_refs=3) as w:   # ✅ 支持
        w.append({'obs': np.zeros(4, np.float32)})
        w.create_item(table='t', priority=1.0, trajectory={...})
        w.flush()

    for s in client.sample('t', num_samples=1, emit_timesteps=False):
        print(np.asarray(s.data[0]))
finally:
    server.stop()
```

## 4. Server flag 组合矩阵

`Server` 的三个传输 flag 可自由组合：

| `in_process` | `port` | `shm` | 结果 | 可用客户端 |
| --- | --- | --- | --- | --- |
| `True` | 忽略（`server.port` 为 `None`） | `False` | 纯内嵌，无 gRPC 无端口 | `LocalClient` |
| `True` | 忽略（`server.port` 为 `None`） | `True` | 内嵌 + SHM | `LocalClient` + `ShmClient` |
| `False` | 自动/指定 | `False` | 纯 gRPC server | gRPC `Client` |
| `False` | 自动/指定 | `True` | gRPC + SHM | gRPC `Client` + `ShmClient` |

约束：

- `in_process=True` 时 `server.port` 为 `None`，`localhost_client()` 抛错，只能
  `server.in_process_client`。
- `in_process=False` 时 `server.in_process_client` 抛错，用 `localhost_client()`
  或自行 `reverb.Client(f'localhost:{server.port}')`。
- `shm=True` 时 `server.shm_socket_path` 可用；`shm=False` 时为 `None`。
- ⚠️ `shm=True` 不隐含 `in_process=True`。如果写
  `Server(in_process=False, shm=True)`，则得到 gRPC + SHM（无 LocalClient）。
  要同时用 `LocalClient` 和 `ShmClient`，必须写 `Server(in_process=True, shm=True)`。

## 5. 常见陷阱

### 5.1 `ShmClient.insert` / `ShmClient.writer` 抛 `NotImplementedError`

`insert` 和 `writer` 是从 `_BaseClient` 继承的，但 legacy `Writer`（writer.h）
没有 SHM seam（其本地构造需要 tables map，SHM client 不持表）。`ShmClient` 在
Python 层覆盖两者，直接抛清晰的 `NotImplementedError`，不再走到底层 C++
`UnimplementedError`。**SHM 写入只能用 `trajectory_writer` / `structured_writer`。**

```python
client = reverb.ShmClient(server.shm_socket_path)
client.insert(data, {'t': 1.0})   # ❌ NotImplementedError (Python 层)
with client.trajectory_writer(3) as w: ...   # ✅
```

### 5.2 `ShmClient.server_info()` 每次调用均实时往返

`server_info()` 每次调用都从服务端取实时 `TableInfo`（`max_size`/
`sampler_options`/`remover_options`/`signature`/`current_size` 等），反映会话
中途的插入/状态变化。往返走 INSERT 流（`insert_c2s`/`insert_s2c`）+
`insert_flow_mu` 串行化。`timeout` 参数被接受（与 gRPC/Local hook 对齐）但忽略
——SHM 往返用自有硬上限（`kReadBlockingHardCap=60s`，见 `ShmClient::ServerInfo`）。

### 5.3 `ShmClient` 的 signature 校验

`trajectory_writer` 的 signature 校验已生效：`NewTrajectoryWriter` 调用
`ServerInfo()` 往返拿实时 `TableInfo`，填 `flat_signature_map`，`create_item`
经 `ItemAndRefs::Validate` 校验 trajectory 与表签名（列数/dtype/shape），不匹配
抛 `ValueError`——与 `LocalClient` 一致（总是校验，无 `validate_items=False`
开关）。无签名的表跳过校验。

`sample(unpack_as_table_signature=True)` 可用：`server_info()` 返回真实
`TableInfo`，带签名的表可正常按签名解包。

> ⚠️ **陷阱**：`NewTrajectoryWriter` 构造时获取签名并缓存。会话中途
> `Table.replace` 改签名后，旧 writer 仍持旧签名，需重建 writer 才能拿到新签名。

### 5.4 `ShmClient` 多表路由

`ShmServer` 构造时接收**全部表**，按表名路由：`sample`/`trajectory_writer` 的
`table` 参数直接定位目标表。引用未知表名：`sample` 返回 `NOT_FOUND`（Python
`FileNotFoundError`）；`trajectory_writer.create_item` 经 `flat_signature_map`
在 client 侧拒为 `ValueError`（对齐 InProcessClient/gRPC 的
`validate_items=True`）。多表与 gRPC / `LocalClient` 行为一致。

### 5.5 `timeout_ms` 行为不对称

`sample(table, num_samples, timeout_ms=...)`：

- gRPC `Client`：**静默忽略** `timeout_ms`（C++ `NewSampler` 无 timeout 参数）。
- `LocalClient` / `ShmClient`：**生效**，rate limiter 阻塞超时抛
  `reverb.errors.DeadlineExceededError`。

在三种客户端之间切换代码时注意：在 gRPC 上"能等"的调用，换 SHM/Local 加了
`timeout_ms` 后可能开始抛超时。

### 5.6 pickle：gRPC `Client` 与 `ShmClient` 可 pickle，`LocalClient` 不可

`LocalClient` 持进程内 `Table` 指针，不可 pickle。gRPC `Client` pickle 时只存
`server_address`，子进程反序列化后重连。`ShmClient` 同理：pickle 时只存
`socket_path`，反序列化调用 `__init__` → `ShmClient::Connect`（重新 bootstrap +
重建 SHM 连接），原进程的 mmap/ring 状态留在原进程、随原 client 销毁释放，无跨
进程泄漏。要分发客户端到多进程 worker（如 `multiprocessing.Pool` 的
initializer）时，`ShmClient` 也可 pickle——worker 必须与 server 运行在同一台
机器上。

### 5.7 `emit_timesteps` 默认值已统一

三者 `_default_emit_timesteps` 都是 `True`（早期 `LocalClient` 曾默认 `False`，
已修正）。显式传 `emit_timesteps=False` 取整条 trajectory。

### 5.8 `ShmClient.checkpoint()` 的保存与恢复

**保存**：`checkpoint()` 与 `mutate_priorities`/`reset` 同走 INSERT 流
（`insert_c2s`/`insert_s2c`），在 `ShmConnection::insert_flow_mu` 下串行化
send→read-ACK。服务端 `HandleCheckpoint` 调注入的 `checkpointer_->Save` 落盘
所有表，经 `CHECKPOINT_RESP` 返回路径。

**恢复**：`ShmClient` 自身不执行恢复。`ShmServer` 与 gRPC/InProcess 共享同一批
`Table` 对象，Server 构造时 `ReverbServiceImpl::Initialize` 或
`InProcessClient::LoadLatest` 已对这些 `Table` 调用 `LoadLatest`，
`ShmClient` 连接后自动看到已恢复的状态。即：调用 `ShmClient.checkpoint()`
保存后重启，恢复由 Server（gRPC 或 InProcess 路径）在构造时完成，SHM 客户端
不需要额外操作。

## 6. 传输层差异速查（供调试参考）

- **gRPC**：`Client` → gRPC stream → `Server` → `Table`。数据序列化为 numpy bytes
  走网络。`ServerInfo(timeout)` 带 timeout；`NewSampler` 无 timeout 参数。
- **Local**：`LocalClient` → `InProcessClient` → `Table`（直接指针，无序列化）。
  `NewSampler` 带 rate-limiter timeout。
- **SHM**：`ShmClient` → Unix Domain Socket bootstrap + mmap（pool + 四个
  per-flow 单生产者单消费者 ring）→ server 侧 `Table`。chunker/column 逻辑在
  client 侧，insert 走 SHM。`NewSampler` 走 `ShmSampler`（独立 worker 线程 +
  SHM ring 往返），带 timeout。server 死亡时通过 control fd 探测，返回
  `UnavailableError` 而非永久阻塞。
