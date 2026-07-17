# Reverb 客户端对比：gRPC `Client` / `LocalClient` / `ShmClient`

> 面向**使用者**的选型与避坑参考。三种客户端都继承 `_BaseClient`，共享
> `sample`/`insert`/`writer`/`mutate_priorities`/`reset`/`server_info`/
> `checkpoint` 的实现，但**传输路径、API 覆盖面、行为细节有真实差异**。
> 设计背景见 [numpy-embed-design.md](numpy-embed-design.md)（gRPC/Local）与
> [numpy-shm-design.md](numpy-shm-design.md)（SHM）。

## 0. TL;DR — 怎么选

| 场景 | 选哪个 | 构造 |
| --- | --- | --- |
| 同进程内嵌训练，零开销 | `LocalClient` | `Server(in_process=True).in_process_client` |
| 同机跨进程，要最快采样 | `ShmClient` | `reverb.ShmClient(server.shm_socket_path)` |
| 跨机 / 分布式 / 需要全套控制面 | gRPC `Client` | `reverb.Client('host:port')` |
| 需 pickle 客户端（如多进程 worker 持有） | 只能 gRPC `Client` | — |
| 需要 `checkpoint` | gRPC 或 `LocalClient` | — |
| 需要实时 `server_info`（反映 `Table.replace`） | gRPC 或 `LocalClient` | — |

一句话：**能内嵌就 `LocalClient`；要跨进程就要么 gRPC（图省事/要全套 API）要么
`ShmClient`（图采样性能）；要 pickle 只能 gRPC。**

## 1. 总览对比

| 维度 | `Client`（gRPC） | `LocalClient`（in_process） | `ShmClient`（shm） |
| --- | --- | --- | --- |
| 传输 / 拓扑 | 跨进程 / 跨机，gRPC stream，序列化 numpy bytes | **同进程**，直接持 `shared_ptr<Table>`，零序列化 | **同机跨进程**，POSIX shm + udsocket，mmap 零拷贝 |
| 性能 | 基线 | 最快（零网络 / 零序列化） | ~9–11× gRPC loopback（见 [shm-benchmark.md](shm-benchmark.md)） |
| 构造 | `Client('localhost:port')` | `server.in_process_client` | `reverb.ShmClient(server.shm_socket_path)` |
| 持有 | gRPC channel | 进程内 Table 指针 | SHM mmap + ring 状态 |
| `pickle` | ✅ 支持（存 `server_address`） | ❌ 不可（持进程内指针） | ❌ 不可（持 mmap + ring） |
| 多表 | ✅ 全部表 | ✅ 全部表 | ✅ 全部表（按表名路由，ticket ⑨） |
| `sample` 的 `timeout_ms` | ⚠️ **静默忽略**（gRPC `NewSampler` 无 timeout 参数） | ✅ 生效（超时抛 `DeadlineExceededError`） | ✅ 生效（超时抛 `DeadlineExceededError`） |
| `sample` 默认 `emit_timesteps` | `True` | `True` | `True`（三者统一） |
| `server_info` | ✅ 真实，带 timeout | ✅ 真实，忽略 timeout | ✅ **bootstrap 快照**（连接时缓存，无往返） |
| `mutate_priorities` / `reset` | ✅ | ✅ | ✅（ticket ⑩，走 insert 流 + 客户端互斥锁） |
| `checkpoint` / 恢复 | ✅ | ✅（`Server(in_process=True)` 构造时自动 `LoadLatest`） | ❌ 不支持 |
| `trajectory_writer` / `structured_writer` | ✅ | ✅ | ✅（chunker/column 在 client 侧，insert 走 SHM） |
| `writer`（legacy）/ `insert` | ✅ | ✅ | ❌ **`NewWriter` 返回 `UnimplementedError`** |

## 2. API 覆盖面

三者对 `_BaseClient` 方法的支持：

```
                    Client      LocalClient     ShmClient
sample              ✅          ✅              ✅
trajectory_writer   ✅          ✅              ✅
structured_writer   ✅          ✅              ✅
writer (legacy)     ✅          ✅              ❌ UnimplementedError
insert              ✅          ✅              ❌ (内部依赖 writer)
mutate_priorities   ✅          ✅              ✅（ticket ⑩）
reset               ✅          ✅              ✅（ticket ⑩）
server_info         ✅ 真实      ✅ 真实          ✅ bootstrap 快照
checkpoint          ✅          ✅              ❌
```

**`ShmClient` 继承了用不了的 `insert`/`writer`**：因为 `_BaseClient.insert`
内部调 `self.writer` → `self._client.NewWriter`，而 SHM 的 `NewWriter` 是
`UnimplementedError`。所以 `shm_client.insert(...)` 能写、能编译，**运行时抛错**。
SHM 写入只能走 `trajectory_writer` / `structured_writer`。

## 3. 三种使用样例

三者写数据的核心 API 完全一致，只有构造和传输不同。下面各给一个最小可运行样例。

### 3.1 gRPC `Client`（跨进程 / 跨机）

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

### 3.2 `LocalClient`（同进程内嵌，零开销）

```python
server = reverb.Server(tables=[...], in_process=True)   # 不起 gRPC，无端口
client = server.in_process_client                        # LocalClient

# 后续 trajectory_writer / sample 用法与 gRPC 完全一致
```

### 3.3 `ShmClient`（同机跨进程，零拷贝）

```python
server = reverb.Server(tables=[...], in_process=True, shm=True)
#   in_process=True 拥有 Table；shm=True 在其上叠加 SHM 传输。
#   注意：shm=True 不隐含 in_process=True，需显式开。
#   SHM 现支持全部表（按表名路由，ticket ⑨）。
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
| `True` | 忽略 | `False` | 纯内嵌，无 gRPC 无端口 | `LocalClient` |
| `True` | 忽略 | `True` | 内嵌 + SHM | `LocalClient` + `ShmClient` |
| `False` | 自动/指定 | `False` | 纯 gRPC server | gRPC `Client` |
| `False` | 自动/指定 | `True` | gRPC + SHM | gRPC `Client` + `ShmClient` |

约束：

- `in_process=True` 时 `server.port` 为 `None`，`localhost_client()` 抛错，只能
  `server.in_process_client`。
- `in_process=False` 时 `server.in_process_client` 抛错，用 `localhost_client()`
  或自行 `reverb.Client(f'localhost:{server.port}')`。
- `shm=True` 时 `server.shm_socket_path` 可用；`shm=False` 时为 `None`。

## 5. 常见陷阱

### 5.1 `ShmClient.insert` / `ShmClient.writer` 运行时抛错

`insert` 和 `writer` 是从 `_BaseClient` 继承的，但 SHM 的 `NewWriter` 返回
`UnimplementedError`。**SHM 写入只能用 `trajectory_writer` / `structured_writer`。**

```python
client = reverb.ShmClient(server.shm_socket_path)
client.insert(data, {'t': 1.0})   # ❌ 运行时 UnimplementedError
with client.trajectory_writer(3) as w: ...   # ✅
```

### 5.2 `ShmClient.server_info()` 返回 bootstrap 快照

`server_info()` 返回**连接时的 bootstrap 快照**——真实的 `TableInfo`（`max_size`/
`sampler_options`/`remover_options`/`signature`/`current_size` 等），由 SHM 握手的
`WelcomeResponse.server_info` 随手捎带，无额外往返。`timeout` 参数被接受（与
gRPC/Local hook 对齐）但忽略——数据在 `Connect` 时已缓存。

**限制**：快照在连接那一刻固定，**不反映会话中途的 `Table.replace` / 签名变更**
（ticket ⑧ step 2 会补一个按需 `SERVER_INFO` ring 往返解决）。需要实时元数据时用
gRPC / `LocalClient`。

**`sample(unpack_as_table_signature=True)` 现在可用**：此前因 `server_info()` 返回 `{}`
导致签名缓存为空、抛 `ValueError: Could not find table`；快照填入缓存后，带签名的表
可正常按签名解包。

### 5.3 `ShmClient` 多表路由（ticket ⑨）

`ShmServer` 构造时接收**全部表**，按表名路由：`sample`/`trajectory_writer` 的
`table` 参数直接定位目标表。引用未知表名时返回 `NOT_FOUND`（Python 侧表现为
`FileNotFoundError`）。多表与 gRPC / `LocalClient` 行为一致。

### 5.4 `timeout_ms` 行为不对称

`sample(table, num_samples, timeout_ms=...)`：

- gRPC `Client`：**静默忽略** `timeout_ms`（C++ `NewSampler` 无 timeout 参数）。
- `LocalClient` / `ShmClient`：**生效**，rate limiter 阻塞超时抛
  `reverb.errors.DeadlineExceededError`。

跨 transport 迁移代码时注意：在 gRPC 上"能等"的调用，换 SHM/Local 加了 `timeout_ms`
后可能开始抛超时。

### 5.5 只有 gRPC `Client` 能 pickle

`LocalClient` / `ShmClient` 都不可 pickle（持进程内指针 / mmap 状态）。
若要把客户端分发给多进程 worker（如 `multiprocessing.Pool` 的 initializer 持有客户端），
只能用 gRPC `Client`——它 pickle 时只存 `server_address`，子进程反序列化后重连。

### 5.6 `emit_timesteps` 默认值已统一

三者 `_default_emit_timesteps` 都是 `True`（早期 `LocalClient` 曾默认 `False`，
已修正）。显式传 `emit_timesteps=False` 取整条 trajectory。

## 6. 传输层差异速查（实现层）

供调试 / 理解报错时参考：

- **gRPC**：`Client` → gRPC stream → `Server` → `Table`。数据序列化为 numpy bytes
  走网络。`ServerInfo(timeout)` 带 timeout；`NewSampler` 无 timeout 参数。
- **Local**：`LocalClient` → `InProcessClient` → `Table`（直接指针，零序列化）。
  `NewSampler` 带 rate-limiter timeout。
- **SHM**：`ShmClient` → udsocket bootstrap + mmap（pool + 四个 per-flow SPSC
  ring）→ server 侧 `Table`。chunker/column 逻辑在 client 侧，insert 走 SHM。
  `NewSampler` 走 `ShmSampler`（独立 worker 线程 + SHM ring 往返），带 timeout。
  server 死亡时通过 control fd 探测，返回 `UnavailableError` 而非永久阻塞。

三者 pybind 命名：gRPC `Client` 绑 PascalCase；`InProcessClient` 同时绑 snake_case 与 PascalCase 双名别名（见 [numpy-embed-design.md §3.3](numpy-embed-design.md)）；`ShmClient` 的 `NewSampler` 用 PascalCase 与 `_BaseClient` 统一路径对齐。
