# Tickets: Reverb SHM 传输层

POSIX 共享内存传输层，作为 gRPC / in_process 之外的第三条路径。同机分进程下跳过
全部序列化与压缩。源 spec：`docs/numpy-shm-spec.md`（含决策 S1–S15、C1–C5、R1–R13）。

> **v1 状态：已完成。** ticket ①-⑦ 全部落地，`bazel test //reverb/cc/shm:*` 8/8 绿，
> Python e2e（`reverb/tests/shm_test.py`）通过。仅 ④ 的 plain Writer SHM 路径有意
> 延期（见 `[~]`，无 SHM seam，⑬ 改为 deprecate 不实现）。已知技术债（单线程
> dispatch、RunShmWorker 重复等）以 `// ponytail:` 注释标注在代码里，升级条件见
> 各注释。
>
> **v2 路线图：** ticket ⑧-⑬，补齐 v1 有意延期（spec S10）的冷路径控制面与多表
> 支持。优先级 P0→P4 标在各 ticket 标题；依赖链与批次建议见文末「v2 依赖图」。
> v1 依据 `docs/client-transports.md` §1 对比表与 `docs/numpy-shm-spec.md` §6。
>
> **当前进度（工作区未提交，2026-07-17）：**
>
> - ✅ ⑧ `server_info`（bootstrap 快照）、✅ ⑨ 多表、✅ ⑩ `mutate_priorities`+`reset`
>   ——三 ticket 同批落地，已提交 `74608c1`。C++ `//reverb/cc/shm:*` 8/8、Python
>   `shm_test.py` 35/35、`transport_parity_test.py` 5/5 全绿。
> - ✅ ⑩ **死锁已修**（本会话）：2026-07-17 复现并抓栈确认根因——**不是**
>   ticket 原猜的 `insert_flow_mu` 锁范围，而是**服务端单线程 dispatch 在
>   `HandleSample` 里阻塞于 `Table::Sample` 的 rate-limiter 无限等待**（队头阻塞），
>   导致该 client 的所有后续 insert/mutate/sample ACK 永远排不进 ring，client
>   `ReadBlocking` 100% CPU 忙等、`timeout` 杀不掉。按方向 A+C 修复：`HandleSample`
>   改异步（`EnqueSampleRequest` + `DrainPendingSamples` 镜像 insert 的
>   callback→outbox 模式）+ `ReadBlocking` 加 60s 硬上限兜底。回归测试
>   `ShmSampleDeadlockRegressionTest` 落地。C++ 8/8 + Python `shm_test.py`
>   31 例 + `transport_parity_test.py` 5/5 全绿，×3 跑无 flaky。决定性三栈与
>   A/B/C/D 方向选型见 ⑩ 的「**已确认根因**」小节。
> - ⬜ ⑪ `checkpoint`（P3，blockedBy ⑩ 已满足）、⑫ `pickle`（P4，一行）、⑬ deprecate
>   legacy `Writer`/`insert`（清理）——均未开始，可任选推进。
>
> **新 session 入口**：⑩ 死锁已修，可放心推进 ⑪/⑫/⑬。本会话改动未提交
> （`shm_server.{h,cc}`/`shm_client.cc`/`trajectory_writer.cc`/`shm_test.py`/
> `tickets.md`/`shm_connection.h`），跑 `git diff` 复核后提交。

---

## ① Ring + Bootstrap echo

**What to build:** 数据首次跨进程边界。一个 C++ 测试里，"server" 线程创建 Unix
domain socket + 一条 SPSC ring SHM 段，"client" 线程连 udsocket、完成握手
（Hello → Welcome）、经 C→S ring 写一条消息、server 读到后经 S→C ring 回一条、
client 读到。无 Table、无业务数据——只证明传输层可用。

**Blocked by:** None — can start immediately.

- [x] `reverb/cc/shm/shm_protocol.proto` 定义（HelloRequest/WelcomeResponse 等全套消息，见 spec §4），`reverb_cc_proto_library` 构建通过
- [x] `ring.{h,cc}`：`RingHeader`/`SlotHeader`/`Ring`，SPSC release/acquire 读写（spec §3.1 + §8.4），跨槽拼接
- [x] `bootstrap.{h,cc}`：udsocket bind/listen/accept（先 unlink 旧 sock 防 PID 复用，R7）、Hello/Welcome length-delimited proto 往返
- [x] `shm_connection.h`：pool + C→S + S→C 三段 mmap 的 RAII 包装
- [x] `ring_test.cc`：单槽、跨槽、capacity 满阻塞、多轮循环、seq 绕回
- [x] `bootstrap_test.cc`：双线程握手往返、段名格式校验、协议版本不匹配拒绝
- [x] 所有 `bazel test //reverb/cc/shm:*` 绿

---

## ② Byte pool round-trip

**What to build:** slab 分配器 + 引用计数跨进程可用。server 在 pool 里分配一块、
写字节、client 按偏移读回字节一致；refcount inc/dec → 0 回收；池满阻塞。复用 ①
的连接。本 ticket 确立分配权归属（决策 C4）：**server 独占分配器，client 经 ring
向 server 申请偏移**，client 以 RW mmap pool 但不自行分配。

**Blocked by:** ① Ring + Bootstrap echo

- [x] `byte_pool.{h,cc}`：`ShmBytePool::Create`（server，`O_CREAT|O_RDWR|O_EXCL`）/ `Open`（client，`O_RDWR`，RW mmap per C4）、slab 档位 `{64…4MB}`、free list、`Allocate`/`Deallocate`/`At`
- [x] 引用计数 `Ref`/`Unref`/`ReleaseAll`（server 进程内 `flat_hash_map`，非 SHM）
- [x] pool 满 → `Allocate` 阻塞（条件变量），有释放后唤醒
- [x] C→S ring 增 `ALLOCATE` 请求 / S→C ring 增 `ALLOCATE_RESP`（偏移）消息类型（C4：client 申请偏移的通道）
- [x] `byte_pool_test.cc`：slab 选档、分配/回收/再分配、refcount→0 回收、池满阻塞、多块同档位复用
- [x] 跨线程测试：server 分配写字节 → client 读回一致

---

## ③ Sample path: server→client, read-only

**What to build:** 第一份真实 Reverb 数据跨 SHM。`ShmServer` 持真实 `Table`（用现有
in_process 路径预灌数据），dispatch 线程轮询 client 的 C→S ring；`ShmClient.NewSampler`
→ `GetNextTrajectory` 返回正确 numpy。insert **暂不接**——server 预灌。

**Sampler 架构按决策 C5**：复用现有 `Sampler` 的 worker 线程 + `samples_` 队列，
只把 worker 内的 gRPC `SampleStream` 换成 SHM ring 往返，不做"调用者线程内联"。

**Blocked by:** ② Byte pool round-trip

- [x] `shm_server.{h,cc}`：`Create`/`Start`/`Stop`、dispatch 线程主循环（轮询所有 client C→S ring + udsocket EOF）、`HandleSample`（`Table::Sample` → `UnpackChunkColumnAndSlice` 在 dispatch 线程同步，A1 → memcpy 成品字节进 pool per C3 → S→C 写 `SAMPLE_RESP`）
- [x] `HandleRelease`：遍历 offsets `Unref`，归零 `Deallocate`
- [x] dispatch 非阻塞写 S→C：满则暂存 `ClientState.outbox`，跳过该 client（§8.7）
- [x] `shm_client.{h,cc}`：`Connect`（bootstrap + mmap 三段）、`NewSampler`、SHM worker：发 `SAMPLE` → 轮询 `SAMPLE_RESP` → 按 `ShmColumn.shm_offset` 读 pool 字节建 `TensorBuffer` → 组装完发 `RELEASE`（C5）
- [x] `shm_sample_test.cc`：同进程双线程，预灌 table → client sample → numpy 与写入一致
- [x] sample 超时返回 `ERROR(DEADLINE_EXCEEDED)` 映射 `reverb.errors.DeadlineExceededError`

---

## ④ Insert path: client→server

**What to build:** 写方向打通。`TrajectoryWriter` append→create_item→flush 经 SHM 落
进 server 的 `Table`，再用 ③ 的 sampler 读回验证。复用现有 `TrajectoryWriter` 的
`is_local_`/`RunLocalWorker` 缝作为第三模式（SHM）。backpressure 靠"等
`INSERT_ACK.offsets_to_release`"约束（决策 C2）。

**Blocked by:** ③ Sample path: server→client, read-only

- [x] `HandleInsert`：读 `ShmInsertRequest` → 按 `ShmChunkRef.specs` 拆多列 → `CompressTensorAsProto` 各列压缩 → `Table::InsertOrAssignAsync`（带 callback）→ callback 写 `INSERT_ACK`（含 `offsets_to_release`）
- [x] C4 insert 时序：client 申请偏移 → memcpy chunk 字节 → 发 `INSERT` → **等 `INSERT_ACK.offsets_to_release` 才释放偏移**（C2）
- [x] `ShmClient.NewTrajectoryWriter`：复用 chunker/column 逻辑，`RunLocalWorker` 的 `InsertOrAssignAsync` 换成 SHM insert 往返；`local_can_insert_more_`/`num_items_in_flight_` 靠 ACK 递减
- [~] `ShmClient.NewWriter`（plain Writer）+ `NewStructuredWriter`（经 `PrepareStructuredWriterConfigs`）— StructuredWriter 完成；plain Writer 延期（无 SHM seam，见 `// ponytail:` TODO）
- [x] 端到端测试：client writer 写 → server table 有数据 → client sampler 读回一致；structured_writer 多表写入；backpressure 触发（in_flight 满 writer 阻塞）

---

## ⑤ Python API: `ShmClient` + `Server(shm=True)`

**What to build:** 用户可见的 Python 接口。`Server(tables, shm=True)` 启 `ShmServer`
（决策 C1，挂在 `Server` 对象上，`stop()`/`__del__` 销毁）；`ShmClient(server.shm_socket_path)`
的 `sample`/`insert`/`trajectory_writer`/`structured_writer` 与 `Client`/`LocalClient`
语义一致，三路无缝迁移。

**Blocked by:** ④ Insert path: client→server

- [x] `reverb/pybind.cc`：`PyShmClient` 包装类 + `PascalCase`/`snake_case` 双名绑定（对齐 `InProcessClient`），复用已有 `_import_array()`（R13）
- [x] `reverb/client.py`：`ShmClient(_BaseClient)`，实现 `_fetch_server_info_proto`/`_new_sampler`/`trajectory_writer`/`structured_writer` 钩子
- [x] ~~`reverb/shm_server.py`：`pybind.ShmServer` 的 Python 包装~~ — 简化：`ShmServer` 直接由 pybind 暴露，`server.py` 内联持有 `pybind.ShmServer`，无需独立包装层
- [x] `reverb/server.py`：`Server` 加 `shm=False`/`shm_socket_path=None` 参数，`shm=True` 时起 `ShmServer`，暴露 `shm_socket_path` 属性
- [x] `ShmClient` 不可 pickle（持 SHM mmap 指针，对齐 `LocalClient`）
- [x] Python 测试：`Server(shm=True)` + `ShmClient` 完整 sample/insert/writer 往返，与 `LocalClient` 行为对齐

---

## ⑥ Crash recovery + cleanup

**What to build:** 可靠性。udsocket EOF 检测断连；client 崩溃时 server 遍历该 client
`outstanding_offsets_` 集中释放、`shm_unlink` 其 ring 段；其他 client 不受影响；
server 启动清旧 sock / 旧 SHM 段（R6/R7/R12）。新 client 可干净重连。

**Blocked by:** ⑤ Python API: `ShmClient` + `Server(shm=True)`

- [x] server 监听 udsocket EOF/ECONNRESET → `HandleDisconnect`：`ReleaseAll`、`shm_unlink` 两条 ring、销毁 `ClientState`
- [x] server `HandleClose`（client 主动 `CLOSE` 消息）走同一清理路径
- [x] `ShmBootstrapServer::Create` 先 `unlink(socket_path)` 再 bind（R7）；`ShmBytePool::Create` 用 `O_EXCL`，失败 `shm_unlink` 旧名重试（R6）
- [x] server 收 SIGTERM/SIGINT（经现有 `Server.stop()` 路径扩展）清理所有 SHM 段 + udsocket（R12）
- [x] client 读 udsocket EOF → 抛 `ConnectionError`，在途请求全失败
- [x] 崩溃恢复测试：kill client → server 无泄漏（outstanding 清空、ring 可 unlink、其他 client 不受影响）→ 新 client 重连正常

---

## ⑦ Performance baseline + v2 spike

**What to build:** 度量而非功能。gRPC loopback vs SHM 的 sample throughput/latency
基准；v2 优化（insert 字节复用为 sample 切片源，省"压缩进 ChunkStore 再解压"往返，
spec §6）的 spike 文档 + 是否值得做的决策。

**Blocked by:** ⑥ Crash recovery + cleanup

- [x] 基准脚本：同表同数据，分别走 `Client('localhost:port')` / `ShmClient`，测 sample throughput（samples/s）+ p50/p99 latency
- [x] 结果落 `docs/shm-benchmark.md`，确认 SHM 在 sample 路径（原设计核心痛点：N 次重复解压）有 measurable 提升
- [x] v2 spike 文档：server 维持 `chunk_key → SHM 偏移` 索引、sample 直接基于 insert 原始字节切片的可行性 + 复杂度评估
- [x] 决策：v2 值不值得做，写进 spike 文档结论（DEFER — v1 已达无序列化天花板，v2 边际收益受限于解压开销，见 `docs/shm-benchmark.md` §4）

---

## v2 路线图（⑧-⑬）

补齐 v1 有意延期（spec S10）的冷路径控制面、多表、pickle，并关闭 ④ 遗留的
legacy Writer。优先级按「用户影响 × 实现成本 × 依赖链」排：P0 最高。每条是
可独立交付的 tracer-bullet，blockedBy 声明依赖边，任一 ticket 的 blocker 完成即可
grab。

---

## ⑧ `server_info` 真实往返  [P0]

**What to build:** `ShmClient.server_info()` 返回真实 `TableInfo` 字典而非空 `{}`。
这是当前最坏的失败模式——静默返回空，用户不知道它坏了，且连锁导致
`sample(unpack_as_table_signature=True)` 抛 `ValueError`、
`trajectory_writer(validate_items=True)` 形同虚设（`flat_signature_map` 永远空）。

分两步交付：先 bootstrap 时填一次（覆盖表签名 rarely change 的场景，零新消息
类型），再加 `SERVER_INFO` 往返支持按需刷新（对齐 gRPC「每次调用刷新」语义）。
`WelcomeResponse.server_info` 字段 proto 已存在，第一步成本极低。

**Blocked by:** None — 可立即开始。

**Step 1 状态：已完成。** bootstrap piggyback 落地，`server_info()` 返回连接时快照，
`sample(unpack_as_table_signature=True)` 修复。C++ 8/8 + Python 19/19 绿。
Step 2（按需 SERVER_INFO 往返）仍在下面，未做。

### Step 1 — bootstrap 快照  [已完成]

- [x] `ShmServer::TryAccept` 填 `welcome.mutable_server_info()`：`*add_table_info() = table_->info()`（`reverb/cc/shm/shm_server.cc`）
- [x] `ShmClient::Connect` 缓存 `welcome.server_info().table_info()` 进 `cached_server_info_`；新增 `ServerInfo(vector<TableInfo>*)` 返回快照（`shm_client.{h,cc}`）
- [x] pybind `PyShmClient` 增 `server_info`/`ServerInfo` 双名绑定，返回 `vector<py::bytes>`（镜像 InProcessClient，`reverb/pybind.cc` + `pybind.pyi`）
- [x] `ShmClient._fetch_server_info_proto` 从 `return []` 改为 `return self._client.ServerInfo()`（`reverb/client.py`）
- [x] `BUILD` 加 `schema_cc_proto` 依赖（`TableInfo` proto）
- [x] 测试：`ShmClientServerInfoTest` 5 例（真实 metadata / connect-time size / signature / `unpack_as_table_signature` 回归 / timeout kwarg），`reverb/tests/shm_test.py`
- [x] 更新 `docs/client-transports.md` §1 表 + §2 + §5.2；`README.md` 同步
- [x] `ponytail:` 注释标注上限：快照不反映会话中途 `Table.replace`/签名变更；升级路径 = step 2

### Step 2 — 按需 `SERVER_INFO` 往返  [deferred]

- [ ] `shm_protocol.proto` 分配 `SERVER_INFO` / `SERVER_INFO_RESP` type 值（spec §8.3 未占号；注意 5 已被 `ALLOCATE` 占用）
- [ ] `ShmServer` dispatch 增 `HandleServerInfo`：序列化各表**当前** `TableInfo` 写 S→C
- [ ] `ShmClient._fetch_server_info_proto` 改为发 `SERVER_INFO` 阻塞读 `SERVER_INFO_RESP`（按需刷新，对齐 gRPC「每次调用刷新签名缓存」）
- [ ] 表签名变更（`Table.replace`）后 `server_info()` 能反映新签名
- [ ] `NewTrajectoryWriter` 把缓存的 `TableInfo` signature 填进 `TrajectoryWriter::Options.flat_signature_map`，使 `validate_items=True` 生效（step 1 已暴露数据，但 writer options 未接，需后续 wiring）
- [ ] 测试：`Table.replace` 后 `server_info()` 反映新签名；`trajectory_writer(validate_items=True)` 拒绝不匹配 trajectory

---

## ⑨ 多表支持  [P1]  [已完成]

**What to build:** `ShmServer` 持 `map<string, shared_ptr<Table>>` 而非单个
`tables[0]`。sample/insert 按消息里已有的 `table` 字段路由（`ShmSampleRequest.table`、
`PrioritizedItem.table`，协议层无需改）。RL 多表场景（uniform + prioritized、短序列

- 长序列）一上手就会撞到 v1 的单表限制，不解决就只能退回 gRPC/Local。

**Blocked by:** None — 协议已带表名，独立于 ⑧。建议与 ⑧/⑩ 同轮做（都是 dispatch
扩展）。

**状态：已完成。** `ShmServer` 持 `flat_hash_map<string, shared_ptr<Table>> tables_`，
`FindTable` 作为共享路由 seam（⑩ 复用）。C++ 8/8 + Python 28/28 绿。未知表名走
`ShmError::NOT_FOUND` → `absl::NotFoundError` → Python `FileNotFoundError`。

- [x] `ShmServer` 把 `shared_ptr<Table> table_` 换 `flat_hash_map<string, shared_ptr<Table>> tables_`，构造时接收 `vector<shared_ptr<Table>>` + 唯一名校验（`shm_server.{h,cc}`）
- [x] `FindTable(name)` 私有 helper 返回 `StatusOr<shared_ptr<Table>>`，缺失返 `NotFoundError`——路由 seam，⑩ 复用
- [x] `HandleSample` 先 `FindTable(req.table())`，缺失写 `ERROR(NOT_FOUND)` 到 sample s2c（镜像 DEADLINE_EXCEEDED 路径）
- [x] `HandleInsert` 移除单表 warn-and-skip 守卫，改为 per-item `FindTable`；ANY item 未知表 → 整请求 `ERROR(NOT_FOUND)`（比旧的静默丢弃更严）
- [x] `TryAccept` 的 server_info 填充从 `table_->info()` 改为遍历 `tables_`（ticket ⑧ 的多表版）
- [x] `ShmSampler::FetchOne` + `RunShmWorker` 把 `ShmError::NOT_FOUND`/`INVALID_ARGUMENT` 映射到 `absl::NotFoundError`/`InvalidArgumentError`（`shm_client.cc` + `trajectory_writer.cc`），不再裹成 `InternalError`
- [x] pybind `ShmServer.__init__`/`Create` 从 `table: Table` 改 `tables: Sequence[Table]`（`pybind.cc` + `pybind.pyi`）
- [x] `Server(shm=True)` Python 层传 `[t.internal_table for t in tables]`（`server.py`）
- [x] C++ 测试更新 `{table}` vector 形式（`shm_crash_test.cc`/`shm_insert_test.cc`/`shm_sample_test.cc`）
- [x] Python 测试 `ShmMultiTableTest` 5 例：sample 路由 / insert 路由 / server_info 列全表 / 未知表 sample 抛 `FileNotFoundError` / 未知表 insert 抛 `FileNotFoundError`（`shm_test.py`）
- [x] 更新 `docs/client-transports.md` §1 表 + §5.3（多表 ✅）；`README.md`；`docs/numpy-shm-design.md` §6（标「已实现」）；`transport_parity_test.py` docstring
- [x] `ponytail:` 注释：map-only 无并行有序 list（server_info 消费为 dict，顺序无关）；升级条件 = 有测试断言 table_info 顺序时再加
- [x] 路由基础设施预留给 ⑩ 的 `mutate_priorities`/`reset` 复用（`FindTable` seam + 注释）

---

## ⑩ `mutate_priorities` + `reset`  [P2]  [已完成，死锁已修]

**What to build:** PER 训练每步都要 `mutate_priorities`，算半热路径。新增两个控制
消息对 + dispatch handler，调现有 `Table::MutateItems` / `Table::Reset`（Table 侧
API 已存在）。两者机制相似，一起做省一次协议改动。有 gRPC fallback（控制面负载
小，gRPC 开销可忍），故非硬阻塞。

**注意 type 值分配**：spec §8.3 早期把 type 5 预留给 `MUTATE_PRIORITIES`，但 v1
实现中 5 被 `ALLOCATE` 占用（C4）。需为 `MUTATE_PRIORITIES`/`RESET`/`MUTATE_ACK`/
`RESET_ACK` 重新分配未占 type 值（spec §8.3 注释已说明此偏移）。

**Blocked by:** ⑨ 多表（复用 table 名路由基础设施；否则单表下做完 ⑩ 还要返工）。

**状态：已完成。** `MUTATE_PRIORITIES=6`/`RESET=7`/`MUTATE_ACK=106`/`RESET_ACK=107`
落地，复用现有 `MutatePrioritiesRequest`/`ResetRequest` proto。控制面经 insert
flow（`insert_c2s`/`insert_s2c`）往返，client 侧 `ShmConnection::insert_flow_mu`
串行化 send→read-ACK 以保 SPSC 不变式（决策 D）。C++ 8/8 + Python 35/35 绿
（含并发 mutate+insert 不坏环回归测试）。但 `insert_flow_mu` 的锁范围过大是
已知技术债，见下。

- [x] `shm_protocol.proto` 分配 `MUTATE_PRIORITIES=6` / `RESET=7` / `MUTATE_ACK=106` / `RESET_ACK=107`（注意 5 已被 ALLOCATE 占用）
- [x] C→S 复用现有 `MutatePrioritiesRequest` / `ResetRequest` proto；S→C 空 ACK
- [x] `ShmServer` `HandleMutatePriorities`：`FindTable` 路由 → `Table::MutateItems` → `EnqueueInsertS2C(MUTATE_ACK)`；未知表 `ERROR(NOT_FOUND)`
- [x] `ShmServer` `HandleReset`：`FindTable` 路由 → `Table::Reset` → `EnqueueInsertS2C(RESET_ACK)`；未知表 `ERROR(NOT_FOUND)`
- [x] dispatch `HandleInsertRequests` switch 增 `case MUTATE_PRIORITIES` / `case RESET`
- [x] `ShmClient` `MutatePriorities`/`Reset`：持 `insert_flow_mu` 覆盖 send→read-ACK，错误映射镜像 `FetchOne`（NOT_FOUND→NotFoundError→Python FileNotFoundError）
- [x] `RunShmWorker`（trajectory_writer.cc）的 ALLOCATE/INSERT round-trip 也持 `insert_flow_mu`，避免两个生产者写 `insert_c2s` 的 `head`
- [x] pybind `ShmClient` 暴露 `MutatePriorities`/`mutate_priorities` + `Reset`/`reset` 双名（镜像 InProcessClient）
- [x] `_BaseClient.mutate_priorities`/`reset` 经 duck-typing 直接走通，Python 无改动
- [x] 测试 `ShmMutateResetTest` 6 例：mutate 改 priority / mutate 删 item / reset 清表 / 未知表 mutate 抛错 / 未知表 reset 抛错 / **并发 mutate+insert 不坏环**
- [x] 更新 `docs/client-transports.md` §0/§1/§2 + `README.md` + `docs/numpy-shm-design.md`§6 + `docs/numpy-shm-spec.md`§6（mutate/reset 从 ❌ 改 ✅）

### 已知技术债：`insert_flow_mu` 锁范围过大（疑似偶发死锁，未复现）

**现象**：另一个 pi 会话跑 ⑩ 的最小复现脚本（`insert → flush → sample →
mutate_priorities(updates={key:99}) → sample`）时观察到卡死——子进程 100% CPU、
35 线程、`timeout 40` 杀不掉（主线程在 C++ `ReadBlocking` 忙等，GIL 释放，
信号处理跑不起来）。本会话单跑 `test_mutate_and_reset_parity` 20 次 +
`ShmMutateResetTest` 全绿，**未复现**。卡死进程跑的代码版本不明（可能是 ⑩
中间状态），无法确证是当前落地代码。

**静态分析发现的真设计缺陷**（无论是否为该次卡死主因，都该修）：
`insert_flow_mu` 的锁范围覆盖了 `RunShmWorker` 里「写请求 + **读 ACK**」的整个
round-trip（trajectory_writer.cc:973-1156，含 line 1012/1107 的
`read_blocking(insert_s2c)`）。后果：

- `mutate_priorities`（主线程）持锁等 `MUTATE_ACK` 时，`RunShmWorker`（后台
  线程）无法获取锁去读**前一个 INSERT 的 ACK**；
- 前一个 ACK 堆在 `insert_s2c`，`FlushOutbox` 的 `TryWrite` 失败（ring 满），
  `MUTATE_ACK` 留在 outbox 永远 flush 不进；
- client `ReadBlocking` 100% CPU spin 等 `MUTATE_ACK`，形成「mutate 等 ACK →
  ACK 在 ring → ring 满 → 等 worker 读 → worker 等锁 → 锁被 mutate 持」的循环。

正常顺序调用（`flush` 等 ACK 返回再 `mutate`）下循环不闭合，故单测全绿；并发
或乱序时可能触发。

> **2026-07-17 复现 + 抓栈结果：上面的 `insert_flow_mu` 分析是错的，根因另在。**
> 见下一小节「**已确认根因**」。

### 已确认根因：单线程 dispatch 在 `HandleSample` 的 rate-limiter 阻塞，造成队头阻塞

**复现**：`min_size=50` 的表只插入 1 条 → 一个线程 `sample(num_samples=1)`
（`timeout_ms=None → InfiniteDuration`）→ 另一线程 `mutate_priorities`。
秒级 100% CPU 双线程卡死，`timeout` 杀不掉（`ReadBlocking` 的 `sched_yield`
忙等，GIL 已释放，Python 信号处理跑不起来）。35 线程，与 ticket 顶部观察一致。

**`gdb thread apply all bt` 抓到决定性三栈**（见 `reverb/tests/shm_deadlock_repro.py`）：

- **dispatch 线程**：`DispatchLoop → HandleSample → Table::Sample →
  SampleFlexibleBatch → Notification::WaitForNotification()`——在 rate limiter
  上**无限等待**（`InfiniteDuration`）。这一条阻塞了服务端**唯一的** dispatch
  线程。
- **sampler worker（100% CPU）**：`RunWorker → FetchOne → ReadBlocking`，
  `sched_yield` 忙等 `sample_s2c` 的 `SAMPLE_RESP`——dispatch 被卡，永无响应。
- **mutate 主线程（100% CPU）**：`MutatePriorities → ReadBlocking`，
  `sched_yield` 忙等 `insert_s2c` 的 `MUTATE_ACK`——dispatch 被卡，永处理不到。

**`insert_flow_mu` 是无辜的**——它序列化整个 round-trip 反而 *防止* 两个生产者
同时写 `insert_c2s`。真正的循环是**服务端单线程 dispatch 在 `HandleSample` 里
阻塞于 `Table::Sample` 的 rate-limiter 等待**（队头阻塞），导致该 client 的所有
后续 insert/mutate/sample ACK 永远排不进 ring。gRPC 不受此害是因为它每请求一个
独立 RPC 线程，不共享单条 dispatch 线程。

**修复方向**（已选 A+C，2026-07-17 落地）：
1. **A. 服务端 `HandleSample` 异步化** ✅ 已实施：`HandleSample` 改调
   `Table::EnqueSampleRequest` 入 table worker 异步队列，dispatch 不阻塞；完成
   回调在 table worker 线程把 `SampledItem`+status 攒进
   `ClientState::pending_samples`（mutex 保护），dispatch 线程每轮
   `DrainPendingSamples` 取出做 unpack+pool+`SAMPLE_RESP`（保持 `pool_`/
   `outstanding_offsets_` 单线程不变式，镜像 `HandleInsert` 的 callback→outbox）。
   keepalive `shared_ptr<SamplingCallback>` 存 `pending_sample_callbacks` vector，
   回调触发时按裸指针 key 自清 erase（裸指针存堆上 `shared_ptr<Callback*>`
   控制块按值捕获，解决「make_shared 后才有值 + 局部变量按引用捕获悬空」的
   bootstrap 问题）。不用 FIFO 弹出——table worker 可能乱序完成请求（rate
   limiter 不满足时放回 current_sampling），FIFO 会弹错 keepalive 导致
   weak_ptr.lock() 失效、flaky hang。
2. **B. client 默认 `rate_limiter_timeout` 改有限值** ⬜ 未做：A 已根治，无需。
3. **C. client `ReadBlocking` 加超时** ✅ 已实施：`shm_client.cc` 的
   `ReadBlocking` 加 `kReadBlockingHardCap=60s`（sampler 用
   `rate_limiter_timeout_ + hardCap`，mutate/reset 用 hardCap）；
   `trajectory_writer.cc` 的 `read_blocking` lambda 加 `kInsertAckTimeout=60s`。
   超时返 `DeadlineExceededError`，让 `timeout` 能杀、Python 信号能跑。
4. **D. 第三条专用控制 ring** ⬜ 未做：A 已解 sample 队头阻塞，控制面不再争
   dispatch。控制面 throughput 真有问题时再加。

**回归测试**：`ShmSampleDeadlockRegressionTest.test_sample_blocking_rate_limiter_
does_not_deadlock_mutate`（`shm_test.py`）——`min_size=50` + 无 timeout sample +
并发 mutate，断言 mutate 10s 内返回（远小于 60s 兜底，证明是 A 的根治而非 C
超时）。

**剩余可选**：B/D 不再必要；如未来 dispatch 单线程成吞吐瓶颈，考虑 per-client
dispatch 线程（spec §8.7 原始升级路径）。

---

## ⑪ `checkpoint` / 恢复  [P3]

**What to build:** `ShmClient.checkpoint()` 触发 server 落盘并返回路径。难点不在
协议，而在集成：checkpointer 现归 Python `Server` 对象所有，不在 `ShmServer`——
需把 checkpointer 句柄或 `std::function<string()>` 回调注入 `ShmServer`，经
`CHECKPOINT`/`CHECKPOINT_RESP` 消息触发。长训练恢复才需要；SHM 常用于瞬态高速
缓冲，要持久化的场景往往本就用 gRPC/Local，故优先级靠后。

**Blocked by:** ⑩（同属控制面冷路径，复用协议扩展模式 + ⑨ 的 table 路由）。

- [ ] `ShmServer` 持 checkpointer 引用或 `std::function<string()>` 回调（由 `Server` Python 侧注入，对齐 `LocalClient` 路径）
- [ ] `shm_protocol.proto` 分配 `CHECKPOINT` / `CHECKPOINT_RESP` type 值，复用现有 `CheckpointRequest`/`CheckpointResponse`
- [ ] `HandleCheckpoint`：调 checkpointer → 返回路径
- [ ] `ShmClient` pybind + Python 暴露 `Checkpoint`，`_BaseClient.checkpoint` 走通
- [ ] 恢复路径：`Server(shm=True)` 构造时 `LoadLatest`（对齐 `LocalClient`，见 `docs/numpy-embed-design.md`）
- [ ] 测试：checkpoint → 新 server load → sample 恢复数据一致
- [ ] 更新 `docs/client-transports.md` §1 表（`checkpoint` 从 ❌ 改 ✅）

---

## ⑫ `pickle` 支持  [P4]

**What to build:** `ShmClient` 可 pickle，反序列化后按 `socket_path` 重连。改
`__reduce__` 从 raise 改为返回构造器。需求低（要分发到多进程 worker 时 gRPC
`Client` 已覆盖；各 worker 自建 `ShmClient(socket_path)` 也行），但成本极低（一行），
可随时插入任何批次。

**Blocked by:** None — 完全独立。

- [ ] `ShmClient.__reduce__` 改为 `return (self.__class__, (self._socket_path,))`
- [ ] 确认反序列化后 `Connect` 重连语义正确（无残留 mmap/ring 泄漏；旧 fd 正确关闭）
- [ ] 测试：`pickle.dumps`/`loads` 后 sample/insert 可用；原 client 仍可用
- [ ] 更新 `docs/client-transports.md` §1 表与 §5.5（pickle 从「只能 gRPC」改「gRPC + ShmClient」）

---

## ⑬ Deprecate legacy `Writer`/`insert` for ShmClient  [清理，不实现]

**What to build:** 不实现 plain `Writer` 的 SHM 路径（`shm_client.h` `TODO(④)`，
无 SHM seam，为 legacy API 复刻 `RunShmWorker` 不划算）。改为在 Python 层把
`ShmClient.insert`/`writer` 覆盖成抛清晰 `NotImplementedError`（带「用
trajectory_writer」提示），避免用户撞 C++ 运行时 `UnimplementedError`。

**Blocked by:** None — 纯清理，不 block 也不被 block。

- [ ] `ShmClient` 覆盖 `writer`/`insert`，抛 `NotImplementedError("ShmClient 不支持 legacy writer/insert，请用 trajectory_writer 或 structured_writer")`
- [ ] 移除 `shm_client.h` 的 `TODO(④)`（改为「won't fix — legacy API, use trajectory_writer」注释）
- [ ] 更新 `docs/client-transports.md` §5.1（从「运行时抛 UnimplementedError」改为「Python 层显式 NotImplementedError」）
- [ ] 更新 `docs/numpy-shm-design.md` §6 与 `docs/numpy-shm-spec.md` §6 的 plain Writer 延期说明（标 won't fix）

---

## v2 依赖图

```
⑧ server_info  [P0]  ──┐
                        ├─ 可同一轮（都是 dispatch + 协议扩展）
⑨ 多表        [P1]  ──┤
                        │
⑩ mutate/reset [P2]  ───┘  blockedBy ⑨
        │
        └── ⑪ checkpoint [P3]  blockedBy ⑩

⑫ pickle      [P4]  ──  独立，可随时插入
⑬ deprecate   [清理] ──  独立，不实现
```

**批次建议**：⑧+⑨+⑩ 一轮（dispatch 扩展，做完 ShmClient 对齐 gRPC/LocalClient
常用面）；⑪ 独立一轮（需 checkpointer 集成）；⑫/⑬ 顺手做。

**关键判断**：⑧/⑨/⑩ 是「让 ShmClient 成为可用客户端」的必经三步；⑪/⑫ 是锦上
添花；⑬ 是该砍不是该补。
