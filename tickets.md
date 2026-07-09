# Tickets: Reverb SHM 传输层

POSIX 共享内存传输层，作为 gRPC / in_process 之外的第三条路径。同机分进程下跳过
全部序列化与压缩。源 spec：`docs/numpy-shm-spec.md`（含决策 S1–S15、C1–C5、R1–R13）。

Work the **frontier**: any ticket whose blockers are all done. 链近似线性
①→②→③→④→⑤→⑥→⑦，每个 ticket 在独立 fresh context 里用 `/implement` 推进，做完
清上下文再领下一个。

---

## ① Ring + Bootstrap echo

**What to build:** 数据首次跨进程边界。一个 C++ 测试里，"server" 线程创建 Unix
domain socket + 一条 SPSC ring SHM 段，"client" 线程连 udsocket、完成握手
（Hello → Welcome）、经 C→S ring 写一条消息、server 读到后经 S→C ring 回一条、
client 读到。无 Table、无业务数据——只证明传输层可用。

**Blocked by:** None — can start immediately.

- [ ] `reverb/cc/shm/shm_protocol.proto` 定义（HelloRequest/WelcomeResponse 等全套消息，见 spec §4），`reverb_cc_proto_library` 构建通过
- [ ] `ring.{h,cc}`：`RingHeader`/`SlotHeader`/`Ring`，SPSC release/acquire 读写（spec §3.1 + §8.4），跨槽拼接
- [ ] `bootstrap.{h,cc}`：udsocket bind/listen/accept（先 unlink 旧 sock 防 PID 复用，R7）、Hello/Welcome length-delimited proto 往返
- [ ] `shm_connection.h`：pool + C→S + S→C 三段 mmap 的 RAII 包装
- [ ] `ring_test.cc`：单槽、跨槽、capacity 满阻塞、多轮循环、seq 绕回
- [ ] `bootstrap_test.cc`：双线程握手往返、段名格式校验、协议版本不匹配拒绝
- [ ] 所有 `bazel test //reverb/cc/shm:*` 绿

---

## ② Byte pool round-trip

**What to build:** slab 分配器 + 引用计数跨进程可用。server 在 pool 里分配一块、
写字节、client 按偏移读回字节一致；refcount inc/dec → 0 回收；池满阻塞。复用 ①
的连接。本 ticket 确立分配权归属（决策 C4）：**server 独占分配器，client 经 ring
向 server 申请偏移**，client 以 RW mmap pool 但不自行分配。

**Blocked by:** ① Ring + Bootstrap echo

- [ ] `byte_pool.{h,cc}`：`ShmBytePool::Create`（server，`O_CREAT|O_RDWR|O_EXCL`）/ `Open`（client，`O_RDWR`，RW mmap per C4）、slab 档位 `{64…4MB}`、free list、`Allocate`/`Deallocate`/`At`
- [ ] 引用计数 `Ref`/`Unref`/`ReleaseAll`（server 进程内 `flat_hash_map`，非 SHM）
- [ ] pool 满 → `Allocate` 阻塞（条件变量），有释放后唤醒
- [ ] C→S ring 增 `ALLOCATE` 请求 / S→C ring 增 `ALLOCATE_RESP`（偏移）消息类型（C4：client 申请偏移的通道）
- [ ] `byte_pool_test.cc`：slab 选档、分配/回收/再分配、refcount→0 回收、池满阻塞、多块同档位复用
- [ ] 跨线程测试：server 分配写字节 → client 读回一致

---

## ③ Sample path: server→client, read-only

**What to build:** 第一份真实 Reverb 数据跨 SHM。`ShmServer` 持真实 `Table`（用现有
in_process 路径预灌数据），dispatch 线程轮询 client 的 C→S ring；`ShmClient.NewSampler`
→ `GetNextTrajectory` 返回正确 numpy。insert **暂不接**——server 预灌。

**Sampler 架构按决策 C5**：复用现有 `Sampler` 的 worker 线程 + `samples_` 队列，
只把 worker 内的 gRPC `SampleStream` 换成 SHM ring 往返，不做"调用者线程内联"。

**Blocked by:** ② Byte pool round-trip

- [ ] `shm_server.{h,cc}`：`Create`/`Start`/`Stop`、dispatch 线程主循环（轮询所有 client C→S ring + udsocket EOF）、`HandleSample`（`Table::Sample` → `UnpackChunkColumnAndSlice` 在 dispatch 线程同步，A1 → memcpy 成品字节进 pool per C3 → S→C 写 `SAMPLE_RESP`）
- [ ] `HandleRelease`：遍历 offsets `Unref`，归零 `Deallocate`
- [ ] dispatch 非阻塞写 S→C：满则暂存 `ClientState.outbox`，跳过该 client（§8.7）
- [ ] `shm_client.{h,cc}`：`Connect`（bootstrap + mmap 三段）、`NewSampler`、SHM worker：发 `SAMPLE` → 轮询 `SAMPLE_RESP` → 按 `ShmColumn.shm_offset` 读 pool 字节建 `TensorBuffer` → 组装完发 `RELEASE`（C5）
- [ ] `shm_server_test.cc` + `shm_client_test.cc`：同进程双线程，预灌 table → client sample → numpy 与写入一致
- [ ] sample 超时返回 `ERROR(DEADLINE_EXCEEDED)` 映射 `reverb.errors.DeadlineExceededError`

---

## ④ Insert path: client→server

**What to build:** 写方向打通。`TrajectoryWriter` append→create_item→flush 经 SHM 落
进 server 的 `Table`，再用 ③ 的 sampler 读回验证。复用现有 `TrajectoryWriter` 的
`is_local_`/`RunLocalWorker` 缝作为第三模式（SHM）。backpressure 靠"等
`INSERT_ACK.offsets_to_release`"约束（决策 C2）。

**Blocked by:** ③ Sample path: server→client, read-only

- [ ] `HandleInsert`：读 `ShmInsertRequest` → 按 `ShmChunkRef.specs` 拆多列 → `CompressTensorAsProto` 各列压缩 → `Table::InsertOrAssignAsync`（带 callback）→ callback 写 `INSERT_ACK`（含 `offsets_to_release`）
- [ ] C4 insert 时序：client 申请偏移 → memcpy chunk 字节 → 发 `INSERT` → **等 `INSERT_ACK.offsets_to_release` 才释放偏移**（C2）
- [ ] `ShmClient.NewTrajectoryWriter`：复用 chunker/column 逻辑，`RunLocalWorker` 的 `InsertOrAssignAsync` 换成 SHM insert 往返；`local_can_insert_more_`/`num_items_in_flight_` 靠 ACK 递减
- [ ] `ShmClient.NewWriter`（plain Writer）+ `NewStructuredWriter`（经 `PrepareStructuredWriterConfigs`）
- [ ] 端到端测试：client writer 写 → server table 有数据 → client sampler 读回一致；structured_writer 多表写入；backpressure 触发（in_flight 满 writer 阻塞）

---

## ⑤ Python API: `ShmClient` + `Server(shm=True)`

**What to build:** 用户可见的 Python 接口。`Server(tables, shm=True)` 启 `ShmServer`
（决策 C1，挂在 `Server` 对象上，`stop()`/`__del__` 销毁）；`ShmClient(server.shm_socket_path)`
的 `sample`/`insert`/`trajectory_writer`/`structured_writer` 与 `Client`/`LocalClient`
语义一致，三路无缝迁移。

**Blocked by:** ④ Insert path: client→server

- [ ] `reverb/pybind.cc`：`PyShmClient` 包装类 + `PascalCase`/`snake_case` 双名绑定（对齐 `InProcessClient`），复用已有 `_import_array()`（R13）
- [ ] `reverb/client.py`：`ShmClient(_BaseClient)`，实现 `_fetch_server_info_proto`/`_new_sampler`/`trajectory_writer`/`structured_writer` 钩子
- [ ] `reverb/shm_server.py`：`pybind.ShmServer` 的 Python 包装
- [ ] `reverb/server.py`：`Server` 加 `shm=False`/`shm_socket_path=None` 参数，`shm=True` 时起 `ShmServer`，暴露 `shm_socket_path` 属性
- [ ] `ShmClient` 不可 pickle（持 SHM mmap 指针，对齐 `LocalClient`）
- [ ] Python 测试：`Server(shm=True)` + `ShmClient` 完整 sample/insert/writer 往返，与 `LocalClient` 行为对齐

---

## ⑥ Crash recovery + cleanup

**What to build:** 可靠性。udsocket EOF 检测断连；client 崩溃时 server 遍历该 client
`outstanding_offsets_` 集中释放、`shm_unlink` 其 ring 段；其他 client 不受影响；
server 启动清旧 sock / 旧 SHM 段（R6/R7/R12）。新 client 可干净重连。

**Blocked by:** ⑤ Python API: `ShmClient` + `Server(shm=True)`

- [ ] server 监听 udsocket EOF/ECONNRESET → `HandleDisconnect`：`ReleaseAll`、`shm_unlink` 两条 ring、销毁 `ClientState`
- [ ] server `HandleClose`（client 主动 `CLOSE` 消息）走同一清理路径
- [ ] `ShmBootstrapServer::Create` 先 `unlink(socket_path)` 再 bind（R7）；`ShmBytePool::Create` 用 `O_EXCL`，失败 `shm_unlink` 旧名重试（R6）
- [ ] server 收 SIGTERM/SIGINT（经现有 `Server.stop()` 路径扩展）清理所有 SHM 段 + udsocket（R12）
- [ ] client 读 udsocket EOF → 抛 `ConnectionError`，在途请求全失败
- [ ] 崩溃恢复测试：kill client → server 无泄漏（outstanding 清空、ring 可 unlink、其他 client 不受影响）→ 新 client 重连正常

---

## ⑦ Performance baseline + v2 spike

**What to build:** 度量而非功能。gRPC loopback vs SHM 的 sample throughput/latency
基准；v2 优化（insert 字节复用为 sample 切片源，省"压缩进 ChunkStore 再解压"往返，
spec §6）的 spike 文档 + 是否值得做的决策。

**Blocked by:** ⑥ Crash recovery + cleanup

- [ ] 基准脚本：同表同数据，分别走 `Client('localhost:port')` / `ShmClient`，测 sample throughput（samples/s）+ p50/p99 latency
- [ ] 结果落 `docs/shm-benchmark.md`，确认 SHM 在 sample 路径（原设计核心痛点：N 次重复解压）有 measurable 提升
- [ ] v2 spike 文档：server 维持 `chunk_key → SHM 偏移` 索引、sample 直接基于 insert 原始字节切片的可行性 + 复杂度评估
- [ ] 决策：v2 值不值得做，写进 spike 文档结论
