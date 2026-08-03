# Tickets 归档

已结案 ticket 组的索引。完整内容见同目录各文件。

## 归档索引

| 完成日期 | 主题 | 归档文件 | 做了什么 |
|----------|------|----------|----------|
| 2026-07 | 消除三客户端传输层代码冗余 | [client-transports-dedup.md](client-transports-dedup.md) | Python `_BaseClient` 合并 `trajectory_writer`/`structured_writer` 三份副本;C++ 提取 `SendInsertFlowRequest` 模板消掉 SHM 四方法 ~150 行复制;`MakeStructuredWriter` 自由函数替代三端 `NewStructuredWriter`(signature 填充判伪需求未提取);pybind 三处重复序列化/proto 转换/Options 构造各提取辅助函数 |
| 2026-07 | SHM 并发正确性 | [shm-concurrency-correctness.md](shm-concurrency-correctness.md) | `NewSampler` 拒绝同连接第二个活 sampler(强制 SPSC 不变式);`Ring::WriteSlots` 槽 0 最后发布,消掉多槽消息「continuation slot missing」致命错误 |
| 2026-07-25 | torch.Tensor 支持 Phase 1 | [torch-tensor-phase1.md](torch-tensor-phase1.md) | `torch_support.py` 写入路径归一化(三 writer + legacy);`output_format='torch'` 采样侧零拷贝;torch 可选依赖(`[torch]` extra);双轴 code-review 追加修复 |
| 2026-07-26 | numpy↔字节流零拷贝(Tier 1+2) | [numpy-zero-copy-tier1-2.md](numpy-zero-copy-tier1-2.md) | `TensorBuffer` 改 `shared_ptr` owner + view;采样侧 `ToNdArray` 零拷贝;写入侧 `FromNdArray(zero_copy=true)` opt-in(env 置 read-only);SHM 插入 `SerializeToArray` 直写 pool;torch 侧零拷贝自动成立 |
| 2026-07-26 | Tier 3 协议层自适应压缩 | [tier3-adaptive-compression.md](tier3-adaptive-compression.md) | `TensorProto.uncompressed` 标志;`WorthCompressing` 采样探测(≥16KB 取首/中/尾 3×4KB 估压缩率,≥0.9 判不可压)跳过 snappy;解压侧分流免 proto 拷贝;shm 768KB float insert +83%、sample +42% |
| 2026-07-31 | ShmServer ClientState 生命周期（UAF，并发评审 #1） | [shm-clientstate-lifetime.md](shm-clientstate-lifetime.md) | `Table::Stop()`=Close+join worker+drain callback executor;`clients_` 改 `shared_ptr<ClientState>`、insert/sample 回调捕获共享所有权;`ShmServer::Stop()` 先停表后清客户端;storm 回归测试 ASan 红绿验证（预改 0.4s 抓到 UAF）;`shm_sample_test` size→medium（预存在 Close-60s 竞态被 ASan 暴露） |
| 2026-07-31 | dispatch 线程 HOL + Ring 写侧 liveness（并发评审 #3+#2） | [dispatch-thread-hol.md](dispatch-thread-hol.md)、[ring-write-liveness.md](ring-write-liveness.md) | checkpoint `Save` 移出 dispatch（专用 `TaskExecutor` + outbox 回传，`Stop()` 先 drain）;新增 `WriteBlocking`（`TryWrite` 轮询+EOF 探测+60s 上限）接入全部 10 个客户端写调用点;红绿验证：闩锁 checkpointer 断言次客户端 5s 内有响应（预改超时、修复 8ms） |
| 2026-07-31 | 关闭时 insert 假 ACK 语义（并发评审 #5，文档化结案） | [table-close-insert-ack.md](table-close-insert-ack.md) | 拍板文档化不改协议;语义写入 [`concepts.md`](../guide/concepts.md) §3.5「写入耐久性」:#1 修复后 SHM 假成功窗口实质关闭，唯一假成功场景是 gRPC server 主动 Close/重启 |
| 2026-07-31 | DeleteItem 错误路径部分修改（并发评审 #6） | [table-deleteitem-partial-mutation.md](table-deleteitem-partial-mutation.md) | `DeleteItem` 改两阶段（先全量校验 `episode_id` 再递减），失败不再留下部分修改的 `episode_refs_`;`TableTestPeer` 注入不一致状态做红绿验证（公开 API 不可达） |
| 2026-07-31 | Ring Open/Read 防御性校验缺口（并发评审 #7） | [ring-open-defensive-gaps.md](ring-open-defensive-gaps.md) | Open 补 version/几何校验（`RingHeader.version` 早已存在，只补校验）;Read 校验 `body_len`（旧代码红测直接 SIGSEGV）;EEXIST 根治 = 段名折入 server epoch（socket+pid+墙上时钟纳秒），同 socket 重启不再 unlink 活段，协议无破坏（段名本就走 Welcome 下发） |
| 2026-07-31 | SHM 全局 signal-stop 波及进程内所有实例（并发审计 finding 4，文档化结案） | [shm-global-signal-stop.md](shm-global-signal-stop.md) | 拍板不改代码：`g_signal_stop` 文件级原子 + `call_once` 的上限与 self-pipe 升级路径早已写在代码注释；多 server 同进程仅测试使用且无实际危害；重开条件 = 多实例同进程成为受支持用法。附 finding 5（回调 `shared_ptr<ClientState>` 捕获）良性评估，亦不动代码 |
| 2026-08-01 | `TrajectoryWriter::RunShmWorker` 竞态 SIGSEGV（压测发现） | [shm-writer-worker-race-sigsegv.md](shm-writer-worker-race-sigsegv.md) | 构造顺序竞态：ctor 成员初始化列表启动 worker 线程，但 `stream_worker_` 声明在 `stream_ok_`/`stream_status_` 之前，竞争下 worker 读未构造成员、拷贝未构造 Status 解引用野指针；修复 = `stream_worker_` 声明挪到最后（三构造器同受益）；`WriterCtorDoesNotRaceMemberInit` 回归；压测修复前 8/300 → 修复后 0/300 |
| 2026-08-02 | SHM 客户端 close-while-in-flight 挂起（压测 hunt 发现） | [shm-close-while-in-flight.md](shm-close-while-in-flight.md) | `ReadBlocking`/writer 读循环的存活探测只看 `control_fd>=0`，fd=-1/被复用时无限自旋（挂起进程忙等 14h，加压 ~47/50 命中）；修复 = `ShmConnection` 加 `closed` 原子标志 + 幂等 `Close()`，两处读循环检查置位即 `UnavailableError`；800 次加压迭代绿 + 全量 65/65；段错误模式结案为前序 WriterCtor 竞态 |
| 2026-08-03 | SHM 批量 INSERT（client-benchmark §3 优化项） | [shm-batch-insert.md](shm-batch-insert.md) | `RunShmWorker` 凑批连续 ready item（≤64 个/≤128KB）进单条 `ShmInsertRequest`：流水化 ALLOCATE burst（N 发 N 收按 FIFO 配对，免改 proto）+ 聚合 ACK，一批 ~2 次 ring 往返（旧每 item 2 次）；服务端 `AckAggregate` mutex 修跨表回调并发竞态、`EnqueueInsertS2C` FIFO 护栏保 RESP 定序；b64 insert small 7.1×/med 3.6× 超 gRPC b64 |
| 2026-08-03 | SHM pool slab 几何可配（client-benchmark §5.3 优化项） | [shm-pool-slab-config.md](shm-pool-slab-config.md) | `Server(shm_pool_slab_sizes=..., shm_pool_blocks_per_slab=...)` 四跳透传（Python→pybind 双 lambda→`ShmServer::Create`→`ShmBytePool::Create`），默认路径零变化；`Create` 补几何校验（严格升序、每档 ≥8B 防 free-list 指针写穿）；超档 insert/sample 客户端收明确错误（InvalidArgument 永久/档耗尽 ResourceExhausted 瞬时分开）；小档配置把单连接高水位从 ~1.4GB 降到档位容量量级 |
| 2026-08-03 | SHM dispatch 事件驱动唤醒（client-benchmark §4 优化项） | [shm-dispatch-eventfd-wakeup.md](shm-dispatch-eventfd-wakeup.md) | dispatch 静止时阻塞 poll（listen_fd + 各 client control_fd + server-local eventfd，50ms 兜底）；客户端 `WriteBlocking` 仅在 ring 共享 `server_asleep` 标志置位时发 1 字节唤醒（热路径零新增 syscall）；三处 off-dispatch 生产者（insert ACK 聚合/sample 完成/checkpoint executor）enqueue 后写 eventfd；seq_cst 双定序防 missed-wakeup；顺带删掉有工作轮次的 50us 地板——sample 吞吐 +54~88%、单 client p50 近减半、w8r8 服务端 CPU 降 ~100-130pp |

新增 ticket 时直接在本目录开新文件，结案后把一行总结追加到上表。

## 编号图例（代码 / design / spec / benchmark 中的 ticket 引用）

代码注释与长期文档里的 ticket 编号有两套来源，均**不指向**本索引表（索引表用描述性
文件名），需要时按下表对照：

**①-⑬ —— 开发期 tracer-bullet / 后续 tickets**：①-⑦ 定义在已删除的
`tickets.md`（git 历史 `544c8c9^:tickets.md`）；⑧-⑬ 定义散见于代码注释
（`shm_server.h`/`shm_client.h`/`ring.h`/`trajectory_writer.cc`）与
[numpy-shm-design.md](../design/numpy-shm-design.md) §8.3 的消息类型表。
本表是两者合一的速查：

| 编号 | 主题 | 落地位置 |
| --- | --- | --- |
| ① | Ring + Bootstrap echo | `ring` / `bootstrap` + echo 测试 |
| ② | Byte pool round-trip（slab + refcount，决策 C4 确立） | `byte_pool` + `ALLOCATE`/`ALLOCATE_RESP` |
| ③ | Sample path（server→client） | `ShmServer` / `ShmSampler` 骨架 |
| ④ | Insert path（client→server，`RunShmWorker`） | `TrajectoryWriter::RunShmWorker` + insert 回调保活 |
| ⑤ | Python API：`ShmClient` + `Server(shm=True)` | `reverb/client.py`、`reverb/server.py` |
| ⑥ | Crash recovery + cleanup（断连检测、集中释放） | `HandleDisconnect` / `CleanupClient` / `IsPeerClosed` |
| ⑦ | Performance baseline + v2 spike | → [client-benchmark.md](../benchmark/client-benchmark.md) |
| ⑧ | `server_info`：step 1 bootstrap 快照 → step 2 按需 `SERVER_INFO` ring 往返；⑧-2b = `NewTrajectoryWriter` 实时签名校验 | `ShmClient::ServerInfo` / `HandleServerInfo` |
| ⑨ | 多表路由（按表名） | `ShmServer::FindTable` |
| ⑩ | 控制面 `mutate_priorities`/`reset`（走 insert 流 + `insert_flow_mu`）+ 死锁修复（方向 A 异步 sample / 方向 C 60s 硬上限） | `HandleMutatePriorities` / `HandleReset` / `EnqueSampleRequest` |
| ⑪ | `checkpoint`（走 insert 流 + 注入 checkpointer，Save 在专用 executor） | `HandleCheckpoint` / `checkpoint_executor_` |
| ⑫ | `ShmClient` pickle（存 `socket_path` 重连） | `ShmClient.__reduce__` |
| ⑬ | legacy `Writer`/`insert` won't fix（无 SHM seam） | Python 层 `NotImplementedError` |

**01/02/03 —— 近期性能/调优 tickets**（benchmark 与代码注释引用）：

| 编号 | 归档文件 | 主题 |
| --- | --- | --- |
| 01 | [shm-batch-insert.md](shm-batch-insert.md) | 批量 INSERT + 聚合 ACK |
| 02 | [shm-pool-slab-config.md](shm-pool-slab-config.md) | pool slab 几何可配 |
| 03 | [shm-dispatch-eventfd-wakeup.md](shm-dispatch-eventfd-wakeup.md) | dispatch 事件驱动唤醒 |

其余归档文件互相引用用评审编号（#1-#7、扫描 #1 等），见上表"做了什么"列。
