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

新增 ticket 时直接在本目录开新文件，结案后把一行总结追加到上表。
