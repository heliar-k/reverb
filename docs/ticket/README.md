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

新增 ticket 时直接在本目录开新文件，结案后把一行总结追加到上表。
