# 01 — SHM 批量 INSERT：一次 ring 往返携带 N 个 item

> 状态：**已完成**（2026-08-03；全量 65/65 两轮绿，新增
> `BatchedFlushCostsOneInsertRoundTrip` 回归；复测 b64 insert
> small 2887→20545 ips（7.1×）、med 2391→8593（3.6×），超 gRPC b64，
> b1 不变。服务端顺带修两个批量才暴露的问题：`AckAggregate` mutex
> 跨表回调竞态、`EnqueueInsertS2C` FIFO 护栏）

**What to build:** `RunShmWorker` 目前每个 item 做一次 INSERT→ACK 往返
（`trajectory_writer.cc`），SHM 写入吞吐被钉死在 per-item 往返延迟
~0.4–0.6ms/item，flush 批量 b64 几乎无效（benchmark §3：b64 仅
~2.4–2.9k ips）。本 ticket 让 worker 每次从 `write_queue_` 收集至多 K 个
已 ready 的 item，合并成**一条** `ShmInsertRequest` 发出，一次 ACK 确认
整批、统一 `RELEASE` 全部 chunk 偏移、一次性从 `in_flight_items_` 抹去
N 个。目标：b64 SHM insert 对齐 gRPC b64 量级（~6–17k ips，≥2× 现状）。

协议与服务端**已就绪**，本 ticket 主要是客户端 worker 改动：

- `ShmInsertRequest` proto 本就是 `repeated ShmChunkRef chunks` +
  `repeated PrioritizedItem items`（`shm_protocol.proto:71`）。
- 服务端 `HandleInsert` 已循环 `req.chunks()` 并按 chunk_key 去重
  （`shm_server.cc:1108`，注释明确「Multiple items may reference the same
  chunk_key」），多 item 请求天然可处理。需顺带验证：批量请求中部分
  item 失败时 ACK/ERROR 语义是否完整覆盖整批（offsets_to_release 必须
  覆盖批内全部 chunk，否则 client 侧偏移泄漏）。

批量边界设计约束：

- 批内 item 必须全部 `AllReady(refs)`（沿用现有逐 item 检查，凑不满就
  发当前已凑到的，不为凑批引入等待）。
- K 上限与单条 ring 消息容量/最大 slab 档挂钩（防超大消息），flush 场景
  自然 K ≤ flush batch。
- `Flush()`/`EndEpisode` 语义不变：批量 ACK 到达前不得返回成功；ACK
  超时（60s 上限）按批处理，整批计入 `unrecoverable_status_`。
- backpressure 记账（`num_items_in_flight`）按批内 item 数增减。

**Blocked by:** None — can start immediately

**Status:** done（2026-08-03）

- [ ] `RunShmWorker` 单次 INSERT 携带 1..K 个 ready item，一次 ACK 确认整批
- [ ] 批量 ACK 的 `offsets_to_release` 覆盖批内全部 chunk 偏移并统一 RELEASE
- [ ] `Flush()`/`EndEpisode`/超时/错误路径语义与逐 item 版一致
- [ ] 新增回归测试：一次多 item flush 只产生一次 ring 往返（可断言请求数）
- [ ] 既有测试全绿（`shm_insert_test`、writer 相关）
- [ ] 复测 benchmark §3 insert 矩阵并更新 `docs/benchmark/client-benchmark.md`
      数字：b64 SHM insert ≥ 2× 现状（~2.4–2.9k → 目标 ≥6k ips）
