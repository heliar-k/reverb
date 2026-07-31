# Tickets: 表关闭时排队 insert 被 ACK 为成功（静默丢数据，2026-07-31）

> 来源：`feat/numpy-embed` 并发专项 adversarial review，MEDIUM #5（1/2 票存活；弃票方
> 认可机制属实，争议在严重度/来源 —— 上游继承、关闭时不可避免、或可视为已接受语义）。
> 全部评审一致：机制真实，客户端无法区分「已插入」与「被丢弃」。

## 关闭路径对 pending insert 回调无状态通道，客户端收到成功 ACK

**What to build**：让关闭时被丢弃的 insert 对客户端可见（状态通道或文档化语义）。

**现状**：

- 关闭路径：`Table::Close`（table.cc:422 附近）→ `NotifyPendingInserts`（table.cc:143）
  对每个 pending insert 调用 `(*to_notify)(r.item->key())`，回调**无 status 参数**。
- SHM 侧：回调被 `HandleInsert` 的 lambda 包装（shm_server.cc:1245），照样发
  `INSERT_ACK` —— 客户端以为已落表，实际数据在 ChunkStore/表内都不存在。
- gRPC 侧：`WorkerlessInsertReactor` 同样上报成功。
- 数据去向：ChunkStore 已写入的 chunk 在断开时由 `ReleaseAll` 回收，无泄漏，但
  insert 语义是「静默丢失报成功」。

**修复选项**：

1. **状态通道**：`InsertCallback` 增加 status 参数（或单独 `InsertDroppedCallback`），
   关闭时回调收到 `CancelledError`，SHM 侧据此发 `ERROR`（非 `INSERT_ACK`），gRPC 侧
   上报失败。客户端可重试/告警。
2. **文档化语义**（最小改动）：在 `LocalClient` / SHM 客户端 docstring 明确
   「server 关闭瞬间的 in-flight insert 可能被丢弃且无法区分」，接受现状。
3. 折中：仅对「用户主动 stop」发 ERROR，对进程崩溃路径（本就无法通知）维持现状。

**Blocked by**：None。

- [ ] 决定：状态通道（改 proto/回调签名）或文档化语义
- [ ] 若做状态通道：SHM + gRPC 两端测试「关闭时 pending insert → 客户端收到明确错误」
