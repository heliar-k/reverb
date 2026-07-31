# Tickets: SHM 单 dispatch 线程 head-of-line blocking（2026-07-31）

> 来源：`feat/numpy-embed` 并发专项 adversarial review，MEDIUM #3（2/2 票存活）。

## 单 dispatch 线程串行服务所有客户端，一个慢操作卡全部

**What to build**：消除 dispatch 线程上的无界同步操作，至少把 checkpoint 移出该线程。

**现状**：

- 全服务恰好一个 dispatch 线程（shm_server.cc:140 `dispatch_thread_`）。
- `DispatchLoop`（shm_server.cc:220-249）逐客户端串行执行：`RecvHello`（250ms 有界）、
  **`HandleCheckpoint`（无界同步磁盘 `Save`，shm_server.cc:1032）**、表互斥操作、
  两个 ring 的 drain、outbox flush。
- 任一客户端任一操作慢（尤其 checkpoint 写盘）→ 之后所有客户端的 insert/sample 响应
  全部延迟，直到该操作完成。直接喂养 [ring-write-liveness.md](ring-write-liveness.md)
  的「客户端写侧永久自旋」：dispatch 卡住 = 全体客户端挂起。

**修复选项**：

1. **checkpoint 移出 dispatch 线程**（最小改动）：`HandleCheckpoint` 入队到独立线程/
   table worker 队列，完成时把响应放回该客户端的 outbox（复用现有 callback→drain 模式）。
   磁盘写从「阻塞所有人」降为「阻塞自己这条响应」。
2. 给 `RecvHello` 之外的操作加超时/预算（治标）。
3. 每客户端独立 dispatch 线程（大改，协议/生命周期面广，不建议近期做）。

**Blocked by**：None — 建议与 ring-write-liveness 一起排期，两者构成同一故障模式
（卡死服务端 → 客户端永久挂起）的两端。

- [ ] `HandleCheckpoint` 不再在 dispatch 线程同步执行 `Save`
- [ ] 慢 checkpoint 期间其他客户端的 insert/sample 延迟有界（回归测试：注入慢
      checkpointer，断言另一客户端 RTT 不受影响）
- [ ] 文档记录 dispatch 线程职责边界（什么可以阻塞、什么必须异步）
