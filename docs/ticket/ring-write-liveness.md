# Tickets: Ring 写入侧忙等无 liveness 检查（2026-07-31）

> 来源：`feat/numpy-embed` 并发专项 adversarial review，HIGH #2（1/2 票存活；弃票方认可代码
> 事实，仅质疑触发概率）。与 [dispatch-thread-hol.md](dispatch-thread-hol.md) 互为因果。

## Ring::Write / TryWrite 忙等无 liveness，服务端卡死时客户端永久挂起

**What to build**：给写入侧忙等加 EOF/liveness 检查与超时上限，镜像读侧已有的保护。

**现状**：

- ring.cc:202-204：`while (first_seq - tail > capacity - num_slots) sched_yield();`
  纯忙等，无控制通道探测、无 deadline、无取消路径。
- 全部写调用点（trajectory_writer.cc:1012 `ALLOCATE`、:1115 `INSERT`、:1128 `RELEASE`、
  以及 shm_client.cc 侧写路径）都没有 `control_fd`/EOF 检查。
- 读侧却有完整保护：shm_client.cc:66 `kReadBlockingHardCap = 60s`，
  `ReadBlocking(..., control_fd, ...)` 每轮探测 `IsPeerClosed`（:76）。
- 触发前提：服务端 dispatch 线程卡死 —— 现实存在，见
  `HandleCheckpoint` 同步执行无界磁盘 `checkpointer_->Save`（shm_server.cc:1010-1032）。

**后果**：dispatch 线程一旦卡住（磁盘慢/挂起），客户端写侧 `sched_yield()` 永久自旋
（100% CPU），`TrajectoryWriter::Close()`、GC 析构全部挂起，用户无感知无错误。

**修复选项**：

1. 写循环加 `control_fd` 探测 + 60s 硬上限（镜像读侧 `kReadBlockingHardCap`），超时返回
   `UnavailableError`/`AbortedError`，调用点按现有错误路径处理。
2. 忙等改阻塞：eventfd/pipe 唤醒（ring 头注释已列为 upgrade 方向），顺带消掉 CPU 自旋。
3. 服务端侧配合：[dispatch-thread-hol.md](dispatch-thread-hol.md) 把 `HandleCheckpoint`
   移出 dispatch 线程，从源头消除「卡死」触发条件。

**Blocked by**：None — 可与 #3 并行；#2 依赖 #3 的场景仅在复现测试中需要。

- [ ] `Ring::Write` 增加带 control_fd + deadline 的变体（或参数化），超时返回明确错误
- [ ] 三个写调用点接入 liveness 检查
- [ ] 回归测试：关闭服务端 fd 后写侧在 ≤60s 内返回错误而非永久自旋
