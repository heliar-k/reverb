# 03 — SHM dispatch 线程事件驱动唤醒（eventfd 替代 50us 轮询睡眠）

**What to build:** `ShmServer::DispatchLoop` 每轮扫完全部 client 后**无条件**
`sched_yield() + usleep(50)`（`shm_server.cc:277-278`）。这带来两个问题：

1. **空闲/轻载 CPU**：无流量时 dispatch 仍以 20k 次/s 空转扫 ring，
   benchmark §4 资源列显示 SHM w8r8 服务端 ~5.5 核（vs gRPC ~2.2 核）。
2. **负载下延迟节流**：有流量的每一轮也照睡 50us，给每次 dispatch 周期
   加了固定延迟地板。

代码内 ponytail 注释已指明升级路径：「A blocking poll on all ring fds
would be cheaper still but needs eventfd plumbing per ring」。本 ticket
实现它：

- 每个 client 连接（或每条 c2s ring）配一个 eventfd；server 在一轮
  **无任何工作**时进入 `poll()` 阻塞（eventfd 集合 + listen fd，accept
  纳入同一 poll，保持 TryAccept 现有语义）。
- 客户端写路径只在 server 已置「asleep」共享标志时才 `eventfd` signal
  （空→非空转换才付一次 syscall，热路径零额外开销）。
- 一轮有工作时**不再睡眠**，直接下一轮（顺带消除问题 2）。
- 唤醒后清标志、重新扫全部 client；停止路径（`Stop()`/信号）保证能
  打断 poll。

**Blocked by:** 02 — SHM pool slab 可配置（用户要求串行执行，无技术依赖）

**Status:** ready-for-agent

- [ ] 空闲 server 的 dispatch CPU 降到 ~0（阻塞在 poll，不再 50us 空转）
- [ ] 客户端热路径（server 未睡时）零新增 syscall；仅 asleep 标志置位时
      signal 一次
- [ ] 有工作的轮次不再执行 usleep(50)
- [ ] 新增测试：dispatch 睡眠中，客户端写 insert/sample 请求可在 ms 级
      唤醒并得到响应；`Stop()` 可从 poll 中干净退出
- [ ] 复测 benchmark §2 sample + §4 pipeline：吞吐/p50 相对现状无回归
      （±10% 以内），更新 `docs/benchmark/client-benchmark.md` 资源列
- [ ] 既有测试全绿（含 shm_crash_test / close-while-in-flight 回归）
