# Tickets: ShmServer ClientState 生命周期（use-after-free，2026-07-31）

> 来源：`feat/numpy-embed` 并发专项 adversarial review（13 条 finding → 8 条存活，本文件为
> HIGH #1）。评审方法：1 个调查 agent 产出 finding，2 个质疑 agent 独立核查，≥1 票存活。
> 本 finding 2/2 票存活。

## ShmServer 销毁 ClientState 时表回调线程仍可能持裸指针执行回调

**What to build**：修复 `ClientState` 在回调仍可能执行时被销毁的 use-after-free。

**现状**：

- `HandleSample`（shm_server.cc:718）与 `HandleInsert`（shm_server.cc:1232）把裸指针
  `ClientState* state_ptr = &state` 捕获进 `SamplingCallback` / `InsertCallback`，
  回调在 **table 的 callback-executor 线程**触发（注释明确写「完成回调在 table worker
  线程触发」）。
- 回调体内解引用 `state_ptr` 操作 `pending_samples`（:727 附近 `absl::MutexLock lock(&state_ptr->pending_samples_mu)`）
  与 `insert_outbox`（:1245 `absl::MutexLock lock(&state_ptr->insert_outbox_mu)`）。
- 销毁路径无握手：
  - `ShmServer::Stop()`（shm_server.cc:155）`clients_.clear()` 直接销毁全部 `ClientState`；
  - `HandleDisconnect`（shm_server.cc:200）`clients_.erase(...)` 逐客户端销毁；
  - 两者都不等 `pending_insert_callbacks` / `pending_sample_callbacks` 清空（这两个 vector
    的存活由 table 持有 weak_ptr 的回调驱动，回调本身只被 ClientState 的 shared_ptr 攥住）。
- 表 worker 生命周期：`Stop()` 不碰表；worker 只在 `~Table` 才 join（table.cc:216-227，
  `Close()` 置 `stop_worker_` 只是异步信号）。且成员声明顺序 `tables_`(shm_server.h:262)
  在 `clients_`(:270) 之前 → 析构逆序先毁 `clients_`，worker 在 ClientState 存活期内
  一直能执行回调。

**复现路径**（竞态，非确定性）：`Server.stop()` / `HandleDisconnect` 时表 worker 已
`insert_completed.lock()` 拿到回调副本并执行到中途（碰 `state_ptr->insert_outbox_mu` /
`pending_samples_mu` 时），主线程同时销毁 `ClientState` 及其 mutex → 堆 use-after-free。
慢表 worker + stop-in-flight 下大概率命中，ASan/TSan 可稳定捕获。

**修复选项**（按推荐序）：

1. **先停表再清客户端**：给 `Table` 加 `Stop()`（`Close()` + join worker，~Table 已隐含此序），
   `ShmServer::Stop()` 在 `clients_.clear()` 前对所有表调 `Stop()` —— 之后 worker 不可能
   再执行回调，从根上消窗。这是最小且可确定验证的修复。
2. **引用计数**：回调 lambda 改捕获 `shared_ptr<ClientState>`（ClientState 由 clients_ 持有
   + 回调共享），最后一个回调返回才析构。**HandleDisconnect 路径（逐客户端、不能停表）
   只能靠这个**，故无论 1 做不做都建议做。
3. 组合（推荐）：1 关 Stop() 大窗 + 2 关 HandleDisconnect 小窗。

**Blocked by**：None — can start immediately。

- [x] `ClientState` 改 shared_ptr 管理；回调捕获 shared_ptr 而非裸指针
- [x] `Table::Stop()`（Close+join）供 `ShmServer::Stop()` 先停表；`Stop()` 先停表后清客户端
- [x] 回归测试：慢表 worker + in-flight insert/sample 时 `Server.stop()` 与 `HandleDisconnect`
      （断开单个客户端），ASan/TSan 无报错，重复多轮
- [x] worker 关闭路径（table.cc:422 `NotifyPendingInserts`）在测试中覆盖

## 修复记录（2026-07-31，已完成）

实现（与方案 3 一致）：`Table::Stop()` = `Close()` + join worker + drain callback executor
（`TaskExecutor::Close` 幂等，`~Table` 重入安全）；`clients_` 改
`vector<shared_ptr<ClientState>>`，`HandleSample`/`HandleInsert` 收 shared_ptr，两处回调按值
捕获；`ShmServer::Stop()` 在清 clients 前停所有表。

**复现教训**（对将来调时序敏感竞态有用）：

1. keepalive 设计使关闭路径**确定性安全**：ClientState 析构先杀回调对象，`weak_ptr.lock()`
   失败，回调体不执行。唯一窗口是 executor 已 lock 且回调体执行中途（亚微秒）。
2. **HandleDisconnect 路径在此 harness 结构性不可达**：dispatch 死检在 pass 顶部，erase 落在
   抽干 pass 之后（ASan 下实测 3-11ms），任何回调流都已结束。
3. **Stop 路径可达**：`Stop()` 的 clear() 时机由测试线程控制，无死检延迟。关键约束：worker
   必须比 dispatch 入队慢（`T > n_tables × d_enqueue`，ASan 下 d≈15µs/req）backlog 才能在
   clear() 时存活；用异步 `TableExtension` 的 `OnSample` 自旋 300µs 节流（
   `WaitForBackgroundWork` 在扩展队列 ≥10 时阻塞 worker）实现精确降速。
4. pool 256 块/tier 上限是隐形杀手：blast 超上限后 drain 逐条打日志，dispatch pass 膨胀到
   几十 ms，一切时序假设作废。每轮 completions 须 <256。
5. 最终配置：8 表 × 10 item、300µs 节流、+250µs 调 Stop()，ASan 预改 0.4s 内抓到
   heap-use-after-free（栈：`HandleSample` 的 sample 完成回调）。测试在
   `reverb/cc/shm/shm_crash_test.cc`（`StopDuringSampleCallbackStormDoesNotUaf`，sanitizer
   300 轮 / plain 5 轮）。
