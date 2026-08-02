# Ticket: SHM 客户端 close-while-in-flight 挂起（closed 标志修复）

> 状态：**已修复**（2026-08-02；全量测试 65/65 绿，800 次加压迭代绿，
> benchmark w8r8 复跑存活；段错误模式结案为前序 WriterCtor 竞态，见 §5）
> 发现日期：2026-08-02
> 关联：ticket ⑥（crash recovery）、ticket ⑩（死锁修复方向 C）

## 1. 现象

多客户端 SHM 并发压测（`/tmp/crashhunt` 遗留 harness + benchmark w8r8 场景）出现两类失败：

- **挂起**：`shm_crash_test` 进程卡死在
  `DisconnectWithInFlightCallbacksDoesNotUaf`，忙等自旋（~10% CPU/进程），
  永不退出。加压复现 47 次命中全部为该模式，iteration 集中于 ~84
  （fd 复用密度阈值），少量散布 28–43。
- **段错误（EXIT=139）**：crashhunt 记录 9 条，**至今未捕获栈**，疑似另一
  根因（见 §5）。

## 2. 根因（挂起模式，已确认，有 core dump 栈）

三证合一（`hang_core_1`，ABRT core）：

- 主线程 `pthread_join` 等 writer/sampler 线程；
- `ShmSampler::RunWorker` 在 `ReadBlocking` 的 `sched_yield` 自旋；
- `TrajectoryWriter::RunShmWorker` 在本地 `read_blocking` lambda 自旋；
- server `DispatchLoop` 健康空转。

机制：**`ReadBlocking` 的死服探测只看 `control_fd >= 0` 时的 EOF/HUP，
fd 为 -1（或 fd 号被复用）时探测被跳过**，于是永远等不到响应的读循环
无限自旋。测试用 `close(control_fd); control_fd = -1` 模拟崩溃，恰好制造
该状态；真实场景对应 close-while-in-flight（一个线程关闭连接、另一个
线程还在采样/写入）。fd 号复用时更隐蔽：`IsPeerClosed` 轮询到的是另一个
健康 fd，返回"存活"（POLLNVAL 语义救不了复用）。

注意重试语义放大：`ShmSampler::RunWorker` 对 60s 硬帽
（`kReadBlockingHardCap`）超时的处理是"无 waiter 则回卷重试"，探测失效时
每次重试都是同样的死循环。

## 3. 修复

`ShmConnection` 增加与 fd 无关的关闭信号：

- `std::atomic<bool> closed{false}` + 幂等 `Close()`（先 release 置标志，
  再关 `control_fd` 置 -1）；析构走 `Close()`；move 语义拷贝标志。
- 两处客户端读循环每轮检查标志（`shm_client.cc::ReadBlocking`、
  `trajectory_writer.cc::read_blocking` lambda），置位即返回
  `UnavailableError`，不再自旋。
- 测试 3 处 `close(conn->control_fd); conn->control_fd = -1;` 改为
  `conn->Close()`——既模拟崩溃（server 侧照样看到 EOF → 回收），又覆盖
  真实的 close-while-in-flight 路径。

改动文件：`reverb/cc/shm/shm_connection.{h,cc}`、
`reverb/cc/shm/shm_client.cc`、`reverb/cc/trajectory_writer.cc`、
`reverb/cc/shm/shm_crash_test.cc`。

## 4. 验证

- [x] 定向回归：4 worker × 2 × 100 次加压迭代全绿（修复前同配置 ~47/50 命中）
- [x] `bazel test //reverb/cc/shm/...` 全量回归（9/9 通过，含 shm_crash_test）
- [x] benchmark w8r8 场景复跑不再无声死亡（v4 core 批次完整跑完 w8r8，CORE_EXIT=0）

## 5. 段错误模式：已结案（前序 ticket 的 WriterCtor 竞态）

crashhunt 的 9 条 EXIT=139（最后记录 08-01 12:46 Asia）早于修复提交
`fix(reverb): TrajectoryWriter 构造顺序竞态致 RunShmWorker SIGSEGV`
（08-01 15:02 UTC，见 [shm-writer-worker-race-sigsegv.md](shm-writer-worker-race-sigsegv.md)）。
该 harness 是此修复的压测遗留（修完后忘了收尸），段错误模式**不是新 bug**。
佐证：本 ticket 修复后 300 + 800 次加压迭代零 139。

遗留硬化项（未观测到故障，不另开 ticket）：`ShmSampler`/`TrajectoryWriter`
持有 `ShmConnection* borrowed`（`shm_client.h:92`），client 先于
writer/sampler 析构理论上可悬空 UAF；本 ticket 的 flag 不覆盖该路径。
若未来出现对应故障，做所有权改造（`shared_ptr<ShmConnection>` 或 client
集中管理生命周期）。

## 6. 顺手记录的债

- **压测环境是 k8s pod（cgroup v1），内存上限 16 GiB**——不是节点内存
  （251GB）。`free`/`/proc/meminfo` 在容器里显示节点值，会误导。本轮
  benchmark 两次 SIGKILL（退出码 137）+ gRPC 子进程假死均系 cgroup
  oom_kill（取证 2026-08-02：`memory.failcnt=23841`、`oom_kill=97`、
  `max_usage` 顶穿 16GiB）。SHM 段是 tmpfs，计入 cgroup cache 配额。
  压测前应按 pod 配额设计负载规模。
- `ReadBlocking`（shm_client.cc）与 `read_blocking` lambda
  （trajectory_writer.cc）是同一段阻塞读逻辑的两份拷贝，本次修复不得不
  两边同步改。应提取共享实现（`WriteBlocking` 已共享，读侧也该如此）。
- `WriteBlocking` 的探测同样只看 `control_fd >= 0`：ring 满 + 连接已关
  时理论上同款自旋（本次未观测到，未改）。
- 压测 harness 教训：对被测进程用 `timeout`（默认 SIGTERM）会被
  `InstallSignalHandlers` 的优雅停机吞掉，制造"超时但进程不死"的假象；
  抓死锁现场须用 `timeout -s ABRT`。
