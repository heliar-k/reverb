# Ticket: SHM 客户端 `TrajectoryWriter::RunShmWorker` 竞态 SIGSEGV（压测发现）

> 状态：**已结案**（2026-08-01 修复并验证）。
> 来源：2026-08-01 并发审计 P0/P1/P2 修复后的全量验证压测中发现；曾用 HEAD
> 基线对照证实为存量 bug（与当日审计修复无关）。

## 现象

`shm_crash_test` 在高并行压测下以 ~3%/轮 SIGSEGV（exit 139）；同一签名也在
Python `shm_test` 的 `ShmConcurrentWriterSamplerTest`（writer 高频建拆 + 并发
sampler）出现。常规 `bazel test //reverb/...` 单次运行因此偶尔变红。

## 根因（已确认）：构造顺序竞态

`TrajectoryWriter` 三个构造器都在**成员初始化列表**里启动 worker 线程
（`stream_worker_(internal::StartThread(..., [this]{ RunShmWorker(); }))`），而
成员按**声明顺序**初始化 —— `stream_worker_` 声明在 `response_`/
`write_inflight_`/`stream_done_`/`stream_ok_`/`stream_status_` **之前**。
于是新 worker 线程启动时这些成员尚未构造。CPU 竞争下（10 路并行压测）新线程可
能被立即调度、而构造线程在两个成员初始化之间被抢占，窗口从纳秒拉到毫秒：

- worker 读未初始化的 `stream_ok_`；若为假（50% 随机）→ 跳过 `data_cv_` 等待；
- `if (!stream_ok_) return stream_status_;` 拷贝**未构造**的 `stream_status_`；
- 垃圾指针低位为 0（50%）→ 当作 `absl::StatusRep*` 做 `Ref()`（`lock addl
  $0x1,(%rax)`）→ 地址不可映射 → SIGSEGV。

`ShmSampler` 无此问题：worker 在 `Create()` 里、对象构造完成后才启动（两阶段）。

### 取证链（core + 反汇编 + 布局分析）

- 崩溃线程 = `TrajectoryWriter::RunShmWorker`；故障指令 `lock addl $0x1,(%rax)`，
  `rax=0x2e0`（非法 StatusRep 指针），来自 `this+0x2d0`（`mov 0x2d0(%r15),%rax`，
  r15 对象 vtable = TrajectoryWriter）。
- core 快照时 `this+0x2d0` 已变为合法 inline Status 值（1）—— 即构造器随后完成
  了该成员的初始化，与「worker 在构造中途读到未初始化内存」完全吻合。
- 崩溃时点：测试轮次中途（主线程在 `SleepFor`），服务端 dispatch 存活；writer
  对象本身活着 —— 排除 UAF/僵尸线程（`StartThread` 析构即 join，`Close()` 所有
  路径都 join，无漏洞）。

## 修复

`reverb/cc/trajectory_writer.h`：把 `stream_worker_` 声明挪到**最后**（
`stream_status_` 之后），保证 worker 线程启动时所有成员已构造完毕。三个构造器
（gRPC/local/SHM）同时受益；析构改为逆序后 worker 也先于其使用的成员被 join，
顺带消除一个潜在的析构顺序隐患。`write_inflight_`/`stream_done_` 仅 gRPC 路径
使用且 worker 自写自读，不受影响。

回归测试：`ShmCrashTest.WriterCtorDoesNotRaceMemberInit`（4 线程 × 5s 高频建拆
writer；修复前该强度下秒级崩溃，整二进制死亡即回归信号）。

## 验证

- 修复前崩溃率：HEAD 8/300、改动树 9/~200（同签名）；修复后：**0/300**（同一
  10 worker × 30 轮压测脚本）。
- `shm_crash_test`（含新回归）、`trajectory_writer_test`、
  `streaming_trajectory_writer_test`、`structured_writer_test`：全过。

## 复现/压测脚本（留存）

```bash
ulimit -c unlimited
for i in $(seq 1 10); do ( for j in $(seq 1 30); do
  bazel-bin/reverb/cc/shm/shm_crash_test >/dev/null 2>&1 || \
    echo "EXIT=$? worker=$i iter=$j" >> crashes.txt
done ) & done; wait
```
