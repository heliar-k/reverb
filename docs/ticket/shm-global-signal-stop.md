# Ticket: SHM 全局 signal-stop 波及进程内所有 ShmServer 实例（并发审计 finding 4）

> 状态：**文档化结案**（不改代码）。记录结论与重开条件。
> 来源：2026-07-31 SHM 并发审计（5 findings 交叉验证，0 误报）；P0/P1/P2 已修复，本条为 finding 4。

## 现象

`g_signal_stop`（shm_server.cc:58）是文件级 `std::atomic<bool>`，SIGTERM/SIGINT 经
`SignalHandler` 置位，且 `InstallSignalHandlers` 用 `std::call_once` 只装一次。后果：

1. 任一实例的 dispatch 循环看到 `g_signal_stop` 都会退出 —— 信号会停掉**进程内所有**
   `ShmServer` 实例，无法按实例独立关停；
2. 被信号停掉而未走 `Stop()` 的实例，其 SHM 段（pool/rings）直到进程退出才被回收
   （泄漏窗口 = 进程剩余寿命）。

## 结论：不改代码

- 代码注释早已声明该上限与升级路径（"Ceiling: only one ShmServer per process is
  signal-driven; if multiple need independent shutdown, switch to a per-instance
  self-pipe. Upgrade path noted, not built."）；
- 多 server 同进程目前只有测试这么用（`ShmTwoServersOneProcessTest`，且其 `stop()`
  走 `running_` 实例标志而非信号，不受影响）；
- 信号路径的真实语义就是"进程要退出"，段泄漏随进程退出自然清零，无实际危害。

## 重开条件

多 server 同进程成为受支持的使用方式（而非测试特例）时，按注释里的 self-pipe 方案
（每实例独立管道替代全局原子）重做信号驱动关停。

## 附：finding 5 评估（同批，亦不动代码）

`HandleSample`/`HandleInsert` 回调捕获 `shared_ptr<ClientState>`，使 ClientState 在
`HandleDisconnect` 后存活到 table worker 完成。结论：良性 —— 无 UAF，且有界
（每 client sampler 队列 ≤8、insert in_flight ≤1）。不开独立 ticket。
