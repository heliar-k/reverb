# Tickets: SHM 并发正确性（2026-07 分支扫描，2026-07 完成）

> 归档自原 `tickets.md` 活动板。本组 ticket 全部结案。

来源：对 `feat/numpy-embed` 全分支潜在 bug 扫描（三个并行审计代理 + 人工抽查验证）。
同批发现中已修复的不在此列：`fix(numpy)` bytes 序列化、`fix(shm)` pool offset 校验、
`fix(shm)` 池耗尽死锁（见对应 commit）。以下两张为协议级改动，单独推进。

---

## SHM 单连接多 sampler 竞态（扫描 #1）

**What to build**：修复同一 `ShmClient` 连接上并发多个 sampler 的 SPSC 破坏与响应错配。
现状：`ShmClient::NewSampler`（shm_client.cc:480）不限制 sampler 数量，每个 `ShmSampler`
的 worker 都在同一对 `sample_c2s`/`sample_s2c` ring 上生产/消费 —— 双生产者无 CAS 写
`head`（槽交错损坏）；且 `ShmError.request_seq`（shm_protocol.proto:161）从未赋值，
服务端异步完成 sample 乱序时 sampler A 可消费 B 的 `SAMPLE_RESP`，**静默返回错误表的
数据**。复现：`client.sample("a")` 与 `client.sample("b")` 并发（或同一表两个迭代器）。

分两期：
1. **近期（本 ticket 必须）**：`NewSampler` 在同一连接已有活 sampler 时返回
   `FailedPreconditionError`，把 SPSC 不变式从“隐式约定”变成强制约束；文档说明
   多表采样用多连接或串行迭代。
2. **远期（可选，另议）**：`request_seq` 贯穿 `ShmSampleRequest`/`SAMPLE_RESP`/`ShmError`，
   响应按 seq 路由到对应 sampler，恢复多 sampler 能力；`sample_c2s` 改 MPMC 或每
   sampler 独立 ring 对。

**Blocked by**：None — can start immediately。

- [x] `NewSampler` 拒绝同一连接的第二个活 sampler（`FailedPrecondition`），sampler 关闭后释放名额
- [x] 并发 `sample("a")`/`sample("b")` 回归测试：第二个 sampler 得到明确错误而非静默错数据（shm_sample_test `SecondConcurrentSamplerRejected`）
- [x] Python 层 `ShmClient.sample()` 并发调用行为文档化（client.py `sample()` docstring）
- [x] （远期）`request_seq` 路由方案评估并记录结论（做/不做）

> **远期 request_seq 评估结论（2026-07-25）：暂不做。** 恢复多 sampler 需
> `request_seq` 贯穿请求/响应 + 按 seq 路由 + `sample_c2s` 改 MPMC 或每 sampler
> 独立 ring 对 —— 协议面改动大，而单连接单 sampler + 多连接已覆盖并发采样需求。
> 若未来单连接多路采样成为真实瓶颈，按 ticket 中的远期方案重开。

> **实现说明**：permit 为 `ShmClient::sampler_active_` 原子布尔，NewSampler
> `exchange` 认领、`ShmSampler::Close` 释放（析构经 Close 幂等）。直接
> `ShmSampler::Create`（测试）不受限——约束在客户端 seam 而非 sampler 类。

---

## Ring 多槽消息发布竞态（扫描 #4）

**What to build**：修复 `Ring::WriteSlots`（ring.cc:250）多槽消息的发布时序。现状：
按槽序 0..n-1 逐个 release-store `seq`，消费者可 acquire 到槽 k 的新 `seq` 而槽 k+1
的 store 还在生产者 store buffer —— `Read` 返回 `InternalError("ring continuation
slot missing")`，而这是**正常竞态**非数据损坏。服务端下一圈自愈，但客户端
`ReadBlocking`（shm_client.cc:74）把非 NotFound 错误当致命错误传播，直接杀掉 writer
流/sampler。触发：任何 >240B 消息（INSERT body、多列 SAMPLE_RESP）在负载下。

方向（任选其一，倾向 a）：
a. **批量发布**：先写全部数据槽（relaxed），最后单次 release-store 首槽 `seq` 作为
   “整条消息就绪”信号；消费者缺续槽时视为 `NOT_READY` 下轮重试。
b. 消费者侧容忍：`Read` 遇缺续槽返回 NotFound(NOT_READY) 而非 InternalError —— 最小
   改动但留下“部分发布被消费者目击”的语义，需确认生产者不会长期停在半条消息。

**Blocked by**：None — can start immediately。与上一张 ticket 无依赖，可并行。

- [x] 多槽消息在并发读写下不再产生 "continuation slot missing" 致命错误
- [x] ring_test 新增多槽消息并发压力用例（大消息 + 高频读写，断言无 InternalError，`MultiSlotConcurrentReadNeverSeesPartialMessage`——修复前稳定复现 red）
- [x] 客户端长消息（大 INSERT / 多列 SAMPLE_RESP）在负载下不再被杀流，shm_insert_test/shm_sample_test 回归通过

> **方案评估结论（2026-07-25）：采纳 a 的精化版——槽 0 最后发布。** 续槽照旧
> 逐个 release，仅把槽 0 的 seq store 挪到循环末尾；消费者对槽 0 的 acquire 与
> 该 release 构成 synchronizes-with，整条消息（含续槽 seq）对其原子可见，
> 消费者代码零改动。方案 b（容忍 NOT_READY）会把“半条消息被目击”常态
> 化、稀释真损坏信号，否。崩溃语义反而变强：槽 0 未发布 = 整条未就绪，
> "continuation slot missing" 从此只指示真损坏。
