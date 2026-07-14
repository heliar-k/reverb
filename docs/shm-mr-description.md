# MR: SHM 传输层 — POSIX 共享内存作为 gRPC/in_process 之外的第三条路径

## 概述

为 Reverb 添加基于 POSIX 共享内存的传输层,作为现有 gRPC 和 in_process 之外的第三条 client↔server 通道。同机分进程下,SHM 跳过全部序列化、压缩和 socket 往返——**sample 路径比 gRPC loopback 快 ~9-11×,达到无序列化的 in-process 天花板**。

设计源自 `docs/numpy-shm-design.md`,开发规范见 `docs/numpy-shm-spec.md`(含审查问题 R1-R13、决策 C1-C5),按 7 个 tracer-bullet ticket 顺序交付(`tickets.md`),每张票过 implement(TDD)→ 两轴 code-review(Standards + Spec)→ 按需 fix 闭环。

## 架构

```
┌─ Python ─────────────────────────────────────────────────┐
│  Server(tables, in_process=True, shm=True) ──┐           │
│    └ ShmServer (C++): dispatch 线程 + pool   │           │
│  ShmClient(socket_path) ─────────────────────┘           │
│    └ TrajectoryWriter / Sampler / StructuredWriter       │
└───────────────────────────────────────────────────────────┘
        │  C→S ring (SPSC) x2     S→C ring (SPSC) x2
        │  (insert 流 / sample 流)  (insert 流 / sample 流)
        │  (client 写请求)         (server 写响应)
        ▼                          ▼
┌─ C++ reverb/cc/shm/ ─────────────────────────────────────┐
│  Ring (SPSC, 非阻塞 Read / TryWrite)                     │
│  ShmBytePool (slab 分配器, server 独占, refcount)        │
│  Bootstrap (udsocket, A3 PID 命名, R7 stale unlink)     │
│  ShmServer: HandleSample/Insert/Release/Allocate/Disconnect │
│  ShmClient: Connect + ShmSampler(复用 worker 架构)       │
│  TrajectoryWriter::RunShmWorker (第三模式, 复用 RunLocalWorker 骨架) │
└───────────────────────────────────────────────────────────┘
```

**关键设计决策:**

- **C1** — ShmServer 生命周期挂在 `Server` Python 对象(`shm=True` 参数),不独立成对象
- **C2** — insert 的 SHM 偏移在收到 `INSERT_ACK.offsets_to_release` 前不得重用(writer backpressure)
- **C3** — sample 的 SHM 偏移引用计数起点=1,client RELEASE 后归零回收
- **C4** — pool 读写权限:client RW mmap,但分配权独占在 server(无跨进程锁);client 经 ALLOCATE 请求向 server 申请偏移
- **C5** — SHM Sampler 复用现有 worker 线程 + samples_ 队列架构(非调用者线程内联)
- **D** — per-flow 独立 ring(insert + sample 各一对 SPSC)。原 v1 每客户端一对 c2s/s2c ring,但 `TrajectoryWriter::RunShmWorker` 与 `ShmSampler` 都起后台 worker 线程,两者同时往同一 c2s ring 写(两个 producer)会损坏无 CAS 的 SPSC `head`。D 给 insert 流和 sample 流各开一对 ring,恢复 SPSC 单 producer 契约,writer/sampler 线程真正并发无锁。`ShmConnection` 持 4 条 ring;`WelcomeResponse` 带 4 个 ring 名;`MakeShmNames` 产 4 个名;`ShmServer` 建 4 条 ring + per-flow outbox;client 侧 sampler 用 sample_*、writer 用 insert_*。

## 交付内容(7 tickets)

| # | Ticket | 核心文件 |
| --- | -------- | --------- |
| ① | Ring + Bootstrap | `ring.{h,cc}`、`bootstrap.{h,cc}`、`shm_protocol.proto`、`shm_connection.h` |
| ② | Byte pool | `byte_pool.{h,cc}`(slab 分配器 + refcount + pool-full 阻塞) |
| ③ | Sample path | `shm_server.{h,cc}`(dispatch + HandleSample)、`shm_client.{h,cc}`(ShmSampler) |
| ④ | Insert path | `HandleInsert`/`HandleAllocate`、`TrajectoryWriter::RunShmWorker` |
| ⑤ | Python API | `pybind.cc`(PyShmClient/ShmSampler/ShmServer)、`client.py::ShmClient`、`server.py` `shm=` 参数 |
| ⑥ | Crash recovery | `Ring::TryWrite`(修 ③ outbox 债)、`HandleDisconnect`、fd-liveness EOF、client-side `ConnectionError` |
| ⑦ | Benchmark + v2 | `benchmarks/shm_benchmark.py`、`docs/shm-benchmark.md`(v2 DEFER 决策) |

## 性能基准(`docs/shm-benchmark.md`)

| transport | table size | throughput (sps) | p50 (µs) | p99 (µs) | vs gRPC |
| ----------- | ----------- | ------------------ | ---------- | ---------- | --------- |
| gRPC-loopback | 1000 | 806 | 1218 | 1671 | 1.0× |
| **SHM** | 1000 | **9100** | **104** | **163** | **11.3×** |
| in-process | 1000 | 8301 | 112 | 177 | 10.3× |
| gRPC-loopback | 10000 | 934 | 1061 | 1265 | 1.0× |
| **SHM** | 10000 | **8776** | **107** | **178** | **9.4×** |

- SHM 比 gRPC loopback 快 **~9-11×**(吞吐量 + 延迟)
- SHM 与无序列化的 in-process 路径**持平**——已触及理论上限
- gRPC 用 subprocess 强制走真 `SampleStream` 路径(非同进程捷径),数据真实

**v2 决策:DEFER** — v1 已达无序列化天花板,v2(insert 字节复用为 sample 切片源,省解压往返)的边际收益受限于 `UnpackChunkColumnAndSlice` 解压开销,对当前 workload 不显著,+200-400 LOC 复杂度不划算。重开条件:大 payload(图像/视频)场景解压成瓶颈。

## 测试覆盖

9 个测试 target,全绿:

- **C++ 单元/集成**(8):`ring_test`、`bootstrap_test`、`echo_test`、`byte_pool_test`、`byte_pool_echo_test`、`shm_sample_test`、`shm_insert_test`、`shm_crash_test`
- **Python e2e**(1):`shm_test`(sample/insert/structured_writer/selector/dtypes/pickle-refusal/lifecycle)
- 现有 `trajectory_writer_test`、`streaming_trajectory_writer_test` 无回归(SHM 是 additive 模式,未动 gRPC/local 路径)

崩溃恢复测试覆盖:client 崩溃 → server `ReleaseAll` 集中释放 + `shm_unlink` rings、其他 client 不受影响、新 client 重连;server 崩溃 → client 抛 `ConnectionError` 不永挂。

## 使用示例

```python
import reverb

# Server 启动 SHM 传输(可与 in_process 共存)
server = reverb.Server(
    tables=[reverb.Table(name='queue', sampler=reverb.selectors.Fifo(),
                         remover=reverb.selectors.Fifo(), max_size=10000)],
    in_process=True, shm=True)

# Client 经 SHM 连接(同 gRPC/LocalClient API)
client = reverb.ShmClient(server.shm_socket_path)
with client.trajectory_writer(num_keep_alive_refs=1) as w:
    w.append({'obs': np.array([1.0, 2.0], dtype=np.float32)})
    w.create_item(table='queue', num_timesteps=1, priority=1.0)
    w.flush()

for sample in client.sample('queue', num_samples=1):
    print(np.asarray(sample.data[0]))  # [[1.0, 2.0]]
```

## 已知遗留债务(非阻塞)

- **R12 signal→Stop**:SIGTERM handler 只 flip `running_`,完整 `Stop()` 靠 owner/析构;SIGTERM 直接 kill 才漏(SHM 段 + udsocket)。正常 `Server.stop()` 路径已覆盖。
- **RunShmWorker 与 RunLocalWorker 60%+ 重复**:结构性债务,提共享骨架是大重构,⑥ 后再说。
- **ShmServer 单表**:v1 ponytail 限制,多表延后。
- **ShmClient/LocalClient Python 重复**:`trajectory_writer`/`structured_writer` 近乎逐字复刻,可提共享 helper。
- **跨 ticket 累积**:`ErrnoStatus`/`ReadBlocking` 多处重复(4-5 份拷贝)。

## 文档

- `docs/numpy-shm-design.md` — 原始设计文档
- `docs/numpy-shm-spec.md` — 开发规范(审查问题 R1-R13、决策 C1-C5、文件布局、类声明、proto、pybind、5 阶段顺序、测试策略、边缘情况)
- `docs/shm-benchmark.md` — 基准结果 + v2 spike + DEFER 决策
- `tickets.md` — 7 个 tracer-bullet ticket 定义

---

**提交统计:** 16 commits,38 文件,+7301 行。从 spec 到交付完整闭环,每张票过 TDD + 两轴 review。最终 SHM sample 路径 ~11× 快于 gRPC,达无序列化天花板。
