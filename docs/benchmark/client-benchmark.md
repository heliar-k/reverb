# Client Transports: Performance Baseline + v2 Spike (ticket ⑦)

Benchmark of the three client transports — in-process (`LocalClient`), SHM
(`ShmClient`), gRPC loopback (`Client`) — across the **sample**, **insert**,
and **concurrent pipeline** paths, plus a feasibility spike on the v2
optimization from `docs/spec/numpy-shm-spec.md` §6 / `docs/design/numpy-shm-design.md` §6.

> 前身文档只测 sample 路径（原名 shm-benchmark.md）。本次扩展：insert 路径
> （含 flush 批量效果）、并发读写混合（表 worker 饱和点）、大 payload。

## 1. Methodology

**What is measured.**

- *sample*: per-call throughput (samples/s) and p50/p90/p99 latency of
  `client.sample(table, num_samples=1, emit_timesteps=False)`, timed with
  `time.perf_counter()` around the full round-trip.
- *insert*: per-item throughput (items/s) and per-**flush** latency of
  single-step items via `trajectory_writer`, at batch sizes 1/8/64 items per
  flush.
- *pipeline*: W writer threads + R reader threads running concurrently for 5s;
  aggregate insert/sample throughput + latencies. Extra threads get **their own
  client connections** (SHM/gRPC; in-process threads share the one
  `LocalClient`, which is just table pointers).

**Transports (same table definition + same data on each):**

| transport | server | client |
| --- | --- | --- |
| in-process | `Server(tables, in_process=True)` | `server.in_process_client` (`LocalClient`) |
| SHM | `Server(tables, in_process=True, shm=True)` | `ShmClient(server.shm_socket_path)` |
| gRPC-loopback | `Server(tables, port=N)` **in a separate process** | `Client('localhost:N')` |

**Table.** Single FIFO table, `max_times_sampled=0` (items reusable, so draws
can exceed insertions), `MinSize(1)` rate limiter, `max_size=20000`. Seeded
with N single-step items.

**Payloads.** `small=float32[1]` (~4B, transport-overhead probe),
`med=float32[84,84,4]` (~110KB, small image), `large=float32[256,256,3]`
(~770KB, large observation).

**⚠️ Critical methodology note — the gRPC same-process shortcut.** A gRPC
`Client` constructed in the *same process* as its `Server` does **not** use
gRPC: `Client::NewSampler` calls `GetLocalTablePtr`, which sends the client PID
to the server via `InitializeConnection`; if the PIDs match the server returns
the raw `Table*` address and the sampler accesses the table directly, bypassing
gRPC entirely (`reverb/cc/client.cc:165`). Measuring gRPC in-process therefore
measures the in-process path, not gRPC. **To get a true loopback number the gRPC
server is spawned in a separate process** (`multiprocessing.Process`) so the PID
check fails and the real gRPC `SampleStream` path is exercised.

**Environment (this run).** Kubernetes pod, cgroup v1: **memory limit 16 GiB**
(`/sys/fs/cgroup/memory/memory.limit_in_bytes`), CPU quota **8 cores**
(`cfs_quota 800000/100000`, cpuset 3,5,7,9,43,45,47,49), Linux x86_64,
loopback only. ⚠️ `free`/`/proc/meminfo` in a container show **node** memory
(251GB), not the pod limit — two benchmark runs were OOM-killed (cgroup
`oom_kill` counter) before the matrix was sized to the pod. Numbers are
single-run snapshots; relative ordering is robust across runs, absolute values
vary ±10% with system load. SHM pool segments are tmpfs and **count toward the
pod's 16 GiB** (see §5.3).

**Repro.**

```bash
# 全矩阵（=本文 core 批次）
bazel run //reverb/benchmarks:client_benchmark -- \
  --mode all --transport all --payload small,med --table-size 1000,10000 \
  --batch 1,8,64 --writers 1,2,4,8 --readers 1,2,4,8 --duration 5 --num-ops 500
# 大 payload（按传输拆进程，避免单进程跨阶段内存累积，见 §5.3）
for t in inprocess shm grpc; do
  bazel run //reverb/benchmarks:client_benchmark -- \
    --mode all --transport $t --payload large --table-size 500 \
    --num-ops 100 --batch 1,8 --writers 1,2 --readers 1,2 --duration 5
done
```

Script: `reverb/benchmarks/client_benchmark.py`. Prints comparison tables +
machine-readable `# RAW` block; `--json` for JSON. 单配置失败会记入
`ERRORS` 段并继续矩阵（不拖垮整批）。

## 2. Sample path（单 client 顺序采样）

吞吐 samples/s（p50 延迟）：

| transport | small/1k | small/10k | med/1k | med/10k |
| --- | --- | --- | --- | --- |
| in-process | 11967 (82us) | 7942 (105us) | 11706 (84us) | 1821 (98us) |
| SHM | 3779 (247us) | 4482 (221us) | 4043 (237us) | 4005 (241us) |
| gRPC | 1465 (657us) | 1506 (655us) | 1282 (770us) | 1191 (802us) |

- **gRPC** 全程 ~1.2–1.5k sps，被 per-call RPC 开销钉死，payload/表大小不敏感。
- **SHM** ~4k sps，对 payload 与表大小都稳——传输零拷贝，成本在 per-call
  ring 往返。
- **in-process** 小表最快（~12k），但 **med/10k 塌到 1821**——1.1GB 表上
  直接访问散布在 chunkstore 里的压缩 chunk，cache/TLB 压力主导；同配置
  SHM（4005）反而快 2.2 倍，因为结果字节是拷贝进紧凑 pool 区域的。
- 与旧版文档（shm-benchmark 时代）数字不可直接比：旧数据在另一台机器
  （无 cgroup CPU 配额），SHM 相对 gRPC 的倍数从 ~9–11× 变为 ~3×，主要
  是 gRPC 在本环境变快（1.5k vs 806 sps）。**排序结论不变**：
  in-process ≥ SHM ≫ gRPC。

## 3. Insert 路径（单 writer，flush 批量效应）

吞吐 items/s；延迟为 per-**flush**（b64 时含 64 items）：

| transport | small b1 | small b8 | small b64 | med b1 | med b8 | med b64 |
| --- | --- | --- | --- | --- | --- | --- |
| in-process | 6036 | 21284 | 25083 | 3294 | 7263 | 14275 |
| SHM（ticket 01 后） | 1685 | 8141 | 20545 | 1543 | 3774 | 8593 |
| gRPC | 1722 | 5740 | 17471 | 1397 | 4110 | 6542 |

（表 1k；10k 趋势相同，见 `# RAW`。SHM 行为 2026-08-03 ticket
[shm-batch-insert](../ticket/shm-batch-insert.md) 落地后复测，
其余两行是原批次数字。）

- **in-process** 批量摊薄明显（b1→b64 ≈ 4×）。
- **gRPC** 批量收益大（b1→b64 ≈ 10×）：streaming 天然流水，批量把
  per-RPC 固定成本摊掉。
- **SHM 批量已生效**（ticket 01 前 b64 仅 ~2.4–2.9k ips）：`RunShmWorker`
  现在把至多 64 个 ready item 合并成**一条** `ShmInsertRequest`——
  流水化 ALLOCATE burst（N 发 N 收按序配对）+ 单条 INSERT + 聚合 ACK，
  一批 ~2 次 ring 往返（旧实现每 item 2 次）。b64 提速 **small 7.1× /
  med 3.6×**，超过 gRPC b64（17471/6542）；b1 不变（单 item 仍是 2 次
  往返）。b64 的残余成本是服务端 per-item 工作（parse/压缩/表插入）——
  约 45us/item，不再是传输往返。

## 4. 并发读写混合（pipeline，5s 稳态）

聚合吞吐（表 10k，w1r1 → w8r8）：

| transport | ins/s w1→w8 | sam/s w1→w8 | 饱和点 |
| --- | --- | --- | --- |
| in-process (small) | 4007 → 3411 | 7457 → 6913 | **w1 已饱和**：加线程只增争抢 |
| SHM (small) | 1546 → 2494 | 3821 → 6616 | w4 附近平台（dispatch 单线程 + 表 worker） |
| gRPC (small) | 1441 → 1900 | 1112 → 2260 | w4 后微增，p99 劣化到 ~15ms |
| in-process (med) | 2867 → 2826 | 5654 → 5829 | w1 饱和 |
| SHM (med) | 1435 → 2090 | 2950 → 5155 | 同 small |
| gRPC (med) | 1091 → 1719 | 960 → 1932 | 同 small |

- **单表并发天花板 = 单 table worker 线程**：in-process 下 w1r1 即满，
  w8r8 反而略降（锁争抢）。要更高单表吞吐只能靠**加表分片**（多 worker
  线程），不是加 client 线程。
- **SHM 采样随连接数扩展良好**（w1→w8 采样 ~1.7×）：每连接独立 ring，
  服务端单 dispatch 线程轮询不是瓶颈直到 w4 后。写入受 per-item 往返
  限制，多连接并行写 ~1.7×。
- **w8r8 全配置存活**——修复 ticket「shm-close-while-in-flight」前，
  此场景会让客户端进程无声死亡（连接被服务端回收后客户端读循环无限
  自旋，见 `docs/ticket/shm-close-while-in-flight.md`）。
- 资源列（`# RAW` 有 rss/cpu）：SHM w8r8 服务端 ~5.5 核（dispatch 自旋
  + 表 worker + checkpoint executor），gRPC ~2.2 核。

## 5. 大 payload（float32[256,256,3] ≈ 770KB）

（本批次因 §5.3 的内存约束按传输拆进程跑，
`--table-size 500 --num-ops 100 --writers 1,2`。）

### 5.1 结果

sample / insert 吞吐（表 500，单 client）：

| transport | sample sps (p50) | insert b1 ips | insert b8 ips |
| --- | --- | --- | --- |
| in-process | 7427 (129us) | 827 | 1191 |
| SHM | 2201 (449us) | 568 | 938 |
| gRPC | 667 (1370us) | 595 | 1109 |

pipeline（5s，聚合）：

| transport | w1r1 ins/sam | w2r2 ins/sam |
| --- | --- | --- |
| in-process | 1147 / 3185 | 1289 / 2810 |
| SHM | 661 / 1326 | 857 / 1436 |
| gRPC | 498 / 563 | 717 / 708 |

- 大 payload 下三者的 **insert 趋同**（b1 ~570–830 ips）：770KB 的序列化/
  压缩主导，传输差异被摊平。SHM b1→b8 有 ~1.7×（本行为 ticket 01 前批
  次；ticket 01 后批量效应见 §3）。
- **sample 差距拉开**：in-process 7427（无序列化直切）≫ SHM 2201 ≈ 3.3×
  gRPC 667。SHM sample 的 p50 从 small 247us 涨到 449us，**payload 相关
  成本 ≈ 200us/770KB-sample**（服务端解压 + memcpy 进 pool + 客户端读出）。
- 内存：in-process w2r2 峰值 RSS ~10.3GB、SHM w2r2 ~6.8GB（`# RAW`）。

### 5.2 v2（insert 字节复用为 sample 切片源）重启裁决

旧版 §4 的 re-open 条件：「大 payload benchmark 显示解压成本主导 sample
路径，且 in-process 天花板不再是约束」。用本节数据复核：

- v2 能消除的是服务端 **解压→memcpy 进 pool** 那一段，量级 ≈
  100–200us/770KB-sample，占 SHM large sample 耗时的 **~20–45%**——
  有真实收益，但有上界：客户端从 pool 读出的那份拷贝 v2 消不掉。
- 新出现的数据点**对 v2 不利**：v2 要在 pool 里长期保留 insert 原始
  （未压缩）字节，而 §5.3 刚证明 pool 触及页就是内存高水位——v2 会把
  高水位从「流量工作集」抬到「全量未压缩表内容」，在 16GiB pod 里这是
  硬伤。
- **裁决：维持 DEFER。** 重启条件更新为：(a) payload 远大于 1MB 且采样
  侧成为系统瓶颈；(b) pool 内存配额不再是约束（如 slab 档位/尺寸可配
  或节点级部署）。

### 5.3 SHM 内存成本：每连接 pool 高水位

大 payload 批次两次被 cgroup OOM 杀死（`oom_kill` 97→100），定位结论：

- **每个 SHM 连接的 pool 段默认 ~1.4GB**（9 档 slab × 256 blocks，
  `byte_pool.h: kDefaultSlabSizes` 最大 4MB×256）。ftruncate 稀疏分配，
  **按触及页计费**（tmpfs 计入 cgroup 内存配额），高水位后不回落。
- w4r4 一轮 pipeline 会建 7 条 factory 连接 + 1 基础连接 + server 侧
  pool ≈ 9 个 pool 映射，大 payload 流量几分钟内把触及页推向 ~12GB。
- 单进程跨传输阶段（in-process→SHM→gRPC）内存**只增不减**（表、pool、
  mmap 高水位叠加），大 payload 必须按传输拆进程跑。
- **容量规划公式**：SHM 服务端内存 ≈ 表内容 + Σ（活跃连接数 × pool
  高水位）；多客户端大 payload 场景按 ~1.4GB/连接预留。
- **pool 几何已可配**（ticket 02）：`Server(shm_pool_slab_sizes=...,
  shm_pool_blocks_per_slab=...)`，pool 容量 = Σ(档位 × 块数)；上述
  1.4GB/连接 为默认档位（9 档 × 256 blocks）口径。small-payload 场景
  砍掉大档即可把单连接高水位降到档位容量量级。

旧文档 §4 的 v2 复杂度分析仍然成立，摘录在下节。

## 6. v2 Spike: insert bytes reused as sample slice source（原文保留）

Source: `docs/design/numpy-shm-design.md` §6 / `docs/spec/numpy-shm-spec.md` §6.

### 6.1 The idea

**v1 (current) sample round-trip** (`reverb/cc/shm/shm_server.cc` `HandleSample`,
`HandleInsert`):

- *Insert*: client memcpy's a serialized `ChunkData` proto into the SHM pool →
  server `ParseFromArray` → wraps it in a `ChunkStore::Chunk` (which owns a
  *copy* of the proto) → `Table::InsertOrAssignAsync`. The original SHM insert
  bytes are then released (`INSERT_ACK.offsets_to_release`).
- *Sample*: `Table::Sample` → for each column slice,
  `UnpackChunkColumnAndSlice` (decompress the chunk from the ChunkStore proto)
  → concat slices → `pool_.Allocate` + `memcpy` the flat result bytes into the
  pool → `SAMPLE_RESP`. Client reads, then `RELEASE`.

So v1 does, per sample: **decompress-from-ChunkStore → memcpy into pool**. The
insert bytes were discarded after insert; the sample bytes are a fresh
decompressed copy.

**v2**: the server maintains a `chunk_key → SHM pool offset` index. On insert,
instead of (or in addition to) compressing into the ChunkStore, the *original*
insert bytes are retained in the pool. On sample, the server slices the sample
columns directly from the original insert bytes — skipping the
compress-into-ChunkStore-then-decompress round-trip entirely. Zero-copy from
insert to sample.

### 6.2 Feasibility — what would need to change

1. **Server-side `chunk_key → SHM offset` index** (in `ShmServer`, alongside
   the ChunkStore). New component: a map from chunk key to the pool offset +
   length + column layout of the retained insert bytes. ~50–100 LOC.
2. **Pool retention policy (the hard part).** v1 releases insert bytes on
   `INSERT_ACK` (C2). v2 must *keep* them alive until no sample references them.
   This is a **second reference-counting layer** across chunks: a chunk's insert
   bytes can be dealloc'd only when (a) the ChunkStore has evicted the chunk
   *and* (b) no outstanding sample references it. v1 already refcounts *sample*
   pool offsets (C3); v2 extends this to *insert* offsets with a different
   lifecycle (long-lived, tied to chunk eviction, not per-sample).
3. **ChunkData eviction when the table is full.** When the FIFO remover evicts
   an item, its referenced chunks may become unreferenced; the ChunkStore drops
   them. v2 must then also release the corresponding insert pool offsets — but
   only if no in-flight sample still slices from them. This couples pool
   retention to the Table's removal callback, which today fires asynchronously
   on the table worker thread (not the dispatch thread that owns the allocator —
   see A1/R11). A cross-thread handoff back to the dispatch thread is needed to
   keep allocation single-threaded.
4. **Column layout from raw bytes.** `UnpackChunkColumnAndSlice` today operates
   on a *deserialized* `ChunkData` proto (it decompresses compressed column
   data). v2's "slice directly from insert bytes" only works if the insert bytes
   are stored *uncompressed* and the column slice boundaries are known without
   full deserialization. But the insert path deliberately **compresses**
   (`CompressTensorAsProto`) for ChunkStore storage efficiency. So v2 must
   either (a) store an *uncompressed* copy in the pool (memory cost: doubles
   retention), or (b) re-decompress on sample anyway (defeating the point), or
   (c) change the wire format so columns are independently sliceable. (c) is a
   non-trivial format change touching `ChunkData`/chunker.
5. **Delta encoding** breaks direct slicing further: a delta-encoded column
   can't be sliced without reconstructing the prefix.

### 6.3 Complexity estimate vs v1

| component | v1 (exists) | v2 (new/changed) |
| --- | --- | --- |
| `HandleSample` | decompress + concat + memcpy into pool | **replaced**: index lookup + offset arithmetic + (if uncompressed) direct offset return; ~40 LOC of the ~80-LOC `HandleSample` body removed |
| `HandleInsert` | parse proto → ChunkStore::Chunk → InsertOrAssignAsync | **extended**: also register `chunk_key → offset` in the index, mark offset as retained |
| `HandleRelease` | unref sample offsets | **extended**: also unref insert offsets when ChunkStore evicts |
| pool refcounting | sample offsets only (C3) | **doubled**: insert-offset refcounting with chunk-eviction-driven release |
| Table removal callback | none (ChunkStore handles its own eviction) | **new**: hook eviction → release retained insert offsets on dispatch thread |
| wire format / chunk storage | compressed ChunkData proto | **potentially changed** if independent column slicing is required |

Rough estimate: **+200–400 LOC and 2–3 new components** (index, eviction hook,
retention refcounting) vs v1's sample path. The existing `HandleSample` body
(~80 LOC) is partially replaced, but the bulk of v1's infrastructure (ring,
pool, bootstrap, dispatch loop, client side) is reused unchanged.

The genuinely hard parts are (4)/(5) — the wire format — and (3) — coupling
retention to async chunk eviction across threads. These are not
afternoon-of-work changes; they touch the chunk storage format and the
Table↔pool lifecycle.

### 6.4 Decision

**DEFER（旧版裁决，2026-08 用大 payload 数据复核，见 §5.2）。**

Reasoning:

- The benchmark shows v1 SHM is already **faster than gRPC loopback** and
  **on par with the no-serialization in-process ceiling** on small payloads.
  The transport overhead v2 would remove is already negligible — the remaining
  per-sample cost is dominated by `UnpackChunkColumnAndSlice` (decompress)
  work that v2 only avoids by changing the storage format (hard part 4/5),
  not by the indexing alone.
- v2's payoff (skipping the compress→decompress round-trip) is bounded by the
  decompress cost. With the tiny payloads originally benchmarked that cost is
  small; it grows with payload size, so v2's *relative* win is larger for big
  observations (e.g. images). 大 payload 复核见 §5.2。
- The complexity estimate (+200–400 LOC, 2–3 new components, a wire-format
  decision) is disproportionate to a win that is, on the measured workload,
  already consumed by the no-serialization ceiling.

**Re-open v2 if:** (a) 大 payload 数据显示解压成本主导 sample 路径（§5.2 的
复核结论）, *and* (b) the in-process ceiling is no longer the binding
constraint.
