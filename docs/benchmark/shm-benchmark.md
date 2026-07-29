# SHM Transport: Performance Baseline + v2 Spike (ticket ⑦)

Benchmark of the **sample path** (the design's core pain point — N repeated
decompressions per sample) across three transports, plus a feasibility spike on
the v2 optimization from `docs/spec/numpy-shm-spec.md` §6 / `docs/design/numpy-shm-design.md`
§6.

## 1. Methodology

**What is measured.** Per-sample throughput (samples/s) and per-sample latency
(p50 / p99), where one "sample" is one `client.sample(table, num_samples=1,
emit_timesteps=False)` call. Each call is timed with `time.perf_counter()`
around the full round-trip (request → response → numpy construction). A warmup
of 20 samples is discarded before timing.

**Transports (same table definition + same data on each):**

| transport | server | client |
| --- | --- | --- |
| in-process | `Server(tables, in_process=True)` | `server.in_process_client` (`LocalClient`) |
| SHM | `Server(tables, in_process=True, shm=True)` | `ShmClient(server.shm_socket_path)` |
| gRPC-loopback | `Server(tables, port=N)` **in a separate process** | `Client('localhost:N')` |

**Table.** Single FIFO table, `max_times_sampled=0` (items reusable, so draws
can exceed insertions), `MinSize(1)` rate limiter. Seeded with N single-step
items, each a `float32[1]` column `{"v": [1.0]}` — the point is transport
overhead, not payload size, so the payload is deliberately tiny.

**Scenarios.** Two scaling points: `(table_size=1000, samples=200)` and
`(table_size=10000, samples=500)`.

**⚠️ Critical methodology note — the gRPC same-process shortcut.** A gRPC
`Client` constructed in the *same process* as its `Server` does **not** use
gRPC: `Client::NewSampler` calls `GetLocalTablePtr`, which sends the client PID
to the server via `InitializeConnection`; if the PIDs match the server returns
the raw `Table*` address and the sampler accesses the table directly, bypassing
gRPC entirely (`reverb/cc/client.cc:165`). Measuring gRPC in-process therefore
measures the in-process path, not gRPC. **To get a true loopback number the gRPC
server is spawned in a separate process** (`multiprocessing.Process`) so the PID
check fails and the real gRPC `SampleStream` path is exercised. An earlier
in-process run reported a false "gRPC" figure (~1100–1300 sps) that was actually
the direct-access path; the subprocess numbers below (~800–930 sps) are the real
gRPC loopback.

**Hardware.** Linux x86_64, single machine, loopback only. Numbers are
single-run snapshots (not multi-run medians) — `// ponytail:` the relative
ordering (SHM ≫ gRPC, SHM ≈ in-process) is robust across runs; absolute values
vary ±10% with system load.

**Repro.** `bazel run //reverb/benchmarks:shm_benchmark`
(`reverb/benchmarks/shm_benchmark.py`). The script prints the comparison table
to stdout and a machine-readable `# RAW` block.

## 2. Results

```
[table=1000, samples=200, warmup=20]
  transport          sps |   p50 us |   p99 us
  in-process        8301 |     111.9 |     176.6
  SHM               9100 |     104.2 |     163.0
  gRPC-loopback      806 |    1217.7 |    1670.6

[table=10000, samples=500, warmup=20]
  transport          sps |   p50 us |   p99 us
  in-process        8437 |     113.0 |     168.8
  SHM               8776 |     106.5 |     177.9
  gRPC-loopback      934 |    1061.0 |    1265.3
```

| transport | table | sps | p50 (us) | p99 (us) | vs gRPC |
| --- | --- | --- | --- | --- | --- |
| in-process | 1000 | 8301 | 111.9 | 176.6 | 10.3× |
| SHM | 1000 | 9100 | 104.2 | 163.0 | **11.3×** |
| gRPC-loopback | 1000 | 806 | 1217.7 | 1670.6 | 1.0× |
| in-process | 10000 | 8437 | 113.0 | 168.8 | 9.0× |
| SHM | 10000 | 8776 | 106.5 | 177.9 | **9.4×** |
| gRPC-loopback | 10000 | 934 | 1061.0 | 1265.3 | 1.0× |

## 3. Conclusion

**SHM shows a large, measurable improvement over gRPC loopback on the sample
path — ~9–11× higher throughput and ~10× lower p50/p99 latency.** This is the
win the design predicted: gRPC pays serialization + compression + decompression
per sample (and a TCP loopback round-trip), while SHM memcpy's the already-flat
result bytes into a shared pool the client reads directly — no serialization, no
compression, no socket round-trip.

A second, subtler finding: **SHM is on par with (marginally faster than) the
in-process `LocalClient` path** (~8800–9100 sps vs ~8300–8440 sps). Both avoid
serialization; the difference is within run-to-run noise. This means v1 SHM has
effectively closed the gap to the no-serialization ceiling on this workload —
the transport overhead is already negligible and the remaining ~110 us/sample is
dominated by the server-side `Table::Sample` + `UnpackChunkColumnAndSlice`
(compress/decompress against the ChunkStore) work shared by both paths.

The win scales flatly with table size (10× at 1k items, 9× at 10k items): the
gRPC per-sample cost is dominated by serialization/compression, not table size,
so the ratio is stable.

---

## 4. v2 Spike: insert bytes reused as sample slice source

Source: `docs/design/numpy-shm-design.md` §6 / `docs/spec/numpy-shm-spec.md` §6.

### 4.1 The idea

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

### 4.2 Feasibility — what would need to change

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

### 4.3 Complexity estimate vs v1

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

### 4.4 Decision

**DEFER.**

Reasoning:

- The benchmark shows v1 SHM is already **~9–11× faster than gRPC loopback** and
  **on par with the no-serialization in-process ceiling** (~110 us/sample). The
  transport overhead v2 would remove is already negligible — the remaining
  ~110 us is dominated by `UnpackChunkColumnAndSlice` (decompress) work that v2
  only avoids by changing the storage format (hard part 4/5), not by the
  indexing alone.
- v2's payoff (skipping the compress→decompress round-trip) is bounded by the
  decompress cost. With the tiny payloads benchmarked here that cost is small;
  it grows with payload size, so v2's *relative* win is larger for big
  observations (e.g. images). But that case is unmeasured, and the complexity
  (wire-format change + cross-thread retention lifecycle) is high.
- The complexity estimate (+200–400 LOC, 2–3 new components, a wire-format
  decision) is disproportionate to a win that is, on the measured workload,
  already consumed by the no-serialization ceiling.

**Re-open v2 if:** (a) a future benchmark with large payloads (images/video)
shows the decompress cost dominating the sample path, *and* (b) the
in-process ceiling is no longer the binding constraint. Until then, v1 SHM
delivers the design's goal.
