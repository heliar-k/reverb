# 02 — SHM pool slab 档位/块数可配置（打通 Python → ShmServer → BytePool）

> 状态：**已完成**（2026-08-03；全量 65/65 绿。`ShmBytePool::Create` 顺带补
> 几何校验：严格升序 + 每档 ≥8B（free-list 指针写穿邻块的 footgun）。
> 偏差裁决：超最大档报 `InvalidArgument`（永久错误）而非 ticket 原写的
> `RESOURCE_EXHAUSTED`（瞬时档耗尽），语义更准且验收原文仍满足）

**What to build:** 每个 SHM 连接的 pool 段默认 ~1.4GB 高水位（9 档 slab ×
256 blocks，最大 4MB×256，`byte_pool.h: kDefaultSlabSizes`），tmpfs
ftruncate 稀疏分配、**按触及页计入 cgroup 内存配额**且高水位不回落——
大 payload 多连接场景在 16GiB pod 里已两次 OOM（benchmark §5.3）。
small-payload 场景根本用不到 1MB/4MB 档，却被迫按 1.4GB/连接预留容量。

`BytePool` 构造器**已接受** `slab_sizes` / `blocks_per_slab`
（`byte_pool.h:107-111`，空时用默认值），缺的是打通配置链路：

`Python Server(shm=True, shm_pool_slab_sizes=..., shm_pool_blocks_per_slab=...)`
→ pybind `ShmServer` 构造参数 → `ShmServer` ctor → `pool_` 构造。

默认不传时行为与现状完全一致（`kDefaultSlabSizes` /
`kDefaultBlocksPerSlab`）。这是 benchmark §5.2 重启 v2 的条件 (b) 的前置。

**Blocked by:** 01 — SHM 批量 INSERT（用户要求串行执行，无技术依赖）

**Status:** done（2026-08-03）

- [ ] Python `Server(shm=True)` 接受 slab 档位与每档块数两个可选参数，
      缺省时 pool 几何与现状逐字节一致
- [ ] 配置生效可验证：pool 段 ftruncate 容量 = Σ(档位 × 块数)；砍大档
      （如最大 256KB）后单连接高水位从 ~1.4GB 降到档位容量量级
- [ ] insert/sample 对象超过最大档位时客户端收到明确错误（非挂起/崩溃）；
      若现状已有该行为则补测试锁定
- [ ] 既有测试全绿；新增测试：自定义小档位配置下 insert+sample 往返正常
- [ ] 文档更新：`docs/guide/client-transports.md`（或 concepts.md）的
      容量规划公式改为可配口径，benchmark §5.3 加注
