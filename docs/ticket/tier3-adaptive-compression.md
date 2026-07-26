# Tickets: Tier 3 协议层自适应压缩（2026-07-26 完成）

> 归档自原 `tickets.md` 活动板。本组 ticket 全部结案。

profile 驱动:`tier3_bench`(新增诊断工具,//reverb/cc/support:tier3_bench)
分解写入链发现热点不是 Tier1+2 遗留假设的 Concat/set_tensor_content 物化拷贝
(合计仅占 insert E2E ~4%),而是 `CompressTensorAsProto` 对数值 tensor 无条件
snappy 压缩——768KB 随机 float 上 ~1000μs ≈ insert E2E 的 37%,输出≥输入,
SHM/in-process 传输下纯浪费。

**What to build**:自适应压缩。`TensorProto` 加 `bool uncompressed = 5`
(proto3 加字段 wire 兼容,旧数据默认 false=压缩态);写入侧 payload ≥16KB
时采样首/中/尾 3×4KB 估算压缩率,≥0.9 判定不可压 → 存原始字节+置标志位,
跳过全量 snappy;解压侧按标志位直接 DeserializeFromProto,顺带跳过原有的
`inflated = proto` 整 proto 拷贝。可压数据(图像/稀疏观测)路径不变。

- [x] proto 字段 + `WorthCompressing` 采样探测 + 解压分流
- [x] 单测两例(随机 bytes 判定不可压+roundtrip;常量 bytes 维持压缩
  +roundtrip);初版"可压"用例误用 `Pattern<int32_t>`(步进序列对 snappy
  实际不可压——4 字节最小匹配被低位变化打断),改常量数据
- [x] 全量回归 65 目标:64 绿;`byte_pool_echo_test` 为存量时序 flake
  (200 次 sched_yield 连接窗口,依赖闭包与本改动零交集),单独 5/5 通过
- [x] 基准复测(shm,768KB float32 随机,table 1000):
  insert 378→691 ops/s(+83%),p50 2557→1424μs(-44%,省下 ~1130μs ≈
  profile 预测的 snappy 成本);sample 1507→2135 ops/s(+42%),
  p50 652→468μs;微基准 CompressTensorAsProto 1036→38μs(×33→×1.2 地板)

> **遗留(已关闭,2026-07-26)**:wire 行 ~470-490μs@768KB 的 mmap 阈值假设
> 经证伪为**微基准假象,非真实路径成本**。实验:`GLIBC_TUNABLES` 抬
> mmap_threshold(16MB)+trim_threshold(128MB)后,微基准 wire 行
> 485→119μs/467→94μs、缺页 725K→2.4K(阈值假设本身成立——仅抬
> mmap_threshold 反而更慢:glibc 动态适应把阈值钉在恰好等于消息尺寸,
> `nb >= threshold` 判定下同尺寸分配永远走 mmap;须同时抬 trim 才跳出
> churn);但 e2e 复测(client_benchmark,shm|large|1000)零变化:
> insert 708→687 ips(p50 1413→1399μs)、sample 2093→2069 sps(p50
> 470→470μs),均在噪声内。真实路径尺寸有自然波动,动态适应正常生效,
> 稳态走 free-list 复用。arena/tcmalloc 判为伪需求,不再跟进;
> Concat/set_tensor_content 拷贝(~4%)确认不值得动。
