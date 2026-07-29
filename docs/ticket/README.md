# Tickets 归档

已结案 ticket 组的索引。完整内容见同目录各文件。

## 归档索引

| 完成日期 | 主题 | 归档文件 | 做了什么 |
|----------|------|----------|----------|
| 2026-07 | 消除三客户端传输层代码冗余 | [client-transports-dedup.md](client-transports-dedup.md) | Python `_BaseClient` 合并 `trajectory_writer`/`structured_writer` 三份副本;C++ 提取 `SendInsertFlowRequest` 模板消掉 SHM 四方法 ~150 行复制;`MakeStructuredWriter` 自由函数替代三端 `NewStructuredWriter`(signature 填充判伪需求未提取);pybind 三处重复序列化/proto 转换/Options 构造各提取辅助函数 |
| 2026-07 | SHM 并发正确性 | [shm-concurrency-correctness.md](shm-concurrency-correctness.md) | `NewSampler` 拒绝同连接第二个活 sampler(强制 SPSC 不变式);`Ring::WriteSlots` 槽 0 最后发布,消掉多槽消息「continuation slot missing」致命错误 |
| 2026-07-25 | torch.Tensor 支持 Phase 1 | [torch-tensor-phase1.md](torch-tensor-phase1.md) | `torch_support.py` 写入路径归一化(三 writer + legacy);`output_format='torch'` 采样侧零拷贝;torch 可选依赖(`[torch]` extra);双轴 code-review 追加修复 |
| 2026-07-26 | numpy↔字节流零拷贝(Tier 1+2) | [numpy-zero-copy-tier1-2.md](numpy-zero-copy-tier1-2.md) | `TensorBuffer` 改 `shared_ptr` owner + view;采样侧 `ToNdArray` 零拷贝;写入侧 `FromNdArray(zero_copy=true)` opt-in(env 置 read-only);SHM 插入 `SerializeToArray` 直写 pool;torch 侧零拷贝自动成立 |
| 2026-07-26 | Tier 3 协议层自适应压缩 | [tier3-adaptive-compression.md](tier3-adaptive-compression.md) | `TensorProto.uncompressed` 标志;`WorthCompressing` 采样探测(≥16KB 取首/中/尾 3×4KB 估压缩率,≥0.9 判不可压)跳过 snappy;解压侧分流免 proto 拷贝;shm 768KB float insert +83%、sample +42% |

新增 ticket 时直接在本目录开新文件，结案后把一行总结追加到上表。
