# Tickets: 消除三客户端传输层代码冗余（2026-07 完成）

> 归档自原 `tickets.md` 活动板。本组 ticket 全部结案。

基于 `docs/client-transports.md` 文档审计和全库扫描，将 Python/C++/pybind 三层中发现的高价值重复代码合并，降低维护面。

工作方式：从**前沿**（所有阻塞已完成的 ticket）开始，四张 ticket 无相互依赖，可并行推进。

---

## 合并 Python 层客户端 API 三份副本

**What to build**：`client.py` 中 `Client`/`LocalClient`/`ShmClient` 三个类的 `trajectory_writer` 和 `structured_writer` 方法合并为 `_BaseClient` 上的公共实现，差异部分（`max_chunk_length`、`validate_items`）参数化。同时移除 `_BaseClient.writer` 中与底层 C++ `Validate()` 重复的参数校验。

**Blocked by**：None — can start immediately。

- [x] `trajectory_writer` 三份实现合并为 `_BaseClient` 上的一个方法，`max_chunk_length` 和 `validate_items` 通过子类属性或参数控制
- [x] `structured_writer` 三份实现同样合并
- [x] `_BaseClient.writer` 中冗余的参数范围校验移除，信任 C++ 层 `options.Validate()` （经核查非冗余：legacy `Writer` 路径无 `options.Validate()`，gRPC `Client::NewWriter` 零校验，`InProcessClient::NewWriter` 仅校验三个 `<1` 不含 `chunk_length>max`；移除会将 ValueError 退化为 C++ CHECK abort 或静默构造损坏的 writer，故保留并加注，见 client.py `_BaseClient.writer` docstring）
- [x] 全量 Python 测试通过，三种客户端 `trajectory_writer`/`structured_writer`/`writer` 行为不变

---

## 提取 SHM 写入-读取-确认公共循环

**What to build**：`shm_client.cc` 中 `MutatePriorities`/`Reset`/`Checkpoint`/`ServerInfo` 四个方法共享的「序列化请求 → 加锁 → 写 ring → 读 ring → 解析响应 → 错误映射」模式，提取为模板方法 `SendInsertFlowRequest<Req, Resp>`，消掉约 150 行复制粘贴。

**Blocked by**：None — can start immediately。

- [x] 模板方法 `SendInsertFlowRequest` 覆盖四个方法的公共流程，请求/响应类型参数化
- [x] `MutatePriorities`/`Reset`/`Checkpoint`/`ServerInfo` 各自改为调用该模板
- [x] 错误映射（`ShmError::NOT_FOUND` → `NotFoundError` 等）内聚在模板中，四处行为一致
- [x] 全量 C++ 构建和 SHM 相关测试通过

---

## 消除 C++ 层 NewStructuredWriter 和 signature 填充三份副本

**What to build**：`client.cc`/`in_process_client.cc`/`shm_client.cc` 三份几乎逐字相同的 `NewStructuredWriter` 实现提取为单一自由函数；三份相同模式的 signature 填充逻辑提取为模板函数，各客户端通过钩子提供 `TableInfo` 来源。

**Blocked by**：None — can start immediately。

- [x] `MakeStructuredWriter` 自由函数替代三份 `NewStructuredWriter` 重复实现
- [ ] `PopulateFlatSignatureMap` 模板函数消除三份 signature 填充副本，客户端提供 signature 获取钩子
- [ ] `client.h`/`in_process_client.h`/`shm_client.h` 中移除已不再需要的重复声明
- [ ] 全量 C++ 构建和传输层测试通过

> **实现说明（2026-07）**：`MakeStructuredWriter` 已提取为 `structured_writer.h/.cc` 中的自由函数，三端 `NewStructuredWriter` 收敛为传入各自 `NewTrajectoryWriter` 钩子的薄封装。
>
> `PopulateFlatSignatureMap` 刻意不提取——经核查三端 signature 填充路径为「概念相似但实现不同」而非复制粘贴：gRPC 走 `MaybeUpdateServerInfoCache` 从缓存 RPC 取 `RepeatedPtrField<TableInfo>` 调 `FlatSignatureFromTableInfo`；in-process 遍历本地 `flat_hash_map<string,shared_ptr<Table>>` 调 `FlatSignatureFromSignatureProto` 且有 `has_value()` 条件分支；SHM 从 live `SERVER_INFO` ring 取 `std::vector<TableInfo>` 调 `FlatSignatureFromTableInfo`。容器类型、助手函数、控制流均不同，共享的仅「循环 + 赋给 `signatures[name]`」约三行。模板化需双钩子（元素转换 + 源获取）换三行收益，净增复杂度，故判为伪需求。
>
> 头声明无冗余可删：`NewStructuredWriter` 仍是三端的公开 API 入口（pybind/tests/Python 直接调用），提取的是函数体而非方法本身；仅同步修正了 `in_process_client.h`/`shm_client.h` 中描述内部实现的注释指向 `MakeStructuredWriter`。

---

## 消除 pybind 层重复序列化逻辑

**What to build**：`pybind.cc` 中重复三次的 `TableInfo` → `py::bytes` 序列化循环、重复三次的 `pair<uint64,double>` → `KeyWithPriority` proto 转换、重复三次的 `Sampler::Options` 构建，各提取为一个共享辅助函数。

**Blocked by**：None — can start immediately。

- [x] `SerializeTableInfoToPyBytes` 辅助函数消除三处重复序列化
- [x] `UpdatesToKeyWithPriorityProtos` 辅助函数消除三处重复 proto 转换
- [x] `BuildSamplerOptions` 辅助函数消除三处重复构造
- [x] 全量 pybind 构建和 Python API 测试通过，所有客户端方法行为不变
