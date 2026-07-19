# Tickets: 消除三客户端传输层代码冗余

基于 `docs/client-transports.md` 文档审计和全库扫描，将 Python/C++/pybind 三层中发现的高价值重复代码合并，降低维护面。

工作方式：从**前沿**（所有阻塞已完成的 ticket）开始，四张 ticket 无相互依赖，可并行推进。

---

## 合并 Python 层客户端 API 三份副本

**What to build**：`client.py` 中 `Client`/`LocalClient`/`ShmClient` 三个类的 `trajectory_writer` 和 `structured_writer` 方法合并为 `_BaseClient` 上的公共实现，差异部分（`max_chunk_length`、`validate_items`）参数化。同时移除 `_BaseClient.writer` 中与底层 C++ `Validate()` 重复的参数校验。

**Blocked by**：None — can start immediately。

- [ ] `trajectory_writer` 三份实现合并为 `_BaseClient` 上的一个方法，`max_chunk_length` 和 `validate_items` 通过子类属性或参数控制
- [ ] `structured_writer` 三份实现同样合并
- [ ] `_BaseClient.writer` 中冗余的参数范围校验移除，信任 C++ 层 `options.Validate()`
- [ ] 全量 Python 测试通过，三种客户端 `trajectory_writer`/`structured_writer`/`writer` 行为不变

---

## 提取 SHM 写入-读取-确认公共循环

**What to build**：`shm_client.cc` 中 `MutatePriorities`/`Reset`/`Checkpoint`/`ServerInfo` 四个方法共享的「序列化请求 → 加锁 → 写 ring → 读 ring → 解析响应 → 错误映射」模式，提取为模板方法 `SendInsertFlowRequest<Req, Resp>`，消掉约 150 行复制粘贴。

**Blocked by**：None — can start immediately。

- [ ] 模板方法 `SendInsertFlowRequest` 覆盖四个方法的公共流程，请求/响应类型参数化
- [ ] `MutatePriorities`/`Reset`/`Checkpoint`/`ServerInfo` 各自改为调用该模板
- [ ] 错误映射（`ShmError::NOT_FOUND` → `NotFoundError` 等）内聚在模板中，四处行为一致
- [ ] 全量 C++ 构建和 SHM 相关测试通过

---

## 消除 C++ 层 NewStructuredWriter 和 signature 填充三份副本

**What to build**：`client.cc`/`in_process_client.cc`/`shm_client.cc` 三份几乎逐字相同的 `NewStructuredWriter` 实现提取为单一自由函数；三份相同模式的 signature 填充逻辑提取为模板函数，各客户端通过钩子提供 `TableInfo` 来源。

**Blocked by**：None — can start immediately。

- [ ] `MakeStructuredWriter` 自由函数替代三份 `NewStructuredWriter` 重复实现
- [ ] `PopulateFlatSignatureMap` 模板函数消除三份 signature 填充副本，客户端提供 signature 获取钩子
- [ ] `client.h`/`in_process_client.h`/`shm_client.h` 中移除已不再需要的重复声明
- [ ] 全量 C++ 构建和传输层测试通过

---

## 消除 pybind 层重复序列化逻辑

**What to build**：`pybind.cc` 中重复三次的 `TableInfo` → `py::bytes` 序列化循环、重复三次的 `pair<uint64,double>` → `KeyWithPriority` proto 转换、重复三次的 `Sampler::Options` 构建，各提取为一个共享辅助函数。

**Blocked by**：None — can start immediately。

- [x] `SerializeTableInfoToPyBytes` 辅助函数消除三处重复序列化
- [x] `UpdatesToKeyWithPriorityProtos` 辅助函数消除三处重复 proto 转换
- [x] `BuildSamplerOptions` 辅助函数消除三处重复构造
- [x] 全量 pybind 构建和 Python API 测试通过，所有客户端方法行为不变
