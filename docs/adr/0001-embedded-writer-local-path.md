# 内嵌模式补全 Writer/insert 的本地路径(推翻 A1 砍 Writer 决策)

## 背景

设计文档 §1.3 决策 A1 当初决定:内嵌模式只做 `TrajectoryWriter` + `StructuredWriter`,
旧 `Writer`/`StreamingTrajectoryWriter` 的本地路径"价值低,砍掉省复杂度"。这导致
`LocalClient` 无 `insert`/`writer`,与 gRPC `Client` 的 API 不对称(§3.5),并连带逼出
`LocalClient.sample` 默认 `emit_timesteps=False` 的偏差(§3.4)。

## 触发推翻的事实

核实代码后发现 A1 的前提站不住:

- `Writer` **没有真正废弃**:仍在 `__init__.py:44` 正式导出,docstring 是 "will
  eventually be deprecated"(将来某天),非已废弃;本 fork `__init__.py:24-28` 还主动
  恢复了它(去掉 `NotImplementedError`)。
- `insert` **硬依赖** `writer`(`client.py:460`):`insert` 实现就是
  `with self.writer(max_sequence_length=1) as w: w.append(data); w.create_item(...)`。
  `insert` 是 gRPC `Client` 高频主力 API(tests 26 处用法),A1 砍 `writer` 等于内嵌
  用户也丢了 `insert` 这个便捷入口。
- gRPC `Client` 三套写入 API(`insert`/`writer`/`trajectory_writer`)并存且都活,
  `trajectory_writer` 是推荐项但不替代另两者。

## 决策

**推翻 A1 关于 Writer 的部分**:给 `Writer` 类加本地路径(基于 server 侧
`ProcessIncomingRequest` 的三件事——`SaveChunks`/`GetItemWithChunks`/
`InsertOrAssignAsync`——全不依赖 gRPC,可平移进 `Writer::WritePendingData`)。`LocalClient`
补 `writer`/`insert`,与 gRPC `Client` API 严格镜像。

A1 的"省复杂度"理由被"API 严格镜像"取代——用户要求本地/gRPC API 对齐,`insert` 的
高频用法使补全 `writer` 成为必要。

## 后果

- `LocalClient` 具备 `insert`/`writer`/`trajectory_writer`/`structured_writer` 全套,
  与 gRPC `Client` 签名一致,代码可直接迁移。
- `_default_emit_timesteps` 两端统一为 `True`(消解 §3.4)。
- §3.5 缩减为仅"无 pickle"(`__reduce__`),这是 `LocalClient` 持进程内指针的真实
  物理约束,非 API 债。
- 代价:`Writer` 本地化引入 gRPC `Writer` 没有的跨线程 callback 确认机制(同步模型下
  `InsertCallback` 递减 `num_items_in_flight_`,`Close` drain 保安全),详见
  `docs/unbind-local-writer-plan.md` D3。
- `StreamingTrajectoryWriter` 仍不在内嵌范围(A1 该部分保留,本决策只推翻 Writer 部分)。

## 状态

accepted(随 `docs/unbind-local-writer-plan.md` 实施落地)

supersedes: 设计文档 §1.3 决策 A1 中"旧 `Writer` 本地路径砍掉"的部分(A1 关于
`StreamingTrajectoryWriter` 的部分不变)。
