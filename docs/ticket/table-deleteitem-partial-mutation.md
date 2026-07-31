# Tickets: Table::DeleteItem 错误路径部分修改（2026-07-31）

> 来源：`feat/numpy-embed` 并发专项 adversarial review，LOW #6（1/2 票存活；弃票方认为
> 失败路径在正常 insert 路径下不可达 —— 所有 insert 先递增 `episode_refs_`）。定级 LOW：
> 不变式目前靠偶然维持而非构造保证，作为 defense-in-depth 修复。

## DeleteItem 逐 chunk 修改 episode_refs_，中途失败留下部分修改

**What to build**：先校验后修改（两阶段），或文档化依赖的调用前提。

**现状**：

- table.cc:815-843 `DeleteItem`：对 item 的每个 chunk 依次
  `episode_refs_` 递减 → 归零则 erase + `num_deleted_episodes_++`；
  中途遇到缺失 episode_id 时 `return FailedPreconditionError` —— 已递减/已 erase 的
  部分不回滚，item 仍在 `data_` 中，引用计数与数据不一致。
- 触发条件：chunk 的 `episode_id` 不在 `episode_refs_` 中。正常 insert 路径先递增
  `episode_refs_`（table.cc:562 `InsertOrAssignAsync`），故当前不可达；但该不变式
  无构造性保证（外部构造 `TableItem`/绕过 insert 的路径可破坏）。

**修复选项**：

1. **两阶段**：第一遍全量校验所有 episode_id 存在，第二遍再修改（改动小，DeleteItem
   本身已是异常路径，性能不敏感）。
2. 或回滚：出错时恢复已修改的引用（代码更绕，不推荐）。
3. 或 `REVERB_CHECK`/文档化前提「调用方保证引用一致」并保持现状。

**Blocked by**：None。

- [ ] `DeleteItem` 先校验全部 episode_id 再修改
- [ ] 单元测试：构造引用不一致的 item（绕过正常 insert），DeleteItem 返回错误且
      `episode_refs_` 与 `data_` 完全未变
