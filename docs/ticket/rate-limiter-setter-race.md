# Tickets: RateLimiter 公共 setter 数据竞争（潜伏，2026-07-31）

> 来源：`feat/numpy-embed` 并发专项 adversarial review，MEDIUM #4（2/2 票存活）。
> 当前为潜伏问题：setter 未从 Python 绑定（死代码），一旦绑定即实锤竞态。

## RateLimiter 公开 setter 无锁写，表 worker 并发读

**What to build**：给 setter 加锁（或明确声明线程不安全的接口约束），消除「绑定即竞态」。

**现状**：

- rate_limiter.h:89-96：`set_samples_per_insert` / `set_min_size_to_sample` /
  `set_min_diff` / `set_max_diff` 均为裸字段赋值，无 `mu_`、无线程注解。
- 表 worker 线程在 `mu_` 下经 `CanSample` / `CanInsert` / `MaybeCommitSample`
  （rate_limiter.cc）读同一批字段。
- 字段本身非原子（`double` / `int64_t`），写读并发是 C++ 数据竞争（UB），不只是
  「读到旧值」。当前无绑定方（死代码），故为潜伏 —— 评审无法证伪机制本身。

**修复选项**（按推荐序）：

1. **删除这组 setter**（首选）：全仓零调用方（grep 仅命中 proto setter 与 checkpoint 回填，
   rate_limiter.cc:137-140/157-159 写的是 proto 字段），死代码直接删，竞态随之消失。
2. 若必须保留（上游 API 兼容诉求）：setter 内 `absl::MutexLock lock(&mu_)`（若 `mu_` 与表
   worker 同一把锁则零额外成本），或加 `ABSL_GUARDED_BY` 注解 + 文档「仅构造期调用」。

**Blocked by**：None。

- [ ] 确认 setter 是否有绑定计划（pybind/`RateLimiterInfo` 调整路径）
- [ ] 无绑定 → 删除 + 全仓 grep 确认无引用；有绑定 → 加锁 + TSan 并发测试
