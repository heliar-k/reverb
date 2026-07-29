# AGENTS.md

本文件给 coding agent 提供本仓库的项目规范。只列已验证的事实；细节以链接到的文档为准。

## 项目概况

DeepMind Reverb 的 fork：纯 numpy 内嵌库（无 TensorFlow 依赖），另支持同机跨进程 SHM 传输与可选的 torch.Tensor。包名 `dm-reverb-numpy`，Python ≥ 3.9，C++ 部分经 pybind11 暴露。

- 为什么去掉 TF、架构差异：`docs/design/numpy-embed-design.md`
- 使用者向概念与客户端选型：`docs/guide/concepts.md`、`docs/guide/client-transports.md`

## 构建与测试

- 构建系统：**Bazel**（`MODULE.bazel` + `WORKSPACE`，各目录 `BUILD` 文件）。测试用 `bazel test //reverb/...`，不要直接跑 pytest。
- 覆盖率：`scripts/coverage.sh`（插桩构建 + llvm-profdata，注意必须 `--cache_test_results=no`）。
- 新 Python 模块要在对应 `BUILD` 文件里登记，否则 bazel 测试找不到。

## 代码风格

- Lint/format：**pre-commit**（ruff，配置在 `ruff.toml`）。提交前跑 `pre-commit run --all-files`。
- `target-version = py39`：**禁止 PEP 604 写法**（`X | None`），用 `Optional[X]`。这是 ruff 故意不开 UP 规则的原因。
- `__init__.py` 的 re-export、`test_*.py` 的 unused import 等已在 per-file-ignores 白名单，不要为消警告改这些文件的结构。

## 依赖纪律

- torch 是**可选依赖**（`[torch]` extra）：reverb 本体任何模块不得顶层 `import torch`，一律走 `reverb/torch_support.py` 的懒导入。
- 新增第三方依赖前先确认现有依赖（见 `pyproject.toml`）做不到。

## 文档

- 目录结构与归档规则见 `docs/README.md`；根目录只放 README，其余按类型进 `guide/ design/ spec/ research/ benchmark/ adr/ ticket/` 子目录（pre-commit 强制）。
- 文档互链用相对路径；搬动文档时同步修正所有引用并验证无死链。
- ADR 不可变：推翻旧决策就新开一份 ADR，在原 ADR 标注 superseded。

## Ticket 工作流

- 活跃 ticket 直接在 `docs/ticket/` 开新文件；结案后在 `docs/ticket/README.md` 索引表追加一行（日期、主题、一句话总结）。
