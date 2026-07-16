# 回退到预编译 Python (python-build-standalone) 方案

> 状态：计划，待执行
> 上一 session 产出。当前分支 `feat/numpy-embed`，HEAD `76ab36f`。
> 工作区有未提交改动（见下「当前工作区状态」），执行前先处理。

## 背景：为什么回退

Session 2（commit `558daa7`）把 bazel 的 hermetic standalone python 换成了
`local_runtime_repo` 指向 uv venv/系统 python，理由是「standalone 3.11 与 cc_test
硬编码 libpython3.10 混搭」。

**这个理由现在不成立了**：cc_test 后来已独立解耦（走
`@rules_python//python/cc:current_py_cc_libs`，自动跟随任何 toolchain 的
libpython）。混搭根因已除，standalone 方案的缺点（下载 142M）相对其优点变得不值得。

`local_runtime_repo` 方案引出一连串问题：

1. 必须从系统/venv 找一个 python → 「哪个 python」「PATH 怎么设」
2. 系统 python 的 `/usr/local/dist-packages` 旧 protobuf 3.20.3 污染 bazel sandbox，
   `@pypi` 的 7.35.1 挡不住（py_test 运行时 ImportError `runtime_version`）
3. venv 要手动 `source .venv/bin/activate` 才能隔离
4. `_build_backend.py` 要注入 `sys.executable` 的 PATH 才能让源码安装版本对齐
5. 换机器/换 venv 要改 WORKSPACE 硬编码路径

standalone 方案全部消除这些问题：bazel 自己下 python，与系统无关，site-packages
天然隔离，版本 hermetic 锁死。

## 决策

- **dev / bazel 源码构建**：恢复 `python_register_toolchains`，bazel 自动下载
  python-build-standalone。零配置，不碰系统 python。
- **pip 源码安装 (`pip install .`)**：`_build_backend.py` 传
  `--repo_env=HERMETIC_PYTHON_VERSION=<用户python版本>`，bazel 下载对应版本
  standalone python。.so 链接的 libpython 与用户安装目标 python 版本一致。
- **保留**：`protobuf>=5.27` 依赖约束（已修，独立于本回退）。
- **保留**：cc_test 的 `current_py_cc_libs` 解耦（不回退）。
- **代价**：bazel 首次下载 ~142M standalone python（一次性，进 bazel cache）。

## 当前工作区状态（执行前必读）

本 session 已把工作区清理到干净的回退起点。当前 `git status`：

- `_build_backend.py`（M）：`protobuf>=5.27` 约束 — **保留**（已删 PATH 注入）
- `pyproject.toml`（M）：`protobuf>=5.27` — **保留**
- `configure.py`（D，staged）：已删 — **保留**（T2 成果，独立于本回退）
- `ruff.toml`（M）：configure.py 的 lint 例外清理 — **保留**（T2 成果）
- `docs/plan-revert-to-standalone-python.md`（??）：本计划文件
- `.bazelrc` / `WORKSPACE`：**已 reset 到 HEAD `76ab36f`**（即 venv 硬编码
  版本：`REVERB_PYTHON_INTERPRETER = "/home/.../.venv/bin/python"` +
  `local_runtime_repo`）。Step 2-3 会把它们改成 standalone。
- `uv.lock`：**已删除**。

## 执行步骤

### Step 0：确认工作区起点

`git status` 应只含上面列的 5 项。`WORKSPACE` 应仍是 venv 硬编码版本
（`grep REVERB_PYTHON_INTERPRETER WORKSPACE` 显示绝对路径）。若不符,参考
背景节清理。

### Step 2：恢复 standalone python toolchain

**`third_party/py/python_init_repositories.bzl`**：回退到 `558daa7` 之前的版本。
即把 `python_init_toolchains` 从 `local_runtime_repo` 改回
`python_register_toolchains`，恢复 `get_toolchain_name_per_python_version`。

参考 `git show 558daa7~1:third_party/py/python_init_repositories.bzl` 取回原貌。
关键点：

- `load("@rules_python//python:repositories.bzl", "py_repositories", "python_register_toolchains")`
- `get_toolchain_name_per_python_version(name, version)` 返回 `"{name}_{version}"`
  （version 的 `.` 换 `_`）
- `python_init_toolchains(name, hermetic_python_version)` 调
  `python_register_toolchains(name=get_toolchain_name_per_python_version(...),
  ignore_root_user_error=True, python_version=hermetic_python_version)`

实际执行：`git checkout 558daa7~1 -- WORKSPACE third_party/py/python_init_repositories.bzl`
精确还原两个文件（558daa7 是唯一触碰它们的 commit）。

### Step 3：WORKSPACE 改回 standalone

（由 Step 2 的 checkout 一并完成,内容与下述一致。）

**WORKSPACE**：

- `python_init_repositories` 的 `default_python_version`：改回 `"3.11"`
  （standalone 默认版本；保持 `558daa7` 之前的值）。
  - 注意：requirements_lock 3.11 的 protobuf 是 6.33.0，满足 `>=5.27`。
- `load` 行恢复 `get_toolchain_name_per_python_version`：
  `load("//third_party/py:python_init_repositories.bzl", "get_toolchain_name_per_python_version", "python_init_toolchains")`
- 删掉 `REVERB_PYTHON_INTERPRETER` 常量及其所有 ponytail 注释。
- `python_init_toolchains(hermetic_python_version = HERMETIC_PYTHON_VERSION)`
  （不传 interpreter_path，standalone 不需要）。
- `pip_parse` 的 `python_interpreter` 改回 `python_interpreter_target`：

  ```python
  python_interpreter_target = "@{}_host//:python".format(
      get_toolchain_name_per_python_version("python", HERMETIC_PYTHON_VERSION),
  ),
  ```

  （`python_register_toolchains` 自动注册 `name + "_host"` 解释器，供 pip_parse 用）

**`.bazelrc`**：删掉 Step 1 遗留的 unified PATH 注释（已在 Step 1 checkout 清掉）。

### Step 4：_build_backend.py 版本对齐（确认现有逻辑即可）

`_build_backend.py` 的 `_run_bazel` 已经传
`--repo_env=HERMETIC_PYTHON_VERSION={py_ver}`（`py_ver = _target_py_version()`
= 用户 python 版本）。standalone 方案下这个 env 会：

1. 选 `@python_version_repo` 生成的 `py_version.bzl` 版本号
2. `python_register_toolchains` 按此版本下载对应 standalone python

**所以源码安装的版本对齐自动成立**，无需额外改动。只需确认 `_run_bazel` 里没有
Step 1 删剩的 PATH 注入残留。

### Step 5：验证

1. `bazel test //reverb/tests:all --test_output=errors` → 19/19 PASSED
   - standalone python 天然隔离系统 site-packages，protobuf 污染问题应消失
2. `bazel test //reverb/cc/...` → cc_test 全绿(**除预存 `trajectory_writer_grpc_test`
   的 `RetriesOnTransientError`,该失败在 venv baseline 同样存在,gRPC 流重试语义
   问题,与本回退无关**)
3. `bazel build //reverb:libreverb //reverb:libpybind.so` → 构建成功
   (注:原计划写的 `//reverb:libreverb.so //reverb:libpybind.so` 目标名有误——
   `libreverb` 是规则名无 `.so` 后缀,`libpybind.so` 是 cc_binary 目标名。)
4. 确认 protobuf 约束生效：
   `grep protobuf pyproject.toml _build_backend.py` 都有 `>=5.27`
5. （可选）源码安装端到端：
   `python3.10 -m pip install . --force-reinstall` 后 `python3.10 -c "import reverb; print('ok')"`
   - **已执行并通过**(2026-07-16):干净 venv(python3.10)跑 `pip install .
     --force-reinstall --no-build-isolation` 成功构建 wheel
     `dm_reverb-0.15.0-cp310-cp310-linux_x86_64.whl`(8.2MB)。pip 解析运行时依赖
     numpy 2.2.6(cp310 兼容)/protobuf 7.35.1(满足 `>=5.27`)。`import reverb`
     成功,`reverb.InProcessClient` 从编译出的 `reverb.libpybind` 模块(.so)加载,
     证明 libpybind.so 正确链接对应版本 libpython3.10 且可加载。wheel METADATA
     含 `Requires-Dist: protobuf >= 5.27`。bazel 由 `_build_backend.py` 传
     `--repo_env=HERMETIC_PYTHON_VERSION=3.10` 下载 standalone 3.10 编译 .so,
     版本对齐自动成立。

## 计划遗漏的回归与修复（执行中发现）

计划假设「cc_test 自动跟随 standalone toolchain」**只对链接成立,不对运行时成立**。
rules_python 的 `current_py_cc_libs` 故意只管链接不管运行时([官方 howto](https://rules-python.readthedocs.io/en/latest/howto/linking-libpython.html)

- [discussion #3153](https://github.com/bazel-contrib/rules_python/discussions/3153))。
py_test 能跑是因为 rules_python 给它套了 bootstrap stub;裸 cc_test 没这个 stub。

**回归**:`//reverb/cc/support:tensor_proxy_test`(用 `py::scoped_interpreter`
内嵌 libpython + `import numpy`)在 standalone 下运行时找不到 stdlib+numpy：

- python-build-standalone 的 libpython 编译期 `base_prefix='/install'`,stdlib实际在
  bazel cache 解压的 standalone 树 → `failed to get the Python codec of the filesystem encoding`
- `@pypi` numpy 的 site-packages 不在 cc_test runfiles → `No module named 'numpy'`

venv 方案恰好无此问题:venv python 的 `base_prefix=/usr` 真实,且 venv site-packages
自动发现 numpy。**这是 558daa7 切 venv 的隐性收益,计划遗漏了。**

**修复**(已做):`reverb/cc/platform/default/build_rules.bzl` 新增
`reverb_embed_py_test` wrapper(模式取自 [pybind11_bazel commit 9d8c6b4](https://github.com/pybind/pybind11_bazel/commit/9d8c6b4))：

- 分析期从 py3 toolchain 取 `interpreter` 路径 + 从 `@pypi//numpy:numpy` 取 site-packages
- 生成 shell 脚本设 `PYTHONHOME`(= standalone 树根)+ `PYTHONPATH`(= numpy site-packages)
- 把 standalone 树(`py3_runtime.files`)+ numpy 拉进 runfiles,`exec` 真正的 cc_binary
- `tensor_proxy_test` 改为 `cc_binary` + `reverb_embed_py_test` 包裹(其余 cc_test 不动,
  它们传递链接 libpython 但不 Py_Initialize,无需 wrapper)

代价:仅 `tensor_proxy_test` 一个目标多一层 wrapper(~100 行 starlark,复用)。非 Windows
用 shell 脚本;Windows 直链 binary(reverb 无 Windows 内嵌 cc_test 目标)。

## 注意事项

- **cc_test 不要动**:`reverb_cc_test` 宏 + `current_py_cc_libs` 解耦保留,它自动
  跟随 standalone toolchain 的 libpython(链接层面)。运行时 stdlib/site-packages 由
  `reverb_embed_py_test` wrapper 补(仅给真正 Py_Initialize 的 tensor_proxy_test)。
- **requirements_lock 文件不动**:5 个版本的 lock 都在,protobuf 都满足 `>=5.27`。
- **default_python_version = "3.11"**:保持 `558daa7` 之前的值。dev 默认用 3.11
  standalone。源码安装时 `_build_backend.py` 按用户 python 版本覆盖。
- **uv.lock 删除**:standalone 方案下 uv 只用于多版本矩阵测试(T5,bazel 外),
  不需要项目级 uv.lock。
- **.venv 目录**:可保留(多版本矩阵 T5 会用),但 bazel 不再依赖它。
- **transport_parity_test 并发挂起**:若同时跑 `//reverb/cc/... //reverb/tests:all`,
  `transport_parity_test`(SHM/进程内 server)可能与并发 cc/shm 测试争资源而 hang。
  单独/按计划分开跑 `//reverb/tests:all` 则 1.4s 通过,非回退回归。

## 验收标准

- [x] `bazel test //reverb/tests:all` 19/19 PASSED（无 protobuf ImportError）
- [x] `bazel test //reverb/cc/...` 42/43 绿（唯一失败 `trajectory_writer_grpc_test`
      为预存 gRPC bug,venv baseline 同样失败,与本回退无关）
- [x] WORKSPACE 无 `REVERB_PYTHON_INTERPRETER` 硬编码绝对路径
- [x] `python_init_repositories.bzl` 用 `python_register_toolchains`
- [x] `pip_parse` 用 `python_interpreter_target`（host 解释器）
- [x] `protobuf>=5.27` 约束保留在 pyproject.toml + _build_backend.py
- [x] `configure.py` 仍删除（T2 成果不丢）
- [x] 无 `uv.lock`
- [x] `tensor_proxy_test` 经 `reverb_embed_py_test` wrapper 在 standalone 下通过
      （计划遗漏的运行时 stdlib+numpy 回归,见上「计划遗漏的回归与修复」）
