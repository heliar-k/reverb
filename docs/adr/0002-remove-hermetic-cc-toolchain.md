# 移除 hermetic C++ 工具链,改用系统 gcc

## 背景

Reverb fork 在去 TensorFlow 重构(commit `2ea3014`)时,从 `@org_tensorflow` 继承了
`rules_ml_toolchain`——一套 hermetic C++ 工具链,提供 7.1G 的 llvm18 下载 + glibc 2.27
sysroot(323M)。其 cc_toolchain BUILD 引用 `@cuda_redist_json`,迫使 CUDA/NCCL/NVSHMEM
初始化链也必须保留在 WORKSPACE 里。Reverb 是纯 CPU C++,完全不用 GPU,这套链是纯粹的
TF 遗产死重量。

`.bazelrc` 默认开启 hermetic cc:

```
common --incompatible_enable_cc_toolchain_resolution
common --@rules_ml_toolchain//common:enable_hermetic_cc=True
common --repo_env USE_HERMETIC_CC_TOOLCHAIN=1
```

## 触发决策的事实

- **CUDA 链无根**:Reverb 源码 + BUILD 零 CUDA 引用。CUDA/NCCL/NVSHMEM 链存在的唯一
  原因是 `rules_ml_toolchain` 的工具链 BUILD 文件引用 `@cuda_redist_json`,分析期
  missing-repo 会报错,故防御性保留。移除 `rules_ml_toolchain` 即连根拔起。
- **编译时间瓶颈不在工具链**:1921 个 `.o` 里仅 71 个是 reverb 自身,1850 个是
  grpc/boringssl/upb/protobuf/absl/re2 等依赖。工具链换不换都要编译这 1850 个,
  这是当初评估"是否迁 CMake"时的关键数据——结论是不迁。

## 决策

**移除 `rules_ml_toolchain` + CUDA/NCCL/NVSHMEM 链,改用 Bazel 内置 `@local_config_cc`
(系统 gcc)**。

- `.bazelrc`:`--incompatible_enable_cc_toolchain_resolution` →
  `--noincompatible_enable_cc_toolchain_resolution`(回退 autoconf 的 local_config_cc);
  删 `@rules_ml_toolchain`/`USE_HERMETIC_CC_TOOLCHAIN` flags;`clang_local` config
  清理死引用。
- `WORKSPACE`:删 `rules_ml_toolchain` http_archive + `cc_toolchain_deps()` +
  `register_toolchains` + cuda/nccl/nvshmem 全部初始化(-96 行)。
- BUILD/`build_rules.bzl`:`external/sysroot.../libc_nonshared.a` 改系统路径
  `/usr/lib/x86_64-linux-gnu/libc_nonshared.a`(glibc 2.35 的 Scrt1.o 已不再引用
  `__libc_csu_init`,系统 libc_nonshared.a 可用)。
- `scripts/coverage.sh`:hermetic llvm18 检测(bazel external cache)改系统 llvm 检测。

## 触发的修复(切换暴露的 latent bug)

切换到系统 gcc 暴露了三个被 hermetic clang 掩盖的问题:

1. **`--linkopt="-lrt -lm"` 链接失败**:`.bazelrc` 把 `-lrt -lm` 作为单个引号参数传入。
   hermetic lld 按空格拆分;系统 `ld.gold` 当成单个库名 → `cannot find -lrt -lm`。
   修复:拆成 `--linkopt=-lrt --linkopt=-lm` 两个独立 flag。

2. **`tensor_proxy_test` 等链接丢失符号**:`//reverb:libreverb` 是 `cc_shared_library`,
   `:tensor_proxy` 在其 `exports_filter` 内,故 cc_test 依赖 `:tensor_proxy` 时走动态
   库路径。lld 正确解析共享库符号;gold 在 `--gc-sections` 下漏拉 `tensor_proxy.o`
   → `undefined reference`。修复:`.bazelrc` 加 `test --dynamic_mode=off` 强制 cc_test
   静态链依赖。`libreverb.so`/`libpybind.so` 仍按 cc_shared_library 正常产出。

3. **`shm_insert_test` 稳定挂起(根因 bug)**:`ShmServer::HandleInsert` 中
   `chunks.try_emplace(cd.chunk_key(), make_shared<Chunk>(std::move(cd)))` 的实参求值
   顺序 unspecified。gcc 先求值 `std::move(cd)` 移走 `cd`,再求值 `cd.chunk_key()`
   读到 moved-from 默认值 0,map 以 0 为键插入,随后 `items.flat_trajectory()` 的真实
   chunk_key 查不到 → "references unknown chunk" + client 永久 spin(75s 超时)。clang
   恰好先求值 key 故从不触发。修复:先取 `uint64_t map_key = cd.chunk_key()` 再
   `try_emplace(map_key, std::move(cd))`。见 commit `cf8af29`。

## 编译速度三向对比

冷构建 `//reverb:libreverb //reverb:pybind`(`bazel clean --expunge` 后):

| 工具链 | 链接器 | 耗时 | actions | 测试 | 额外规避 |
|---|---|---|---|---|---|
| hermetic llvm18 + CUDA chain(baseline) | lld(hermetic) | 637s | 4255 | ✅ | 无 |
| 系统 gcc 11.4(本决策) | gold | 658s | 3359 | ✅ | `test --dynamic_mode=off` |
| 系统 clang 22(备选,未采用) | lld | 721s | 2622 | ✅ | `-Wno-private-header` |

### 速度结论

**换工具链不提升编译速度**——gcc(658s)与 baseline(637s)基本持平(+3.3%,测量噪声内),
clang(721s)反而更慢。这印证了"编译时间由依赖编译主导(2613 个编译 action 绝大多数是
grpc/boringssl/upb/protobuf/absl 的 .o),不是工具链开销"的判断。要真正提速得从依赖
入手(预编译 grpc/protobuf 为系统库、ccache、减少静态链接),而非换工具链。

### 为何选 gcc 而非 clang

- gcc 最快(658s);clang 22(2026 发布)对 grpc 旧代码过严,需持续维护 `-Wno-*`。
- gcc 的 gold bug 已用 `--dynamic_mode=off` 固化规避,无额外负担。
- clang+lld 的优势(lld 无 gold bug)不抵速度劣势 + 警告维护成本。

## 收益

| 项 | 数值 |
|---|---|
| 省掉 hermetic llvm18 下载 | ~7.1GB |
| 省掉 glibc 2.27 sysroot | ~323MB |
| 少拉无用 external repo | 74 个(274→200) |
| 删死代码 | rules_ml_toolchain + CUDA 链(WORKSPACE -96 行) |
| 修复 latent bug | `try_emplace` 求值顺序(clang 掩盖,gcc 暴露) |

## 后果

- 构建产物 `libreverb.so` + `libpybind.so` 正常产出。
- 完整 cc_test sweep 43/43 通过(串行)。
- 环境前提:构建机需有系统 gcc(本机 gcc 11.4 / glibc 2.35 / Ubuntu 22.04)。
  CI 若用不同 glibc,需复核 `libc_nonshared.a` 路径。

## 状态

accepted(commits `cf8af29` + `c644c15`,分支 `remove-hermetic-cc`)

supersedes: `.bazelrc`/`WORKSPACE` 中 `rules_ml_toolchain` hermetic cc + CUDA 链的
防御性保留(去 TF 重构 `2ea3014` 引入)。
