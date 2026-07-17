workspace(name = "reverb")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# ponytail: bzlmod Step 3-4. 删除 @org_tensorflow(453MB)+ tf_workspace0-3 +
# @local_xla,原生 http_archive 注册 reverb 闭包实际需要的 C++ 依赖。
# ponytail: @rules_ml_toolchain(hermetic C++ 工具链 + CUDA/NCCL/NVSHMEM 链)已移除
# ——reverb 纯 CPU C++,改用 Bazel 内置 @local_config_cc(系统 gcc/clang)。
# 见 .bazelrc 的 noincompatible_enable_cc_toolchain_resolution。

http_archive(
    name = "rules_shell",
    sha256 = "bc61ef94facc78e20a645726f64756e5e285a045037c7a61f65af2941f4c25e1",
    strip_prefix = "rules_shell-0.4.1",
    url = "https://github.com/bazelbuild/rules_shell/releases/download/v0.4.1/rules_shell-v0.4.1.tar.gz",
)

# 基础设施仓库(rules_ml_toolchain 的 cc_toolchain_deps / grpc_deps 会引用)。
http_archive(
    name = "bazel_skylib",
    sha256 = "bc283cdfcd526a52c3201279cda4bc298652efa898b10b4db0837dc51652756f",
    urls = [
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
    ],
)

http_archive(
    name = "platforms",
    sha256 = "29742e87275809b5e598dc2f04d86960cc7a55b3067d97221c9abbc9926bff0f",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.11/platforms-0.0.11.tar.gz",
        "https://github.com/bazelbuild/platforms/releases/download/0.0.11/platforms-0.0.11.tar.gz",
    ],
)

http_archive(
    name = "rules_license",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_license/releases/download/0.0.7/rules_license-0.0.7.tar.gz",
        "https://github.com/bazelbuild/rules_license/releases/download/0.0.7/rules_license-0.0.7.tar.gz",
    ],
    sha256 = "4531deccb913639c30e5c7512a054d5d875698daeb75d8cf90f284375fe7c360",
)

http_archive(
    name = "rules_pkg",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.7.1/rules_pkg-0.7.1.tar.gz",
        "https://github.com/bazelbuild/rules_pkg/releases/download/0.7.1/rules_pkg-0.7.1.tar.gz",
    ],
    sha256 = "451e08a4d78988c06fa3f9306ec813b836b1d076d0f055595444ba4ff22b867f",
)

http_archive(
    name = "bazel_features",
    sha256 = "4fd9922d464686820ffd8fcefa28ccffa147f7cdc6b6ac0d8b07fde565c65d66",
    strip_prefix = "bazel_features-1.25.0",
    urls = [
        "https://mirror.bazel.build/github.com/bazel-contrib/bazel_features/releases/download/v1.25.0/bazel_features-v1.25.0.tar.gz",
        "https://github.com/bazel-contrib/bazel_features/releases/download/v1.25.0/bazel_features-v1.25.0.tar.gz",
    ],
)

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
    ],
)

# Hermetic Python: rules_cc / com_google_protobuf / rules_python + @python_version_repo.
# ponytail: 替代 @local_xla 的 python_init_rules/repositories/toolchains。
load("//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@rules_shell//shell:repositories.bzl", "rules_shell_dependencies", "rules_shell_toolchains")

rules_shell_dependencies()

rules_shell_toolchains()

load("//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    default_python_version = "3.11",
    requirements = {
        "3.9": "//reverb/pip_package:requirements_lock_3_9.txt",
        "3.10": "//reverb/pip_package:requirements_lock_3_10.txt",
        "3.11": "//reverb/pip_package:requirements_lock_3_11.txt",
        "3.12": "//reverb/pip_package:requirements_lock_3_12.txt",
        "3.13": "//reverb/pip_package:requirements_lock_3_13.txt",
    },
)

load("@python_version_repo//:py_version.bzl", "HERMETIC_PYTHON_VERSION", "REQUIREMENTS")
load("//third_party/py:python_init_repositories.bzl", "get_toolchain_name_per_python_version", "python_init_toolchains")

python_init_toolchains(hermetic_python_version = HERMETIC_PYTHON_VERSION)

load("@rules_python//python:pip.bzl", "package_annotation", "pip_parse")

# ponytail: pip_parse 注册 @pypi(numpy 等)。numpy 注解暴露 cc_library
# 头文件供 pybind 链接;TF 注解已随去 TF 重构移除。
numpy_annotation = package_annotation(
    additive_build_content = """\
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
name = "numpy_headers_2",
hdrs = glob(["site-packages/numpy/_core/include/**/*.h"]),
strip_include_prefix="site-packages/numpy/_core/include/",
)
cc_library(
name = "numpy_headers_1",
hdrs = glob(["site-packages/numpy/core/include/**/*.h"]),
strip_include_prefix="site-packages/numpy/core/include/",
)
cc_library(
name = "numpy_headers",
deps = [":numpy_headers_2", ":numpy_headers_1"],
)
""",
)

pip_parse(
    name = "pypi",
    annotations = {
        "numpy": numpy_annotation,
    },
    extra_hub_aliases = {
        "numpy": ["numpy_headers"],
    },
    python_interpreter_target = "@{}_host//:python".format(
        get_toolchain_name_per_python_version("python", HERMETIC_PYTHON_VERSION),
    ),
    requirements_lock = REQUIREMENTS,
)

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

# End hermetic Python initialization

# reverb 闭包需要的 C++ 依赖:原生 http_archive 注册(去 @local_xla build_file)。
# 顺序:先注册被依赖的(absl/re2/zlib/snappy/googletest),再注册 grpc(其 grpc_deps
# 用 existing_rules() 保护,已注册的会跳过)。

# com_google_absl:由 grpc_deps() 注册(version 20250512.1)。先占位让 grpc_deps 跳过
# 它会丢失版本,故不预注册,交给 grpc_deps()。

# com_google_absl:ponytail 用 @local_xla vendored 的版本(LTS 20250814.0 +
# patches + com_google_absl.BUILD),与原 tf_workspace 一致。grpc_deps() 的
# existing_rules() 保护会跳过它注册的旧版 absl(20250512.1)。
http_archive(
    name = "com_google_absl",
    sha256 = "f56086f4cdb0ab9b7c3ac46831b1faba3753248d0f06f8bca4c917a1de2a560a",
    strip_prefix = "abseil-cpp-987c57f325f7fa8472fa84e1f885f7534d391b0d",
    build_file = "//third_party/absl:com_google_absl.BUILD",
    patches = [
        "//third_party/absl:btree.patch",
        "//third_party/absl:build_dll.patch",
        "//third_party/absl:endian.patch",
        "//third_party/absl:rules_cc.patch",
        "//third_party/absl:check_op.patch",
        "//third_party/absl:check_op_2.patch",
    ],
    patch_args = ["-p1"],
    urls = [
        "https://github.com/abseil/abseil-cpp/archive/987c57f325f7fa8472fa84e1f885f7534d391b0d.tar.gz",
    ],
    repo_mapping = {
        "@google_benchmark": "@com_google_benchmark",
        "@googletest": "@com_google_googletest",
    },
)

http_archive(
    name = "zlib",
    build_file = "//third_party:zlib.BUILD",
    sha256 = "9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23",
    strip_prefix = "zlib-1.3.1",
    urls = [
        "https://zlib.net/fossils/zlib-1.3.1.tar.gz",
    ],
)

http_archive(
    name = "snappy",
    build_file = "//third_party:snappy.BUILD",
    sha256 = "7ee7540b23ae04df961af24309a55484e7016106e979f83323536a1322cedf1b",
    strip_prefix = "snappy-1.2.0",
    urls = [
        "https://github.com/google/snappy/archive/1.2.0.zip",
    ],
)

http_archive(
    name = "com_googlesource_code_re2",
    sha256 = "ef516fb84824a597c4d5d0d6d330daedb18363b5a99eda87d027e6bdd9cba299",
    strip_prefix = "re2-03da4fc0857c285e3a26782f6bc8931c4c950df4",
    urls = [
        "https://github.com/google/re2/archive/03da4fc0857c285e3a26782f6bc8931c4c950df4.tar.gz",
    ],
)

http_archive(
    name = "com_google_googletest",
    sha256 = "f253ca1a07262f8efde8328e4b2c68979e40ddfcfc001f70d1d5f612c7de2974",
    strip_prefix = "googletest-28e9d1f26771c6517c3b4be10254887673c94018",
    patches = ["//third_party/patches:googletest.patch"],
    patch_args = ["-p1"],
    urls = [
        "https://github.com/google/googletest/archive/28e9d1f26771c6517c3b4be10254887673c94018.zip",
    ],
)

# pybind11:reverb 自有 BUILD(去 @local_xla,改用 rules_python current_py_cc_headers)。
http_archive(
    name = "pybind11",
    urls = ["https://github.com/pybind/pybind11/archive/v2.13.6.tar.gz"],
    sha256 = "e08cb87f4773da97fa7b5f035de8763abc656d87d5773e62f6da0587d1f0ec20",
    strip_prefix = "pybind11-2.13.6",
    build_file = "//third_party:pybind11.BUILD",
)

# grpc:带 @local_xla vendored 的 grpc.patch(自包含,无 @local_xla 引用)。
http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "dd6a2fa311ba8441bbefd2764c55b99136ff10f7ea42954be96006a2723d33fc",
    strip_prefix = "grpc-1.74.0",
    patches = ["//third_party/patches:grpc.patch"],
    patch_args = ["-p1"],
    urls = [
        "https://github.com/grpc/grpc/archive/refs/tags/v1.74.0.tar.gz",
    ],
)

# grpc_deps():注册传递依赖(absl/boringssl/cares/envoy_api/xds/protoc_gen_validate/
# openssl 等),用 existing_rules() 保护已注册的(com_google_protobuf/zlib/
# com_google_googletest/bazel_skylib/platforms/rules_cc 等)。
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

# grpc_extra_deps():调 protobuf_deps()(注册 @proto_bazel_features 等)+
# rules_proto_dependencies/api_dependencies/googletest_deps 等。grpc 1.74 的
# BUILD 依赖这些被 grpc_deps 注册的仓库的 extras。
load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()

# ponytail: @rules_ml_toolchain hermetic C++ 工具链 + CUDA/NCCL/NVSHMEM 初始化链
# 已移除。reverb 纯 CPU C++,改用 Bazel 内置 @local_config_cc(系统 gcc/clang)。
# 原 table_test/sampler_test linkopts 引用的 external/sysroot_linux_x86_64_glibc_2_27
# libc_nonshared.a 已改为系统路径(见 reverb/cc/BUILD)。
