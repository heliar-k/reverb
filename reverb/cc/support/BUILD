load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_cc_library",
    "reverb_cc_test",
    "reverb_embed_py_test",
    "reverb_grpc_deps",
)
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["//third_party/reverb:license"],
    default_visibility = ["//reverb:__subpackages__"],
)

licenses(["notice"])

reverb_cc_library(
    name = "uint128",
    hdrs = ["uint128.h"],
    deps = [
        "//reverb/cc:schema_cc_proto",
        "@com_google_absl//absl/numeric:int128",
    ],
)

reverb_cc_library(
    name = "grpc_util",
    hdrs = ["grpc_util.h"],
    deps = reverb_grpc_deps() + [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

reverb_cc_library(
    name = "queue",
    hdrs = ["queue.h"],
    deps = [
        "//reverb/cc/platform/default:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

reverb_cc_library(
    name = "intrusive_heap",
    srcs = ["intrusive_heap.cc"],
    hdrs = ["intrusive_heap.h"],
    deps = [
        "//reverb/cc/platform/default:logging",
    ],
)

reverb_cc_test(
    name = "intrusive_heap_test",
    srcs = ["intrusive_heap_test.cc"],
    deps = [
        ":intrusive_heap",
        "@com_google_absl//absl/random",
    ],
)

reverb_cc_library(
    name = "periodic_closure",
    srcs = ["periodic_closure.cc"],
    hdrs = ["periodic_closure.h"],
    deps = [
        "//reverb/cc/platform/default:logging",
        "//reverb/cc/platform:thread",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

reverb_cc_test(
    name = "queue_test",
    srcs = ["queue_test.cc"],
    deps = [
        ":queue",
        "//reverb/cc/platform/default:logging",
        "//reverb/cc/platform/default:status_matchers",
        "//reverb/cc/platform:thread",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

reverb_cc_test(
    name = "periodic_closure_test",
    srcs = ["periodic_closure_test.cc"],
    deps = [
        ":periodic_closure",
        "//reverb/cc/platform/default:status_matchers",
        "//reverb/cc/testing:time_testutil",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "signature",
    srcs = ["signature.cc"],
    hdrs = ["signature.h"],
    deps = [
        ":tensor_proxy",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:table",
        "//reverb/cc/platform/default:hash_map",
        "//reverb/cc/platform/default:logging",
        "//reverb/cc/platform/default:status_macros",
        "//third_party/reverb_tensor:reverb_tensor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:optional",
    ],
    visibility = ["//reverb:__subpackages__"],
)

cc_test(
    name = "signature_test",
    srcs = ["signature_test.cc"],
    # ponytail: :signature 依赖 :tensor_proxy(pybind11/numpy),运行时需 libpython。
deps = [
        ":signature",
        "//reverb/cc/platform/default:logging",
        "//reverb/cc/platform/default:status_matchers",
        "//third_party/reverb_tensor:reverb_tensor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

reverb_cc_library(
    name = "state_statistics",
    hdrs = [
        "state_statistics.h",
    ],
    deps = [
        "//reverb/cc/platform/default:hash_map",
        "@com_google_absl//absl/time",
    ],
)

reverb_cc_test(
    name = "state_statistics_test",
    srcs = ["state_statistics_test.cc"],
    deps = [
        ":state_statistics",
        "@com_google_absl//absl/time",
    ],
)

reverb_cc_library(
    name = "task_executor",
    srcs = ["task_executor.cc"],
    hdrs = ["task_executor.h"],
    deps = [
        ":unbounded_queue",
        "//reverb/cc/platform:thread",
        "@com_google_absl//absl/strings",
    ],
)

reverb_cc_library(
    name = "unbounded_queue",
    hdrs = ["unbounded_queue.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

reverb_cc_test(
    name = "unbounded_queue_test",
    srcs = ["unbounded_queue_test.cc"],
    deps = [
        ":unbounded_queue",
        "//reverb/cc/platform/default:logging",
        "//reverb/cc/platform:thread",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "trajectory_util",
    srcs = ["trajectory_util.cc"],
    hdrs = ["trajectory_util.h"],
    copts = ["-Wno-sign-compare"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        ":tensor_proxy",
        "//reverb/cc:chunk_store",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:tensor_compression",
        "//reverb/cc/platform/default:hash_set",
        "//reverb/cc/platform/default:logging",
        "//reverb/cc/platform/default:status_macros",
        "//third_party/reverb_tensor:reverb_tensor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "trajectory_util_test",
    srcs = ["trajectory_util_test.cc"],
    size = "small",
    copts = ["-Wno-sign-compare"],
    # ponytail: :trajectory_util -> :tensor_compression -> :tensor_proxy,
deps = [
        ":trajectory_util",
        ":tensor_proxy",
        "//reverb/cc:chunk_store",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:tensor_compression",
        "//reverb/cc/platform/default:logging",
        "//reverb/cc/platform/default:status_matchers",
        "//third_party/reverb_tensor:reverb_tensor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

reverb_cc_library(
    name = "key_generators",
    hdrs = ["key_generators.h"],
    deps = ["@com_google_absl//absl/random"],
)

# --- TensorBuffer: numpy bytes 载体,原生 cc_library(不注入 TF 依赖) ---
cc_library(
    name = "tensor_proxy",
    srcs = ["tensor_proxy.cc"],
    hdrs = ["tensor_proxy.h"],
    copts = ["-Wno-sign-compare"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@pybind11",
        "@pypi//numpy:numpy_headers",
        "//third_party/reverb_tensor:reverb_tensor_cc_proto",
        "@com_google_protobuf//:protobuf",
    ],
    visibility = ["//visibility:public"],
)

# ponytail: 嵌入解释器(scoped_interpreter + import_array)需链接 libpython;
# 依赖 @rules_python//python/cc:current_py_cc_libs(python-build-standalone)。
# numpy 头取自 @pypi//numpy:numpy_headers。python-build-standalone 下内嵌
# libpython 的 base_prefix='/install',运行时找不到 stdlib+numpy;故构建
# cc_binary 并用 reverb_embed_py_test wrapper 设 PYTHONHOME/PYTHONPATH。
# 模式取自 pybind11_bazel commit 9d8c6b4。仅此测试真正 Py_Initialize。
cc_binary(
    name = "tensor_proxy_test_bin",
    srcs = ["tensor_proxy_test.cc"],
    copts = ["-Wno-sign-compare"],
    testonly = True,
    visibility = ["//visibility:private"],
    deps = [
        ":tensor_proxy",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@pybind11",
        "@pypi//numpy:numpy_headers",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

reverb_embed_py_test(
    name = "tensor_proxy_test",
    binary = ":tensor_proxy_test_bin",
)

# --- length-delimited protobuf IO(替代 TFRecord,checkpoint 序列化基础,D3 决策) ---
# 原生 cc_library/cc_test:不注入 TF 依赖。
cc_library(
    name = "length_delimited_io",
    srcs = ["length_delimited_io.cc"],
    hdrs = ["length_delimited_io.h"],
    copts = ["-Wno-sign-compare"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "length_delimited_io_test",
    srcs = ["length_delimited_io_test.cc"],
    copts = ["-Wno-sign-compare"],
    deps = [
        ":length_delimited_io",
        "//reverb/cc:schema_cc_proto",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
    ],
)

# --- Tier 3 协议层拷贝成本微基准(诊断工具) ---
cc_binary(
    name = "tier3_bench",
    srcs = ["tier3_bench.cc"],
    copts = ["-Wno-sign-compare"],
    deps = [
        ":tensor_proxy",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:tensor_compression",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)
