load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_cc_library",
    "reverb_cc_test",
    "reverb_grpc_deps",
    "reverb_tf_deps",
)
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
        "//reverb/cc/platform:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

reverb_cc_library(
    name = "cleanup",
    hdrs = ["cleanup.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
    ],
)

reverb_cc_test(
    name = "cleanup_test",
    srcs = ["cleanup_test.cc"],
    deps = [
        ":cleanup",
    ],
)

reverb_cc_library(
    name = "intrusive_heap",
    srcs = ["intrusive_heap.cc"],
    hdrs = ["intrusive_heap.h"],
    deps = [
        "//reverb/cc/platform:logging",
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
        "//reverb/cc/platform:logging",
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
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
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
        "//reverb/cc/platform:status_matchers",
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
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
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
    # 升级路径: 若 :tensor_proxy 拆出无 Python 的 data_type 子库,可去掉此 linkopts。
    linkopts = [
        "-L/usr/lib/x86_64-linux-gnu",
        "-lpython3.10",
        "/usr/lib/x86_64-linux-gnu/libc_nonshared.a",
    ],
    deps = [
        ":signature",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
        "//third_party/reverb_tensor:reverb_tensor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
    ],
)

reverb_cc_library(
    name = "state_statistics",
    hdrs = [
        "state_statistics.h",
    ],
    deps = [
        "//reverb/cc/platform:hash_map",
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
        "//reverb/cc/platform:logging",
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
        "//reverb/cc/platform:hash_set",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
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
    # 需链接 libpython 解析 Py* 符号。
    linkopts = [
        "-L/usr/lib/x86_64-linux-gnu",
        "-lpython3.10",
        "/usr/lib/x86_64-linux-gnu/libc_nonshared.a",
    ],
    deps = [
        ":trajectory_util",
        ":tensor_proxy",
        "//reverb/cc:chunk_store",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:tensor_compression",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
        "//third_party/reverb_tensor:reverb_tensor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
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

cc_test(
    name = "tensor_proxy_test",
    srcs = ["tensor_proxy_test.cc"],
    copts = ["-Wno-sign-compare"],
    # 嵌入解释器需链接 libpython;系统 Python 3.10(numpy 1.23.5 基于此构建)
    # ponytail: 去 hermetic sysroot 后用系统 libc_nonshared.a(glibc 2.35 的
    # Scrt1.o 已不再引用 __libc_csu_init,但保留 libc_nonshared.a 供 atexit/
    # pthread_atfork 等非共享符号)。
    linkopts = [
        "-L/usr/lib/x86_64-linux-gnu",
        "-lpython3.10",
        "/usr/lib/x86_64-linux-gnu/libc_nonshared.a",
    ],
    deps = [
        ":tensor_proxy",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@pybind11",
        "@pypi//numpy:numpy_headers",
    ],
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
