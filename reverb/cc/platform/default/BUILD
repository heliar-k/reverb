# Platform-specific code for reverb

load(
    "//reverb/cc/platform/default:build_rules.bzl",
    "reverb_cc_library",
    "reverb_grpc_deps",
    "reverb_tf_deps",
)

package(default_visibility = ["//reverb/cc/platform:__pkg__"])

licenses(["notice"])

exports_files(["LICENSE"])

reverb_cc_library(
    name = "snappy",
    srcs = ["snappy.cc"],
    deps = [
        "//reverb/cc/platform:snappy_hdr",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

reverb_cc_library(
    name = "checkpointer",
    srcs = ["default_checkpointer.cc"],
    deps = [
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/platform:checkpointing_hdr",
        ":simple_checkpointer",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "checkpointing_utils",
    srcs = ["checkpointing_utils.cc"],
    deps = [
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/platform:checkpointing_utils_hdr",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/selectors:heap",
        "//reverb/cc/selectors:interface",
        "//reverb/cc/selectors:lifo",
        "//reverb/cc/selectors:prioritized",
        "//reverb/cc/selectors:uniform",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

reverb_cc_library(
    name = "server",
    srcs = ["server.cc"],
    deps = [
        "//reverb/cc:client",
        "//reverb/cc:reverb_service_impl",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/platform:grpc_utils",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:server_hdr",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:periodic_closure",
        "@com_google_absl//absl/strings",
    ] + reverb_grpc_deps(),
    alwayslink = 1,
)

reverb_cc_library(
    name = "status_macros",
    hdrs = [
        "status_macros.h",
        "status_matchers.h",
    ],
    deps = [
        "//reverb/cc/platform:logging",
        "@com_google_absl//absl/status",
    ],
)

reverb_cc_library(
    name = "status_matchers",
    hdrs = ["status_matchers.h"],
    deps = [
        "@com_google_absl//absl/status",
    ],
)

reverb_cc_library(
    name = "thread",
    srcs = ["thread.cc"],
    deps = [
        "//reverb/cc/platform:thread_hdr",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

reverb_cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
)

reverb_cc_library(
    name = "hash",
    hdrs = ["hash.h"],
    deps = reverb_tf_deps(),
)

reverb_cc_library(
    name = "hash_map",
    hdrs = ["hash_map.h"],
    deps = [
        ":hash",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

reverb_cc_library(
    name = "hash_set",
    hdrs = ["hash_set.h"],
    deps = [
        ":hash",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

reverb_cc_library(
    name = "grpc_utils",
    srcs = ["grpc_utils.cc"],
    deps = [
        "//reverb/cc/platform:grpc_utils_hdr",
    ] + reverb_grpc_deps(),
    alwayslink = 1,
)

reverb_cc_library(
    name = "net",
    srcs = ["net.cc"],
    deps = [
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:net_hdr",
    ],
    alwayslink = 1,
)

# --- SimpleCheckpointer: TFRecordCheckpointer 的去 TF 替代(Task 4) ---
# 原生 cc_library/cc_test:不注入 TF 依赖。用 length-delimited IO + std::filesystem。
cc_library(
    name = "simple_checkpointer",
    srcs = ["simple_checkpointer.cc"],
    hdrs = ["simple_checkpointer.h"],
    copts = ["-Wno-sign-compare"],
    visibility = ["//visibility:public"],
    deps = [
        "//reverb/cc:chunk_store",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:table",
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/platform:checkpointing_utils",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:hash_set",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:length_delimited_io",
        "//reverb/cc/support:trajectory_util",
        "//reverb/cc/table_extensions:interface",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_test(
    name = "simple_checkpointer_test",
    srcs = ["simple_checkpointer_test.cc"],
    copts = ["-Wno-sign-compare"],
    deps = [
        ":simple_checkpointer",
        "//reverb/cc:chunk_store",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:table",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/selectors:uniform",
        "//third_party/reverb_tensor:reverb_tensor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
