# Platform-specific code for reverb

load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_cc_library",
    "reverb_cc_test",
    "reverb_grpc_deps",
)

package(
    default_applicable_licenses = ["//third_party/reverb:license"],
    default_visibility = ["//reverb:__subpackages__"],
)

licenses(["notice"])

reverb_cc_library(
    name = "checkpointing_utils_hdr",
    hdrs = ["checkpointing_utils.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        "//reverb/cc/selectors:interface",
    ],
)

reverb_cc_library(
    name = "checkpointing_utils",
    hdrs = ["checkpointing_utils.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        "//reverb/cc/platform/default:checkpointing_utils",
        "//reverb/cc/selectors:interface",
    ],
)

reverb_cc_library(
    name = "grpc_utils_hdr",
    hdrs = ["grpc_utils.h"],
    deps = reverb_grpc_deps() + ["@com_google_absl//absl/strings:string_view"],
)

reverb_cc_library(
    name = "grpc_utils",
    hdrs = ["grpc_utils.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        "//reverb/cc/platform/default:grpc_utils",
        "@com_google_absl//absl/strings:string_view",
    ] + reverb_grpc_deps(),
)

reverb_cc_library(
    name = "server_hdr",
    hdrs = ["server.h"],
    deps = [
        "//reverb/cc:table",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/support:periodic_closure",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
)

reverb_cc_library(
    name = "server",
    hdrs = ["server.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        "//reverb/cc:table",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/platform/default:server",
        "//reverb/cc/support:periodic_closure",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ] + reverb_grpc_deps(),
)

reverb_cc_test(
    name = "server_test",
    srcs = ["server_test.cc"],
    deps = [
        ":net",
        ":server",
        "//reverb/cc/platform/default:status_matchers",
        "@com_google_absl//absl/status",
    ] + reverb_grpc_deps(),
)

reverb_cc_library(
    name = "snappy_hdr",
    hdrs = ["snappy.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "@snappy",
    ],
)

reverb_cc_library(
    name = "snappy",
    hdrs = ["snappy.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        "//reverb/cc/platform/default:snappy",
        "@com_google_absl//absl/strings:string_view",
    ],
)

reverb_cc_library(
    name = "thread_hdr",
    hdrs = ["thread.h"],
    deps = ["@com_google_absl//absl/strings:string_view"],
)

reverb_cc_library(
    name = "thread",
    hdrs = ["thread.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        "//reverb/cc/platform/default:thread",
        "@com_google_absl//absl/strings:string_view",
    ],
)

reverb_cc_test(
    name = "thread_test",
    srcs = ["thread_test.cc"],
    deps = [
        ":thread",
        "@com_google_absl//absl/synchronization",
    ],
)

reverb_cc_library(
    name = "net_hdr",
    hdrs = ["net.h"],
)

reverb_cc_library(
    name = "net",
    hdrs = ["net.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = ["//reverb/cc/platform/default:net"],
)

reverb_cc_test(
    name = "net_test",
    srcs = ["net_test.cc"],
    deps = [
        "//reverb/cc/platform/default:logging",
        ":net",
    ],
)
