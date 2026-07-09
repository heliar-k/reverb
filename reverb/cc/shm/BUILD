load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_cc_library",
    "reverb_cc_test",
    "reverb_cc_proto_library",
)

package(default_visibility = ["//reverb:__subpackages__"])

# ---- Proto ----

reverb_cc_proto_library(
    name = "shm_protocol_cc_proto",
    srcs = ["shm_protocol.proto"],
    deps = [
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:patterns_cc_proto",
        "//reverb/cc:reverb_service_cc_proto",
        "//third_party/reverb_tensor:reverb_tensor_cc_proto",
    ],
)

# ---- Ring ----

reverb_cc_library(
    name = "ring",
    srcs = ["ring.cc"],
    hdrs = ["ring.h"],
    deps = [
        ":shm_protocol_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

reverb_cc_test(
    name = "ring_test",
    srcs = ["ring_test.cc"],
    deps = [
        ":ring",
        ":shm_protocol_cc_proto",
        "//reverb/cc/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

# ---- Bootstrap (udsocket) ----

reverb_cc_library(
    name = "bootstrap",
    srcs = ["bootstrap.cc"],
    hdrs = ["bootstrap.h"],
    deps = [
        ":shm_protocol_cc_proto",
        "//reverb/cc/platform:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

reverb_cc_test(
    name = "bootstrap_test",
    srcs = ["bootstrap_test.cc"],
    deps = [
        ":bootstrap",
        ":shm_protocol_cc_proto",
        "//reverb/cc/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

# ---- ShmConnection ----

reverb_cc_library(
    name = "shm_connection",
    hdrs = ["shm_connection.h"],
    deps = [
        ":ring",
    ],
)

reverb_cc_test(
    name = "echo_test",
    srcs = ["echo_test.cc"],
    deps = [
        ":bootstrap",
        ":ring",
        ":shm_connection",
        ":shm_protocol_cc_proto",
        "//reverb/cc/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
