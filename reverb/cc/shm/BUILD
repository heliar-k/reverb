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

# ---- Byte Pool ----

reverb_cc_library(
    name = "byte_pool",
    srcs = ["byte_pool.cc"],
    hdrs = ["byte_pool.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

reverb_cc_test(
    name = "byte_pool_test",
    srcs = ["byte_pool_test.cc"],
    deps = [
        ":byte_pool",
        "//reverb/cc/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

reverb_cc_test(
    name = "byte_pool_echo_test",
    srcs = ["byte_pool_echo_test.cc"],
    deps = [
        ":bootstrap",
        ":byte_pool",
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
        ":byte_pool",
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

# ---- ShmServer (ticket ③) ----

# ponytail: libpython linkopts come transitively from :tensor_proxy (via
# :table/:trajectory_util); the server builds TensorBuffers (no GIL) but links
# the numpy C-API symbols through that dep.
reverb_cc_library(
    name = "shm_server",
    srcs = ["shm_server.cc"],
    hdrs = ["shm_server.h"],
    deps = [
        ":bootstrap",
        ":byte_pool",
        ":ring",
        ":shm_connection",
        ":shm_protocol_cc_proto",
        "//reverb/cc:table",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:hash_set",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:tensor_proxy",
        "//reverb/cc/support:trajectory_util",
        "//third_party/reverb_tensor:reverb_tensor_cc_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

# ---- ShmClient (ticket ③) ----

reverb_cc_library(
    name = "shm_client",
    srcs = ["shm_client.cc"],
    hdrs = ["shm_client.h"],
    deps = [
        ":bootstrap",
        ":byte_pool",
        ":ring",
        ":shm_connection",
        ":shm_protocol_cc_proto",
        "//reverb/cc:errors",
        "//reverb/cc:sampler",
        "//reverb/cc:table",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:thread",
        "//reverb/cc/support:queue",
        "//reverb/cc/support:tensor_proxy",
        "//third_party/reverb_tensor:reverb_tensor_cc_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

# ---- Sample-path integration test (ticket ③) ----

reverb_cc_test(
    name = "shm_sample_test",
    srcs = ["shm_sample_test.cc"],
    deps = [
        ":shm_client",
        ":shm_server",
        ":shm_protocol_cc_proto",
        "//reverb/cc:chunk_store",
        "//reverb/cc:sampler",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:table",
        "//reverb/cc:tensor_compression",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/support:tensor_proxy",
        "//third_party/reverb_tensor:reverb_tensor_cc_proto",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
