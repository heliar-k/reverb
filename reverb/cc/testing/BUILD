load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_cc_library",
)

package(
    default_applicable_licenses = ["//third_party/reverb:license"],
    default_visibility = ["//reverb:__subpackages__"],
)

licenses(["notice"])

reverb_cc_library(
    name = "proto_test_util",
    testonly = 1,
    srcs = ["proto_test_util.cc"],
    hdrs = ["proto_test_util.h"],
    deps = [
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:tensor_compression",
        "//reverb/cc/platform:logging",
        "//reverb/cc/support:tensor_proxy",
        "//third_party/reverb_tensor:reverb_tensor_cc_proto",
        "@com_google_absl//absl/log:check",
        "@com_google_protobuf//:protobuf",
    ],
)

reverb_cc_library(
    name = "time_testutil",
    testonly = 1,
    hdrs = ["time_testutil.h"],
    deps = ["@com_google_absl//absl/time"],
)
