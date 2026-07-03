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
        "@com_google_absl//absl/log:check",
        # ponytail: 此 TF-era 测试辅助直接 include tensorflow/... 头文件,显式依赖
        # @pypi//tf_nightly。内嵌路径不依赖此 target,故不影响 numpy-only 链路。
        "@pypi//tf_nightly:headers_lib",
        "@pypi//tf_nightly:framework_lib",
    ],
)

reverb_cc_library(
    name = "tensor_testutil",
    testonly = 1,
    hdrs = ["tensor_testutil.h"],
    deps = [
        "//reverb/cc/platform:logging",
        # ponytail: TF-era 测试辅助,显式依赖 TF 头文件(同上)。
        "@pypi//tf_nightly:headers_lib",
        "@pypi//tf_nightly:framework_lib",
    ],
)

reverb_cc_library(
    name = "time_testutil",
    testonly = 1,
    hdrs = ["time_testutil.h"],
    deps = ["@com_google_absl//absl/time"],
)
