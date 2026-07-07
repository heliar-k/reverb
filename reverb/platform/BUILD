load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_pytype_strict_library",
)

package(
    default_visibility = ["//reverb:__subpackages__"],
)

licenses(["notice"])

reverb_pytype_strict_library(
    name = "checkpointers_lib",
    srcs = ["checkpointers_lib.py"],
    deps = [
        "//reverb:pybind",
    ],
)

# ponytail: checkpointers_lib_test 已迁移至 //reverb/tests:checkpointers_test。
