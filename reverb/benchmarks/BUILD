# Description: Reverb client benchmarks.

load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_py_standard_imports",
)

load("@rules_python//python:py_binary.bzl", "py_binary")

package(
    default_visibility = ["//reverb:__subpackages__"],
)

licenses(["notice"])

# Unified benchmark: sample, insert, and pipeline paths across all transports.
# Usage: bazel run //reverb/benchmarks:client_benchmark -- --quick
py_binary(
    name = "client_benchmark",
    srcs = ["client_benchmark.py"],
    main = "client_benchmark.py",
    deps = reverb_py_standard_imports() + [
        "//reverb",
    ],
)
