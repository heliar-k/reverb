# Description: Reverb SHM transport benchmarks (ticket ⑦).

load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_py_standard_imports",
)

load("@rules_python//python:py_binary.bzl", "py_binary")

package(
    default_visibility = ["//reverb:__subpackages__"],
)

licenses(["notice"])

# ponytail: plain py_binary (not reverb_py_test) so `bazel run` produces a
# runnable script. Mirrors the shm_test deps (//reverb pulls pybind + the full
# client/server surface); portpicker is needed for the gRPC loopback transport.
py_binary(
    name = "shm_benchmark",
    srcs = ["shm_benchmark.py"],
    main = "shm_benchmark.py",
    deps = reverb_py_standard_imports() + [
        "//reverb",
    ],
)
