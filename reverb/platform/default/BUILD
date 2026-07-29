load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_pytype_strict_library",
)

package(default_visibility = ["//reverb:__subpackages__"])

licenses(["notice"])

reverb_pytype_strict_library(
    name = "checkpointers",
    srcs = ["checkpointers.py"],
    srcs_version = "PY3",
    deps = [
        "//reverb:pybind",
    ],
)

reverb_pytype_strict_library(
    name = "server_main_command_line_args",
    srcs = ["server_main_command_line_args.py"],
    srcs_version = "PY3",
    deps = [
        "//reverb/server_executable:reverb_config_py_pb2",
    ],
)
