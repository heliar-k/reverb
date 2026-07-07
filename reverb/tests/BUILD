# Description: Reverb Python tests (migrated from reverb/).

load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_py_test",
)

package(
    default_visibility = ["//reverb:__subpackages__"],
)

licenses(["notice"])

# ponytail: 试点——验证 tests/ 目录能经 bazel 拉入 reverb 包。
reverb_py_test(
    name = "wiring_test",
    srcs = ["wiring_test.py"],
    deps = [
        "//reverb",
        "//reverb:errors",
        "//reverb:replay_sample",
    ],
)

# Runs examples/*.ipynb code cells in a pure-numpy embedded mode as a regression guard.
reverb_py_test(
    name = "examples_notebook_test",
    srcs = ["examples_notebook_test.py"],
    data = [
        "//examples:demo.ipynb",
        "//examples:frame_stacking.ipynb",
    ],
    deps = [
        "//reverb",
    ],
)

reverb_py_test(
    name = "errors_test",
    srcs = ["errors_test.py"],
    deps = [
        "//reverb:errors",
    ],
)

reverb_py_test(
    name = "item_selectors_test",
    srcs = ["item_selectors_test.py"],
    deps = [
        "//reverb:item_selectors",
        "//reverb:pybind",
    ],
)

reverb_py_test(
    name = "replay_sample_test",
    srcs = ["replay_sample_test.py"],
    deps = [
        "//reverb:pybind",
        "//reverb:replay_sample",
    ],
)

reverb_py_test(
    name = "rate_limiters_test",
    srcs = ["rate_limiters_test.py"],
    deps = [
        "//reverb:rate_limiters",
    ],
)

reverb_py_test(
    name = "signature_codec_test",
    srcs = ["signature_codec_test.py"],
    deps = [
        "//reverb:signature_codec",
    ],
)

reverb_py_test(
    name = "reverb_types_test",
    srcs = ["reverb_types_test.py"],
    deps = [
        "//reverb:item_selectors",
        "//reverb:rate_limiters",
        "//reverb:reverb_types",
        "//reverb:server",
        "//reverb:signature_codec",
    ],
)

reverb_py_test(
    name = "checkpointers_test",
    srcs = ["checkpointers_test.py"],
    deps = [
        "//reverb:pybind",
        "//reverb/platform:checkpointers_lib",
        "//reverb/platform/default:checkpointers",
    ],
)

reverb_py_test(
    name = "server_test",
    srcs = ["server_test.py"],
    deps = [
        "//reverb:item_selectors",
        "//reverb:pybind",
        "//reverb:rate_limiters",
        "//reverb:server",
        "//reverb:signature_codec",
    ],
)

reverb_py_test(
    name = "client_test",
    srcs = ["client_test.py"],
    deps = [
        "//reverb:client",
        "//reverb:errors",
        "//reverb:item_selectors",
        "//reverb:rate_limiters",
        "//reverb:server",
    ],
)

reverb_py_test(
    name = "trajectory_writer_test",
    srcs = ["trajectory_writer_test.py"],
    deps = [
        "//reverb:client",
        "//reverb:errors",
        "//reverb:pybind",
        "//reverb:server",
        "//reverb:trajectory_writer",
    ],
)

reverb_py_test(
    name = "structured_writer_test",
    srcs = ["structured_writer_test.py"],
    deps = [
        "//reverb:client",
        "//reverb:server",
        "//reverb:signature_codec",
        "//reverb:structured_writer",
    ],
)

reverb_py_test(
    name = "server_from_proto_test",
    srcs = ["server_from_proto_test.py"],
    deps = [
        "//reverb",
        "//reverb/server_executable:server_from_proto",
        "//reverb/cc:schema_py_pb2",
        "//reverb/cc/checkpointing:checkpoint_py_pb2",
    ],
)

reverb_py_test(
    name = "in_process_test",
    srcs = ["in_process_test.py"],
    deps = [
        "//reverb",
        "//reverb:errors",
        "//reverb/platform/default:checkpointers",
    ],
)

reverb_py_test(
    name = "grpc_roundtrip_test",
    srcs = ["grpc_roundtrip_test.py"],
    deps = [
        "//reverb",
    ],
)