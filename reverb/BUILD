# Description: Reverb is an efficient and easy to use prioritized replay system designed for ML research.

load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_cc_shared_library",
    "reverb_pybind_deps",
    "reverb_pybind_extension",
    "reverb_pytype_library",
    "reverb_pytype_strict_library",
)

package(
    default_visibility = [":__subpackages__"],
)

licenses(["notice"])

exports_files(["LICENSE"])

reverb_pytype_strict_library(
    name = "reverb",
    srcs = ["__init__.py"],
    deps = [
        ":client",
        ":errors",
        ":item_selectors",
        ":pybind",
        ":rate_limiters",
        ":replay_sample",
        ":server",
        ":trajectory_writer",
        "//reverb/platform/default:checkpointers",
    ],
)

reverb_pytype_strict_library(
    name = "rate_limiters",
    srcs = ["rate_limiters.py"],
    deps = [
        ":pybind",
    ],
)

reverb_pytype_library(
    name = "client",
    srcs = ["client.py"],
    strict_deps = True,
    deps = [
        ":errors",
        ":pybind",
        ":replay_sample",
        ":reverb_types",
        ":structured_writer",
        ":trajectory_writer",
    ],
)

reverb_pytype_library(
    name = "errors",
    srcs = ["errors.py"],
    strict_deps = True,
    deps = [],
)

reverb_pytype_library(
    name = "server",
    srcs = ["server.py"],
    strict_deps = True,
    deps = [
        ":client",
        ":item_selectors",
        ":pybind",
        ":rate_limiters",
        ":reverb_types",
        ":signature_codec",
        "//reverb/platform/default:checkpointers",
    ],
)

reverb_pytype_library(
    name = "replay_sample",
    srcs = ["replay_sample.py"],
    strict_deps = True,
)

reverb_cc_shared_library(
    name = "libreverb",
    additional_linker_inputs = [
        ":reverb_exported_symbols.lds",
        ":reverb_version_script.lds",
    ],
    exports_filter = [
        "//reverb/cc:__subpackages__",
        "@com_google_absl//absl:__subpackages__",
        # ponytail: pybind 去 TF 后直连 protobuf/grpc/zlib(经 :table/
        # :in_process_client/:sampler),这些库同时被 libreverb 与 :pybind
        # 持有,必须声明由 libreverb 导出,否则 cc_shared_library 报
        # "linked statically but not exported"。
        "@com_google_protobuf//src/google/protobuf:__subpackages__",
        "@com_google_protobuf//upb:__subpackages__",
        "@com_google_protobuf//third_party/utf8_range:__pkg__",
        "@zlib//:__pkg__",
        "//third_party/reverb_tensor:__pkg__",
        "@com_github_grpc_grpc//:__subpackages__",
    ],
    shared_lib_name = select(
        {
            "@platforms//os:macos": "libreverb.dylib",
            "@platforms//os:linux": "libreverb.so",
        },
    ),
    user_link_flags = select({
        "@platforms//os:macos": [
            "-Wl,-exported_symbols_list,$(location :reverb_exported_symbols.lds)",
            "-Wl,-install_name,@rpath/libreverb.dylib",
        ],
        "@platforms//os:linux": [
            "-Wl,--version-script,$(location :reverb_version_script.lds)",
        ],
        "//conditions:default": [],
    }),
    deps = [
        # ponytail: P4 — client.cc/writer.cc/streaming_trajectory_writer.cc 已去 TF,
        # 恢复 gRPC 模式聚合。
        "//reverb/cc:chunker",
        "//reverb/cc:client",
        "//reverb/cc:errors",
        "//reverb/cc:patterns_cc_proto",
        "//reverb/cc:sampler",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:streaming_trajectory_writer",
        "//reverb/cc:table",
        "//reverb/cc:trajectory_writer",
        "//reverb/cc:writer",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/platform:checkpointing",
        "//reverb/cc/platform:checkpointing_utils",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:server",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/selectors:heap",
        "//reverb/cc/selectors:interface",
        "//reverb/cc/selectors:lifo",
        "//reverb/cc/selectors:prioritized",
        "//reverb/cc/selectors:uniform",
        "//reverb/cc/support:cleanup",
        "//reverb/cc/support:key_generators",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:trajectory_util",
        "//reverb/cc/table_extensions:interface",
    ],
)

reverb_pybind_extension(
    name = "pybind",
    srcs = ["pybind.cc"],
    module_name = "libpybind",
    pytype_srcs = ["pybind.pyi"],
    visibility = [":__subpackages__"],
    # ponytail: P4 — client.cc/writer.cc/streaming_trajectory_writer.cc 已去 TF,
    # 恢复 gRPC Client/Writer pybind 绑定。
    deps = [
        "//reverb/cc:chunker",
        "//reverb/cc:client",
        "//reverb/cc:in_process_client",
        "//reverb/cc:patterns_cc_proto",
        "//reverb/cc:sampler",
        "//reverb/cc:streaming_trajectory_writer",
        "//reverb/cc:structured_writer",
        "//reverb/cc:table",
        "//reverb/cc:trajectory_writer",
        "//reverb/cc:writer",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/platform:checkpointing",
        "//reverb/cc/platform:checkpointing_utils",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:server",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/selectors:heap",
        "//reverb/cc/selectors:interface",
        "//reverb/cc/selectors:lifo",
        "//reverb/cc/selectors:prioritized",
        "//reverb/cc/selectors:uniform",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:tensor_proxy",
        "//reverb/cc/table_extensions:interface",
        "//third_party/reverb_tensor:reverb_tensor_cc_proto",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
    ] + reverb_pybind_deps(),
)

reverb_pytype_library(
    name = "signature_codec",
    srcs = ["signature_codec.py"],
    strict_deps = True,
    deps = [
        "//third_party/reverb_tensor:reverb_tensor_py_pb2",
    ],
)

reverb_pytype_library(
    name = "reverb_types",
    srcs = ["reverb_types.py"],
    strict_deps = True,
    deps = [
        ":pybind",
        ":signature_codec",
        "//reverb/cc:schema_py_pb2",
    ],
)

reverb_pytype_strict_library(
    name = "structured_writer",
    srcs = ["structured_writer.py"],
    deps = [
        ":errors",
        ":pybind",
        ":reverb_types",
        ":signature_codec",
        "//reverb/cc:patterns_py_pb2",
        "//third_party/reverb_tensor:reverb_tensor_py_pb2",
    ],
)

reverb_pytype_strict_library(
    name = "item_selectors",
    srcs = ["item_selectors.py"],
    deps = [":pybind"],
)

reverb_pytype_library(
    name = "trajectory_writer",
    srcs = ["trajectory_writer.py"],
    strict_deps = True,
    deps = [
        ":errors",
        ":pybind",
    ],
)

# ponytail: Python 测试已迁移至 //reverb/tests(见 reverb/tests/BUILD)。
# 原 reverb_py_test/py_test 目标(rate_limiters_test / trajectory_writer_test /
# replay_sample_test / server_test / client_test / structured_writer_test /
# in_process_test / grpc_roundtrip_test)与此处的 *_test.py 源文件一并移走。


