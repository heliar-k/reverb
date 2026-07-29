# Description: Reverb is an efficient and easy to use prioritized replay system designed for ML research.

load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_cc_grpc_library",
    "reverb_cc_library",
    "reverb_cc_proto_library",
    "reverb_cc_test",
    "reverb_grpc_deps",
    "reverb_py_proto_library",
    "reverb_pybind_deps",
)

package(default_visibility = ["//reverb:__subpackages__"])

licenses(["notice"])

exports_files(["LICENSE"])

reverb_cc_test(
    name = "chunk_store_test",
    srcs = ["chunk_store_test.cc"],
    deps = [
        ":chunk_store",
        ":schema_cc_proto",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/platform:thread",
        "//reverb/cc/testing:proto_test_util",
        "@com_google_absl//absl/status",
    ],
)

reverb_cc_test(
    name = "rate_limiter_test",
    srcs = ["rate_limiter_test.cc"],
    deps = [
        ":table",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/platform:thread",
        "//reverb/cc/selectors:uniform",
        "//reverb/cc/testing:proto_test_util",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

# ponytail: 原生 cc_test,去 reverb_tf_deps() + :proto_test_util(后者仍依赖 TF)。
# 测试数据构造内联(MakeChunkData/MakePrioritizedItem/EqualsProto),见 table_test.cc。
cc_test(
    name = "table_test",
    srcs = ["table_test.cc"],
    copts = ["-Wno-sign-compare"],
    # ponytail: :table -> :trajectory_util -> :tensor_compression -> :tensor_proxy
    deps = [
        ":chunk_store",
        ":schema_cc_proto",
        ":table",
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/platform:thread",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/selectors:uniform",
        "//reverb/cc/support:task_executor",
        "//reverb/cc/table_extensions:interface",
        "//third_party/reverb_tensor:reverb_tensor_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

cc_test(
    name = "tensor_compression_test",
    srcs = ["tensor_compression_test.cc"],
    size = "small",
    copts = ["-Wno-sign-compare"],
    # ponytail: :tensor_compression 依赖 :tensor_proxy(pybind11/numpy C-API),
    deps = [
        ":tensor_compression",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/support:tensor_proxy",
        "//third_party/reverb_tensor:reverb_tensor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

# ponytail: 原生 cc_test,去 reverb_tf_deps() + :tensor_testutil(后者依赖 TF)。
# 测试数据构造/比较内联(MakeTensor/SliceRows/ExpectTensorBufferEqual),
# 见 sampler_test.cc。grpc++ 为 FakeStub/gRPC worker 所需。
cc_test(
    name = "sampler_test",
    srcs = ["sampler_test.cc"],
    copts = ["-Wno-sign-compare"],
    deps = [
        ":chunk_store",
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":sampler",
        ":table",
        ":tensor_compression",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/support:tensor_proxy",
        "//reverb/cc/testing:time_testutil",
        "//third_party/reverb_tensor:reverb_tensor_cc_proto",
        "@com_github_grpc_grpc//:grpc++_test",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

reverb_cc_test(
    name = "client_test",
    srcs = ["client_test.cc"],
    deps = [
        ":chunker",
        ":client",
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":trajectory_writer",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/support:uint128",
        "//reverb/cc/testing:proto_test_util",
    ] + reverb_grpc_deps(),
)

reverb_cc_library(
    name = "errors",
    srcs = ["errors.cc"],
    hdrs = ["errors.h"],
    visibility = [
        "//reverb:__subpackages__",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

reverb_cc_library(
    name = "chunk_store",
    srcs = ["chunk_store.cc"],
    hdrs = ["chunk_store.h"],
    deps = [
        ":schema_cc_proto",
        "//reverb/cc/platform:hash_map",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

# ponytail: 原生 cc_library,去 reverb_tf_deps() 注入(Task 8)。table.h/cc 已无 TF include。
cc_library(
    name = "table",
    srcs = [
        "rate_limiter.cc",
        "table.cc",
    ],
    hdrs = [
        "rate_limiter.h",
        "table.h",
    ],
    copts = ["-Wno-sign-compare"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        ":chunk_store",
        ":errors",
        ":schema_cc_proto",
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:hash_set",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:thread",
        "//reverb/cc/selectors:interface",
        "//reverb/cc/support:state_statistics",
        "//reverb/cc/support:task_executor",
        "//reverb/cc/support:trajectory_util",
        "//reverb/cc/table_extensions:interface",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tensor_compression",
    srcs = ["tensor_compression.cc"],
    hdrs = ["tensor_compression.h"],
    copts = ["-Wno-sign-compare"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:snappy",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:tensor_proxy",
        "//third_party/reverb_tensor:reverb_tensor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

# ponytail: 原生 cc_library,去 reverb_tf_deps() 注入(Task 9)。sampler.h/cc 已无 TF include,
# 数据出口改为 TensorBuffer。GrpcSamplerWorker 仍需 grpc++。
# ponytail: 去掉 -lpython3.10 linkopts——Py* 符号保持 undefined,运行时由宿主
# Python 进程注入(libreverb.so 经 libpybind.so 被 dlopen 加载)。这让 .so 跟
# Python 版本解耦,源码安装不再依赖系统装了某版本 libpython。
cc_library(
    name = "sampler",
    srcs = ["sampler.cc"],
    hdrs = ["sampler.h"],
    copts = ["-Wno-sign-compare"],
    deps = [
        ":chunk_store",
        ":errors",
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":schema_cc_proto",
        ":table",
        ":tensor_compression",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:hash_set",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:thread",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:queue",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:tensor_proxy",
        "//reverb/cc/support:trajectory_util",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

reverb_cc_library(
    name = "writer",
    srcs = ["writer.cc"],
    hdrs = ["writer.h"],
    deps = [
        ":chunk_store",
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":schema_cc_proto",
        ":table",
        ":tensor_compression",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:hash_set",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:thread",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:tensor_proxy",
        "//reverb/cc/support:trajectory_util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
    ] + reverb_grpc_deps(),
)

# ponytail: 原生 reverb_cc_test,去 TF,改用 TensorBuffer(numpy)。reverb_cc_test 宏已
# 统一注入 libpython linkopts(tensor_proxy 经 pybind11/numpy C-API 需链接 Py* 符号)。
reverb_cc_test(
    name = "writer_test",
    srcs = ["writer_test.cc"],
    deps = [
        ":client",
        ":writer",
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":schema_cc_proto",
        ":table",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/platform:thread",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:queue",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:tensor_proxy",
        "//reverb/cc/support:trajectory_util",
        "//reverb/cc/support:uint128",
        "//reverb/cc/testing:proto_test_util",
        "//third_party/reverb_tensor:reverb_tensor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
    ] + reverb_grpc_deps(),
)

cc_library(
    name = "trajectory_writer",
    srcs = ["trajectory_writer.cc"],
    hdrs = ["trajectory_writer.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        ":chunker",
        ":chunk_store",
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":schema_cc_proto",
        ":table",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:hash_set",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:thread",
        "//reverb/cc/shm:ring",
        "//reverb/cc/shm:shm_connection",
        "//reverb/cc/shm:shm_protocol_cc_proto",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:key_generators",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:tensor_proxy",
        "//reverb/cc/support:trajectory_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ] + reverb_grpc_deps(),
)

# ponytail: 原生 cc_library,进程内 Client(零 gRPC)。本地路径直接调 :table /
# :trajectory_writer / :sampler 的本地构造函数。NewStructuredWriter 镜像
# Client::NewStructuredWriter,内部调 NewTrajectoryWriter 绑定单一 table。
cc_library(
    name = "in_process_client",
    srcs = ["in_process_client.cc"],
    hdrs = ["in_process_client.h"],
    copts = ["-Wno-sign-compare"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        ":chunk_store",
        ":patterns_cc_proto",
        ":sampler",
        ":schema_cc_proto",
        ":structured_writer",
        ":table",
        ":trajectory_writer",
        ":writer",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:signature",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

reverb_cc_library(
    name = "streaming_trajectory_writer",
    srcs = ["streaming_trajectory_writer.cc"],
    hdrs = ["streaming_trajectory_writer.h"],
    deps = [
        ":chunker",
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":schema_cc_proto",
        ":trajectory_writer",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:hash_set",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:key_generators",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:tensor_proxy",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ] + reverb_grpc_deps(),
)

cc_test(
    name = "in_process_client_test",
    srcs = ["in_process_client_test.cc"],
    copts = ["-Wno-sign-compare"],
    # ponytail: :in_process_client -> :trajectory_writer/:sampler -> :tensor_proxy
    deps = [
        ":chunker",
        ":in_process_client",
        ":schema_cc_proto",
        ":structured_writer",
        ":table",
        ":trajectory_writer",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/platform/default:simple_checkpointer",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/support:tensor_proxy",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)
cc_test(
    name = "trajectory_writer_test",
    srcs = ["trajectory_writer_test.cc"],
    # ponytail: :trajectory_writer -> :chunker -> :tensor_proxy(pybind11/numpy
    deps = [
        ":chunker",
        ":schema_cc_proto",
        ":table",
        ":trajectory_writer",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:tensor_proxy",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

cc_test(
    name = "streaming_trajectory_writer_test",
    srcs = ["streaming_trajectory_writer_test.cc"],
    copts = ["-Wno-sign-compare"],
    # ponytail: 原生 cc_test,去 reverb_tf_deps() + :proto_test_util/:tensor_testutil(后者仍 TF-coupled)。
    # :streaming_trajectory_writer -> :chunker -> :tensor_proxy(pybind11/numpy C-API),
    deps = [
        ":chunker",
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":streaming_trajectory_writer",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/support:queue",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:tensor_proxy",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_github_grpc_grpc//:grpc++_test",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
    "@rules_python//python/cc:current_py_cc_libs",
    ] + reverb_grpc_deps(),
)

cc_test(
    name = "trajectory_writer_grpc_test",
    srcs = ["trajectory_writer_grpc_test.cc"],
    copts = ["-Wno-sign-compare"],
    # ponytail: gRPC reactor-path tests for TrajectoryWriter. Uses a custom
    # FakeAsyncStub whose async()->InsertStream hands out a FakeReactorStream
    # (a ClientCallbackReaderWriter subclass) instead of the synchronous
    # MockClientReaderWriter used by writer_test/streaming_trajectory_writer_test,
    # because TrajectoryWriter drives the stream via the callback reactor API.
    # :trajectory_writer -> :chunker -> :tensor_proxy(pybind11/numpy C-API),
    deps = [
        ":chunker",
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":schema_cc_proto",
        ":trajectory_writer",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/platform:thread",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:queue",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:tensor_proxy",
        "//reverb/cc/support:trajectory_util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_github_grpc_grpc//:grpc++_test",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
    "@rules_python//python/cc:current_py_cc_libs",
    ] + reverb_grpc_deps(),
)

cc_test(
    name = "reverb_service_impl_test",
    srcs = ["reverb_service_impl_test.cc"],
    copts = ["-Wno-sign-compare"],
    # ponytail: 原生 cc_test,去 reverb_tf_deps() + TF 头(struct.pb/env.h/tensor_shape)。
    # :reverb_service_impl -> :table -> :chunker -> :tensor_proxy(pybind11/numpy
    deps = [
        ":reverb_service_impl",
        ":schema_cc_proto",
        ":table",
        ":task_worker",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/platform/default:simple_checkpointer",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/platform:thread",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/selectors:interface",
        "//reverb/cc/selectors:uniform",
        "//reverb/cc/testing:proto_test_util",
        "//third_party/reverb_tensor:reverb_tensor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_github_grpc_grpc//:grpc++_test",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
    "@rules_python//python/cc:current_py_cc_libs",
    ] + reverb_grpc_deps(),
)

cc_library(
    name = "chunker",
    srcs = ["chunker.cc"],
    hdrs = ["chunker.h"],
    copts = ["-Wno-sign-compare"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        ":schema_cc_proto",
        ":tensor_compression",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:hash_set",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:key_generators",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:tensor_proxy",
        "//reverb/cc/support:trajectory_util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "chunker_test",
    srcs = ["chunker_test.cc"],
    copts = ["-Wno-sign-compare"],
    # ponytail: :chunker -> :tensor_proxy(pybind11/numpy C-API),
    # 链接期需 libpython 提供 Py_* 符号(同 tensor_proxy_test_bin 先例)。
    deps = [
        ":chunker",
        ":schema_cc_proto",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:tensor_proxy",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

reverb_cc_library(
    name = "client",
    srcs = ["client.cc"],
    hdrs = ["client.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        ":chunker",
        ":patterns_cc_proto",
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":sampler",
        ":schema_cc_proto",
        ":streaming_trajectory_writer",
        ":structured_writer",
        ":trajectory_writer",
        ":writer",
        "//reverb/cc/platform:grpc_utils",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:uint128",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
    "@rules_python//python/cc:current_py_cc_libs",
    ] + reverb_grpc_deps(),
)

reverb_cc_proto_library(
    name = "schema_cc_proto",
    srcs = ["schema.proto"],
    deps = ["//third_party/reverb_tensor:reverb_tensor_cc_proto"],
)

reverb_py_proto_library(
    name = "schema_py_pb2",
    srcs = ["schema.proto"],
    deps = [
        ":schema_cc_proto",
        "//third_party/reverb_tensor:reverb_tensor_py_pb2",
    ],
)

reverb_cc_proto_library(
    name = "patterns_cc_proto",
    srcs = ["patterns.proto"],
    deps = ["//third_party/reverb_tensor:reverb_tensor_cc_proto"],
    visibility = ["//reverb:__subpackages__"],
)

reverb_py_proto_library(
    name = "patterns_py_pb2",
    srcs = ["patterns.proto"],
    deps = [":patterns_cc_proto"],
)

reverb_cc_proto_library(
    name = "reverb_service_cc_proto",
    srcs = ["reverb_service.proto"],
    deps = [":schema_cc_proto"],
)

reverb_cc_grpc_library(
    name = "reverb_service_cc_grpc_proto",
    srcs = ["reverb_service.proto"],
    generate_mocks = True,
    visibility = [
        "//reverb:__subpackages__",
    ],
    deps = [":reverb_service_cc_proto"],
)

reverb_cc_library(
    name = "reverb_service_impl",
    srcs = ["reverb_service_impl.cc"],
    hdrs = [
        "reverb_service_impl.h",
    ],
    deps = [
        ":reverb_server_reactor",
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":schema_cc_proto",
        ":table",
        ":task_worker",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:trajectory_util",
        "//reverb/cc/support:uint128",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ] + reverb_grpc_deps(),
    alwayslink = 1,
)

reverb_cc_library(
    name = "task_worker",
    hdrs = ["task_worker.h"],
    deps = [
        ":table",
        ":thread_stats",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:thread",
        "//reverb/cc/support:periodic_closure",
        "//reverb/cc/support:unbounded_queue",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

reverb_cc_library(
    name = "reverb_server_reactor",
    hdrs = ["reverb_server_reactor.h"],
    deps = [
        ":task_worker",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:grpc_util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ] + reverb_grpc_deps(),
)

reverb_cc_library(
    name = "thread_stats",
    srcs = ["thread_stats.cc"],
    hdrs = ["thread_stats.h"],
    deps = [
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

reverb_cc_test(
    name = "thread_stats_test",
    srcs = ["thread_stats_test.cc"],
    deps = [
        ":thread_stats",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "structured_writer_test",
    srcs = ["structured_writer_test.cc"],
    copts = ["-Wno-sign-compare"],
    # ponytail: :structured_writer -> :trajectory_writer -> :tensor_proxy
    deps = [
        ":chunker",
        ":patterns_cc_proto",
        ":structured_writer",
        ":trajectory_writer",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:tensor_proxy",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

cc_library(
    name = "structured_writer",
    srcs = ["structured_writer.cc"],
    hdrs = ["structured_writer.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        ":chunker",
        ":patterns_cc_proto",
        ":trajectory_writer",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:tensor_proxy",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
    ],
)

