package(default_visibility = ["//visibility:public"])

# Exposed so reverb_cc_proto_library's genrule can pull the .proto as a
# protoc import source (the macro looks up `<name>_proto_filegroup` for deps).
filegroup(
    name = "reverb_tensor_proto_filegroup",
    srcs = ["reverb_tensor.proto"],
)

proto_library(
    name = "reverb_tensor_proto",
    srcs = ["reverb_tensor.proto"],
)

cc_proto_library(
    name = "reverb_tensor_cc_proto",
    deps = [":reverb_tensor_proto"],
)

# ponytail: Python proto so generated _pb2 modules (e.g. schema_pb2) can
# `from third_party.reverb_tensor import reverb_tensor_pb2` without TF.
# Re-added during the de-TF refactor (Task 15); the fork commit had dropped it.
# Hand-written genrule (not reverb_py_proto_library) because that macro's OUTDIR
# sed assumes the package lives under reverb/, which third_party/ does not.
genrule(
    name = "reverb_tensor_py_pb2_gen",
    srcs = ["reverb_tensor.proto"] + [
        "@com_google_protobuf//:well_known_type_protos",
    ],
    outs = ["reverb_tensor_pb2.py"],
    tools = ["@com_google_protobuf//:protoc"],
    cmd = """
        # Generate with --proto_path=. (matching schema_pb2) so the descriptor
        # source name is third_party/reverb_tensor/reverb_tensor.proto, which is
        # what schema_pb2 depends on. --python_out is the bin dir (RULEDIR with
        # the package suffix stripped) so the output lands at the declared path
        # without package-prefix doubling.
        RULEDIR=$(RULEDIR)
        OUTDIR=$${RULEDIR%/third_party/reverb_tensor}
        $(location @com_google_protobuf//:protoc) \
          --proto_path=external/com_google_protobuf/src \
          --proto_path=. \
          --python_out=$$OUTDIR \
          $(location reverb_tensor.proto)
    """,
)

py_library(
    name = "reverb_tensor_py_pb2",
    srcs = ["reverb_tensor_pb2.py"],
)
