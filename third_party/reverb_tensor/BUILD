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
