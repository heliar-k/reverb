package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

# ponytail: vendored patches(自 @local_xla,无 @local_xla 引用),供
# http_archive 的 patch_file 属性引用。
exports_files(
    [
        "grpc.patch",
        "protobuf.patch",
        "googletest.patch",
    ],
)
