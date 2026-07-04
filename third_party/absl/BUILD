package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

# ponytail: vendored from @local_xla//third_party/absl。com_google_absl.BUILD +
# patches(自包含,无 @local_xla 引用)。absl LTS 20250814.0(与原 tf_workspace
# 一致),grpc_deps() 的 existing_rules() 保护会跳过它注册的旧版 absl。
exports_files(
    [
        "com_google_absl.BUILD",
        "btree.patch",
        "build_dll.patch",
        "endian.patch",
        "rules_cc.patch",
        "check_op.patch",
        "check_op_2.patch",
    ],
)
