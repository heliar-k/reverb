package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

# ponytail: 自包含 Python 初始化 helpers(替代 @local_xla python_init_*)。
# python_repo.bzl / python_init.bzl / wheel_version.bzl 为 .bzl 文件,exports_files
# 暴露 patches 给 http_archive 的 patches/patch_file 属性。
exports_files(
    [
        "rules_cc_protobuf.patch",
        "rules_python_pip_version.patch",
        "rules_python_freethreaded.patch",
        "rules_python_versions.patch",
        "python_repo.bzl",
        "python_init_rules.bzl",
        "python_init_repositories.bzl",
    ],
)
