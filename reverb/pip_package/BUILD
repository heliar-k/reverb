load("//reverb/pip_package:python_wheel.bzl", "collect_data_files", "transitive_py_deps")
load("@python_version_repo//:py_version.bzl", "REQUIREMENTS")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "licenses",
    data = [
        "//:LICENSE",
    ],
)

compile_pip_requirements(
    name = "requirements",
    srcs = [
        "requirements.in",
    ],
    extra_args = [
        "--allow-unsafe",
        "--build-isolation",
        "--rebuild",
    ],
    generate_hashes = True,
    requirements_txt = REQUIREMENTS,
    tags = ["manual"],
)

exports_files(
    glob(["requirements*.txt"]),
)

# cc_deps / py_deps: PEP 517 后端 (_build_backend.py) 经 bazel cquery 收集
# 这两个 target 的产物来组装 wheel。
collect_data_files(
    name = "cc_deps",
    deps = [
        "//reverb",
    ],
)

transitive_py_deps(
    name = "py_deps",
    deps = [
        "//reverb",
        "//reverb/server_executable:server_from_proto",
        "//reverb/server_executable:server_main",
    ],
)
