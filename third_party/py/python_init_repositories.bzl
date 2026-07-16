"""@python_version_repo + py toolchains (reverb-local).

ponytail: 替代 @local_xla 的 python_init_repositories.bzl +
python_init_toolchains.bzl。load @rules_python(py_repositories /
python_register_toolchains)与 //third_party/py:python_repo.bzl——
@rules_python 必须在 WORKSPACE 调用 python_init_rules() 之后(此时
@rules_python 已注册)才能 load 本文件。
"""

load("@rules_python//python:repositories.bzl", "py_repositories")
load(
    "@rules_python//python/local_toolchains:repos.bzl",
    "local_runtime_repo",
    "local_runtime_toolchains_repo",
)
load("//third_party/py:python_repo.bzl", "python_repository")

def python_init_repositories(requirements, default_python_version = "3.11"):
    """Creates @python_version_repo (generates py_version.bzl) + py_repositories."""
    python_repository(
        name = "python_version_repo",
        requirements_versions = requirements.keys(),
        requirements_locks = requirements.values(),
        default_python_version = default_python_version,
    )
    py_repositories()

def python_init_toolchains(
        name = "python",
        hermetic_python_version = "3.11",
        interpreter_path = "python3"):
    """Register a local Python toolchain via rules_python local_runtime_repo.

    ponytail: 原 python_register_toolchains 下载 python-build-standalone(142M),
    与 cc 侧硬编码系统 libpython3.10 形成混搭。改用 local_runtime_repo 指向
    uv venv/系统 python:它通过 sysconfig 自省出 include 路径 + libpython +
    解释器,生成 _python_headers/_libpython/_py3_runtime 三件套 toolchain。
    cc 侧 Python.h 与 libpython 自动跟随该解释器版本,不再绑系统路径。
    多版本测试由 uv venv 在 bazel 外做(bazel 内单一解释器)。
    """
    local_runtime_repo(
        name = name,
        interpreter_path = interpreter_path,
        on_failure = "fail",
    )
    local_runtime_toolchains_repo(
        name = name + "_toolchains",
        runtimes = [name],
    )
    native.register_toolchains("@{}_toolchains//:all".format(name))
