"""@python_version_repo + py toolchains (reverb-local).

ponytail: 替代 @local_xla 的 python_init_repositories.bzl +
python_init_toolchains.bzl。load @rules_python(py_repositories /
python_register_toolchains)与 //third_party/py:python_repo.bzl——
@rules_python 必须在 WORKSPACE 调用 python_init_rules() 之后(此时
@rules_python 已注册)才能 load 本文件。
"""

load("@rules_python//python:repositories.bzl", "py_repositories", "python_register_toolchains")
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

def get_toolchain_name_per_python_version(name, hermetic_python_version):
    return "{name}_{version}".format(
        name = name,
        version = hermetic_python_version.replace(".", "_"),
    )

def python_init_toolchains(name = "python", hermetic_python_version = "3.11"):
    """Register hermetic python toolchains (rules_python native)."""
    python_register_toolchains(
        name = get_toolchain_name_per_python_version(name, hermetic_python_version),
        ignore_root_user_error = True,
        python_version = hermetic_python_version,
    )
