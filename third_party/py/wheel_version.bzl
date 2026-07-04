"""Generate @tf_wheel_version_suffix (reverb-local, replaces @local_xla python_wheel.bzl).

ponytail: 仅保留 python_wheel_version_suffix_repository(其余 wheel helpers已由
reverb/pip_package/python_wheel.bzl 内联)。读取 ML_WHEEL_TYPE/ML_BUILD_DATE/
ML_GIT_HASH/ML_WHEEL_VERSION_SUFFIX 环境变量生成 wheel_version_suffix.bzl。
"""

_WHEEL_TYPE = "ML_WHEEL_TYPE"
_BUILD_DATE = "ML_BUILD_DATE"
_GIT_HASH = "ML_GIT_HASH"
_VERSION_SUFFIX = "ML_WHEEL_VERSION_SUFFIX"

def _get_env(repository_ctx, name, default_value = None):
    if name in repository_ctx.os.environ:
        return repository_ctx.os.environ[name]
    return default_value

def _python_wheel_version_suffix_repository_impl(repository_ctx):
    wheel_type = _get_env(repository_ctx, _WHEEL_TYPE, "snapshot")
    build_date = _get_env(repository_ctx, _BUILD_DATE)
    git_hash = _get_env(repository_ctx, _GIT_HASH)
    custom_version_suffix = _get_env(repository_ctx, _VERSION_SUFFIX)

    if wheel_type not in ["release", "nightly", "snapshot", "custom"]:
        fail("Environment variable ML_WHEEL_TYPE should have values \"release\", \"nightly\", \"custom\" or \"snapshot\"")

    wheel_version_suffix = ""
    semantic_wheel_version_suffix = ""
    if wheel_type == "nightly":
        if not build_date:
            fail("Environment variable ML_BUILD_DATE is required for nightly builds!")
        formatted_date = build_date.replace("-", "")
        wheel_version_suffix = ".dev{}".format(formatted_date)
        semantic_wheel_version_suffix = "-dev{}".format(formatted_date)
    elif wheel_type == "release":
        if custom_version_suffix:
            wheel_version_suffix = custom_version_suffix.replace("-", "")
            semantic_wheel_version_suffix = custom_version_suffix
    elif wheel_type == "custom":
        if build_date:
            formatted_date = build_date.replace("-", "")
            wheel_version_suffix += ".dev{}".format(formatted_date)
            semantic_wheel_version_suffix += "-dev{}".format(formatted_date)
        if git_hash:
            formatted_hash = git_hash.lstrip("0")[:9]
            wheel_version_suffix += "+{}".format(formatted_hash)
            semantic_wheel_version_suffix += "+{}".format(formatted_hash)
        if custom_version_suffix:
            wheel_version_suffix += custom_version_suffix.replace("-", "")
            semantic_wheel_version_suffix += custom_version_suffix
    else:
        wheel_version_suffix = ".dev0+selfbuilt"
        semantic_wheel_version_suffix = "-dev0+selfbuilt"

    repository_ctx.file(
        "wheel_version_suffix.bzl",
        "WHEEL_VERSION_SUFFIX = '{wheel_version_suffix}'\nSEMANTIC_WHEEL_VERSION_SUFFIX = '{semantic_wheel_version_suffix}'".format(
            wheel_version_suffix = wheel_version_suffix,
            semantic_wheel_version_suffix = semantic_wheel_version_suffix,
        ),
    )
    repository_ctx.file("BUILD", "")

python_wheel_version_suffix_repository = repository_rule(
    implementation = _python_wheel_version_suffix_repository_impl,
    environ = [
        _WHEEL_TYPE,
        _BUILD_DATE,
        _GIT_HASH,
        _VERSION_SUFFIX,
    ],
)
