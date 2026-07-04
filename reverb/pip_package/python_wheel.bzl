"""Wheel data-collection helpers, vendored from @local_xla//third_party/py:python_wheel.bzl.

解耦 @local_xla:原宏仅依赖 @rules_python//python:py_info.bzl 的 PyInfo,
无 @local_xla 逻辑,直接内联到 reverb 仓库,pip_package 不再 load @local_xla。
"""

load("@rules_python//python:py_info.bzl", "PyInfo")

def _transitive_py_deps_impl(ctx):
    outputs = depset(
        [],
        transitive = [dep[PyInfo].transitive_sources for dep in ctx.attr.deps],
    )
    return DefaultInfo(files = outputs)

_transitive_py_deps = rule(
    attrs = {
        "deps": attr.label_list(
            allow_files = True,
            providers = [PyInfo],
        ),
    },
    implementation = _transitive_py_deps_impl,
)

def transitive_py_deps(name, deps = []):
    _transitive_py_deps(name = name + "_gather", deps = deps)
    native.filegroup(name = name, srcs = [":" + name + "_gather"])

FilePathInfo = provider(
    "Returns path of selected files.",
    fields = {
        "files": "requested files from data attribute",
    },
)

def _collect_data_aspect_impl(_, ctx):
    files = {}
    extensions = ctx.attr._extensions
    if hasattr(ctx.rule.attr, "data"):
        for data in ctx.rule.attr.data:
            for f in data.files.to_list():
                if not f.owner.package:
                    continue
                for ext in extensions:
                    if f.extension == ext:
                        files[f] = True
                        break

    if hasattr(ctx.rule.attr, "deps"):
        for dep in ctx.rule.attr.deps:
            if dep[FilePathInfo].files:
                for f in dep[FilePathInfo].files.to_list():
                    files[f] = True

    return [FilePathInfo(files = depset(files.keys()))]

collect_data_aspect = aspect(
    implementation = _collect_data_aspect_impl,
    attr_aspects = ["deps"],
    attrs = {
        "_extensions": attr.string_list(
            default = ["so", "pyd", "pyi", "dll", "dylib", "lib", "pd"],
        ),
    },
)

def _collect_symlink_data_aspect_impl(_, ctx):
    files = {}
    symlink_extensions = ctx.attr._symlink_extensions
    if not hasattr(ctx.rule.attr, "deps"):
        return [FilePathInfo(files = depset(files.keys()))]
    for dep in ctx.rule.attr.deps:
        if not (dep[DefaultInfo].default_runfiles and
                dep[DefaultInfo].default_runfiles.files):
            continue
        for file in dep[DefaultInfo].default_runfiles.files.to_list():
            if not file.owner.package:
                continue
            for ext in symlink_extensions:
                if file.extension == ext:
                    files[file] = True
                    break

    return [FilePathInfo(files = depset(files.keys()))]

collect_symlink_data_aspect = aspect(
    implementation = _collect_symlink_data_aspect_impl,
    attr_aspects = ["symlink_deps"],
    attrs = {
        "_symlink_extensions": attr.string_list(
            default = ["pyi", "lib", "pd"],
        ),
    },
)

def _collect_data_files_impl(ctx):
    files = {}
    for dep in ctx.attr.deps:
        for f in dep[FilePathInfo].files.to_list():
            files[f] = True
    for symlink_dep in ctx.attr.symlink_deps:
        for f in symlink_dep[FilePathInfo].files.to_list():
            files[f] = True
    return [DefaultInfo(files = depset(files.keys()))]

collect_data_files = rule(
    implementation = _collect_data_files_impl,
    attrs = {
        "deps": attr.label_list(
            aspects = [collect_data_aspect],
        ),
        "symlink_deps": attr.label_list(
            aspects = [collect_symlink_data_aspect],
        ),
    },
)
