"""Default versions of reverb build rule helpers."""

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_shared_library.bzl", "cc_shared_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

def tf_copts():
    return ["-Wno-sign-compare"]

def reverb_cc_library(
        name,
        srcs = [],
        hdrs = [],
        deps = [],
        testonly = 0,
        **kwargs):
    if testonly:
        new_deps = [
            "@com_google_googletest//:gtest",
        ] + reverb_tf_deps()
    else:
        new_deps = []
    cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        copts = tf_copts(),
        testonly = testonly,
        deps = depset(deps + new_deps),
        **kwargs
    )

def reverb_kernel_library(name, srcs = [], deps = [], **kwargs):
    deps = deps + reverb_tf_deps()
    reverb_cc_library(
        name = name,
        srcs = srcs,
        deps = deps,
        alwayslink = 1,
        **kwargs
    )

reverb_cc_shared_library = cc_shared_library

def _removesuffix(x, txt):
    """Backport of x._removesuffix(txt) for Python version earlier than 3.9."""
    if x.endswith(txt):
        return x[:-len(txt)]
    return x

def _normalize_proto(x):
    return x.removesuffix("_proto").removesuffix("_cc") + "_proto"

def _filegroup_name(x):
    return _normalize_proto(x) + "_filegroup"

def reverb_cc_proto_library(name, srcs = [], deps = [], **kwargs):
    """Build a proto cc_library.

    This rule does three things:

    1) Create a filegroup with name `<name>_filegroup` that contains `srcs`
       and any sources from deps named "x_proto" or "x_cc_proto".

    2) Uses protoc to compile srcs to .h/.cc files.

    3) Creates a cc_library with name `name` building the resulting .h/.cc
       files.

    Args:
      name: The name, should end with "_cc_proto".
      srcs: The .proto files.
      deps: Any reverb_cc_proto_library targets.
      **kwargs: Any additional args for the cc_library rule.
    """
    gen_srcs = [_removesuffix(x, ".proto") + ".pb.cc" for x in srcs]
    gen_hdrs = [_removesuffix(x, ".proto") + ".pb.h" for x in srcs]
    src_paths = ["$(location {})".format(x) for x in srcs]
    dep_srcs = []
    for x in deps:
        if x.endswith("_proto"):
            dep_srcs.append(_filegroup_name(x))
    native.filegroup(
        name = _filegroup_name(name),
        srcs = srcs + dep_srcs,
        **kwargs
    )
    native.genrule(
        name = name + "_gen",
        srcs = srcs + dep_srcs + [
            "@com_google_protobuf//:well_known_type_protos",
        ],
        outs = gen_srcs + gen_hdrs,
        tools = [
            "@com_google_protobuf//:protoc",
        ],
        cmd = """
        OUTDIR=$$(echo $(RULEDIR) | sed -E -e 's#reverb(/.*|$$)##')
        $(location @com_google_protobuf//:protoc) \
          --proto_path=external/com_google_protobuf/src \
          --proto_path=. \
          --cpp_out=$$OUTDIR {srcs}""".format(
            srcs = " ".join(src_paths),
        ),
    )
    cc_library(
        name = "{}".format(name),
        srcs = gen_srcs,
        hdrs = gen_hdrs,
        deps = deps + ["@com_google_protobuf//:protobuf"],
        alwayslink = 1,
        **kwargs
    )

def reverb_py_proto_library(name, srcs = [], deps = [], **kwargs):
    """Build a proto py_library.

    This rule does three things:

    1) Create a filegroup with name `<name>_filegroup` that contains `srcs`
       and any sources from deps named "x_proto" or "x_py_proto".

    2) Uses protoc to compile srcs to _pb2.py files.

    3) Creates a py_library with name `name` building the resulting .py
       files.

    Args:
      name: The name, should end with "_py_pb2".
      srcs: The .proto files.
      deps: Any reverb_cc_proto_library targets.
      **kwargs: Any additional args for the cc_library rule.
    """
    gen_srcs = [_removesuffix(x, ".proto") + "_pb2.py" for x in srcs]
    src_paths = ["$(location {})".format(x) for x in srcs]
    proto_deps = []
    py_deps = []
    for x in deps:
        if x.endswith("_proto"):
            proto_deps.append(_filegroup_name(x))
        else:
            py_deps.append(x)
    native.filegroup(
        name = _filegroup_name(name),
        srcs = srcs + proto_deps,
        **kwargs
    )
    native.genrule(
        name = name + "_gen",
        srcs = srcs + proto_deps + [
            "@com_google_protobuf//:well_known_type_protos",
        ],
        outs = gen_srcs,
        tools = [
            "@com_google_protobuf//:protoc",
        ],
        cmd = """
        OUTDIR=$$(echo $(RULEDIR) | sed -E -e 's#reverb(/.*|$$)##')
        $(location @com_google_protobuf//:protoc) \
          --proto_path=external/com_google_protobuf/src \
          --proto_path=. \
          --python_out=$$OUTDIR {srcs}""".format(
            srcs = " ".join(src_paths),
        ),
    )
    py_library(
        name = name,
        srcs = gen_srcs,
        deps = py_deps,
        data = proto_deps,
        **kwargs
    )

def reverb_cc_grpc_library(
        name,
        srcs = [],
        deps = [],
        generate_mocks = False,
        **kwargs):
    """Build a grpc cc_library.

    This rule does two things:

    1) Uses protoc + grpc plugin to compile srcs to .h/.cc files.
       Also creates mock headers if requested.

    2) Creates a cc_library with name `name` building the resulting .h/.cc
       files.

    Args:
      name: The name, should end with "_cc_grpc_proto".
      srcs: The .proto files.
      deps: reverb_cc_proto_library targets.  Must include src + "_cc_proto",
        the cc_proto library, for each src in srcs.
      generate_mocks: If true, creates mock headers for each source.
      **kwargs: Any additional args for the cc_library rule.
    """
    gen_srcs = [_removesuffix(x, ".proto") + ".grpc.pb.cc" for x in srcs]
    gen_hdrs = [_removesuffix(x, ".proto") + ".grpc.pb.h" for x in srcs]
    proto_src_deps = []
    for x in deps:
        if x.endswith("_proto"):
            proto_src_deps.append(_filegroup_name(x))
    src_paths = ["$(location {})".format(x) for x in srcs]

    if generate_mocks:
        gen_mocks = [_removesuffix(x, ".proto") + "_mock.grpc.pb.h" for x in srcs]
    else:
        gen_mocks = []

    native.genrule(
        name = name + "_gen",
        srcs = srcs + proto_src_deps + [
            "@com_google_protobuf//:well_known_type_protos",
        ],
        outs = gen_srcs + gen_hdrs + gen_mocks,
        tools = [
            "@com_google_protobuf//:protoc",
            "@com_github_grpc_grpc//src/compiler:grpc_cpp_plugin",
        ],
        cmd = """
        OUTDIR=$$(echo $(RULEDIR) | sed -e 's#reverb/.*##')
        $(location @com_google_protobuf//:protoc) \
          --plugin=protoc-gen-grpc=$(location @com_github_grpc_grpc//src/compiler:grpc_cpp_plugin) \
          --proto_path=external/com_google_protobuf/src \
          --proto_path=. \
          --grpc_out={out} {srcs}""".format(
            out = "generate_mock_code=true:$$OUTDIR" if generate_mocks else "$$OUTDIR",
            srcs = " ".join(src_paths),
        ),
    )
    cc_library(
        name = name,
        srcs = gen_srcs,
        hdrs = gen_hdrs + gen_mocks,
        deps = depset(deps + ["@com_github_grpc_grpc//:grpc++_codegen_proto"]),
        **kwargs
    )

def reverb_cc_test(name, srcs, deps = [], **kwargs):
    """Reverb-specific version of cc_test.

    Args:
      name: Target name.
      srcs: Target sources.
      deps: Target deps.
      **kwargs: Additional args to cc_test.
    """
    new_deps = [
        "@com_github_grpc_grpc//:grpc++_test",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:status_matchers",
    ] + reverb_tf_deps()
    size = kwargs.pop("size", "small")
    # ponytail: cc_test 主动嵌入 Python 解释器(scoped_interpreter + import_array),
    # 链接器必须解析 Py* 符号。原硬编码 -lpython3.10 绑死系统版本,与 py 侧
    # hermetic 3.11 混搭。现改依赖 @rules_python//python/cc:current_py_cc_libs——
    # 它跟随当前 py_cc_toolchain(python-build-standalone),libpython 版本自动与
    # 解释器一致,不再硬编码系统路径。链接由 current_py_cc_libs 管;内嵌解释器
    # 的运行时 stdlib/site-packages 由 reverb_embed_py_test wrapper 管(见下)。
    deps = depset(deps + new_deps + ["@rules_python//python/cc:current_py_cc_libs"]).to_list()
    cc_test(
        name = name,
        size = size,
        copts = tf_copts(),
        srcs = srcs,
        deps = deps,
        **kwargs
    )

# ponytail: 内嵌解释器的 cc_test(scoped_interpreter + import numpy)在
# python-build-standalone 下运行时找不到 stdlib(编译期 base_prefix='/install')
# 与 numpy(@pypi site-packages 不在 cc_test runfiles)。该 wrapper 在分析期
# 从 py3 toolchain 取 interpreter 路径,生成 shell 脚本设 PYTHONHOME(指向
# standalone 树)+ PYTHONPATH(指向 @pypi//numpy site-packages),再把 standalone
# 树(py3_runtime.files)与 numpy 拉进 runfiles,最后 exec 真正的 cc_binary。
# 模式取自 pybind11_bazel commit 9d8c6b4(pybind_py_env_test)。非 Windows:wrapper
# 脚本;Windows:直链 binary(reverb 无 Windows 内嵌 cc_test 目标)。
def _embed_py_env_test_impl(ctx):
    toolchain = ctx.toolchains["@rules_python//python:toolchain_type"]
    py3_runtime = toolchain.py3_runtime
    if not py3_runtime:
        fail("No python3 runtime found in toolchain")

    binary = ctx.executable.binary

    if ctx.target_platform_has_constraint(ctx.attr._windows_constraint[platform_common.ConstraintValueInfo]):
        extension = binary.extension
        exe = ctx.actions.declare_file(ctx.label.name + ("." + extension if extension else ""))
        ctx.actions.symlink(output = exe, target_file = binary, is_executable = True)
        return [DefaultInfo(
            executable = exe,
            runfiles = ctx.runfiles(files = [exe])
            .merge(ctx.attr.binary[DefaultInfo].default_runfiles)
            .merge(ctx.runfiles(transitive_files = py3_runtime.files))
            .merge(ctx.attr.numpy[DefaultInfo].default_runfiles),
        )]

    interpreter = py3_runtime.interpreter

    # 从 @pypi//numpy:numpy(py_library,多文件)里找 numpy/__init__.py,
    # 其父目录 = site-packages。不用 allow_single_file(它不是单文件)。
    numpy_files = ctx.attr.numpy[DefaultInfo].files.to_list()
    numpy_init = None
    for f in numpy_files:
        if f.short_path.endswith("/numpy/__init__.py"):
            numpy_init = f
            break
    if numpy_init == None:
        fail("Could not find numpy/__init__.py in @pypi//numpy")
    numpy_site_packages = numpy_init.short_path
    numpy_site_packages = numpy_site_packages[:numpy_site_packages.rfind("/numpy/__init__.py")]

    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    content = "#!/bin/bash\n"
    content += "set -euo pipefail\n"
    content += "if [ -z \"${RUNFILES_DIR:-}\" ]; then\n"
    content += "  if [ -d \"$0.runfiles\" ]; then\n"
    content += "    RUNFILES_DIR=\"$0.runfiles\"\n"
    content += "  else\n"
    content += "    RUNFILES_DIR=\"$(dirname \"$0\")/../..\"\n"
    content += "  fi\n"
    content += "fi\n"
    # PYTHONHOME = standalone 树根(interpreter 的 bin 的父目录)。
    content += "INTERPRETER=\"$RUNFILES_DIR/" + ctx.workspace_name + "/" + interpreter.short_path + "\"\n"
    content += "if [ ! -f \"$INTERPRETER\" ]; then\n"
    content += "  INTERPRETER=$(find \"$RUNFILES_DIR\" -path \"*/" + interpreter.short_path + "\" | head -n1)\n"
    content += "fi\n"
    content += "export PYTHONHOME=$(dirname $(dirname $(readlink -f \"$INTERPRETER\")))\n"
    # PYTHONPATH = numpy 的 site-packages 目录(numpy 自带 .libs,无传递运行时依赖)。
    content += "NUMPY_SP=\"$RUNFILES_DIR/" + ctx.workspace_name + "/" + numpy_site_packages + "\"\n"
    content += "if [ ! -d \"$NUMPY_SP\" ]; then\n"
    content += "  NUMPY_SP=$(find \"$RUNFILES_DIR\" -type d -path \"*/" + numpy_site_packages + "\" | head -n1)\n"
    content += "fi\n"
    content += "export PYTHONPATH=\"${PYTHONPATH:+$PYTHONPATH:}$NUMPY_SP\"\n"
    content += "BIN=\"$RUNFILES_DIR/" + ctx.workspace_name + "/" + binary.short_path + "\"\n"
    content += "if [ ! -f \"$BIN\" ]; then\n"
    content += "  BIN=$(find \"$RUNFILES_DIR\" -path \"*/" + binary.short_path + "\" | head -n1)\n"
    content += "fi\n"
    content += "exec \"$BIN\" \"$@\"\n"
    ctx.actions.write(script, content, is_executable = True)

    runfiles = ctx.runfiles(files = [script, binary])
    runfiles = runfiles.merge(ctx.attr.binary[DefaultInfo].default_runfiles)
    runfiles = runfiles.merge(ctx.runfiles(transitive_files = py3_runtime.files))
    runfiles = runfiles.merge(ctx.attr.numpy[DefaultInfo].default_runfiles)
    return [DefaultInfo(executable = script, runfiles = runfiles)]

_embed_py_env_test = rule(
    implementation = _embed_py_env_test_impl,
    test = True,
    attrs = {
        "binary": attr.label(executable = True, cfg = "target", mandatory = True),
        "numpy": attr.label(
            default = "@pypi//numpy:numpy",
        ),
        "_windows_constraint": attr.label(default = "@platforms//os:windows"),
    },
    toolchains = ["@rules_python//python:toolchain_type"],
)

def reverb_embed_py_test(name, binary, size = "small", **kwargs):
    """Wrap a cc_binary 内嵌 Python 的测试,设 PYTHONHOME+PYTHONPATH。

    仅给真正 Py_Initialize 的 cc_test 用(tensor_proxy_test)。其余 cc_test
    虽传递链接 libpython(经 :tensor_proxy/:chunker)但不初始化解释器,无需此 wrapper。

    Args:
      name: 测试目标名。
      binary: 已构建的 cc_binary(:name_bin),deps 由调用方精确指定。
      size: 传递给 _embed_py_env_test。
      **kwargs: 额外参数(如 tags)。
    """
    _embed_py_env_test(
        name = name,
        binary = binary,
        size = size,
        **kwargs
    )

def reverb_gen_op_wrapper_py(name, out, kernel_lib, ops_lib = None, linkopts = [], **kwargs):
    """Generates the py_library `name` with a data dep on the ops in kernel_lib.

    The resulting py_library creates file `$out`, and has a dependency on a
    symbolic library called lib{$name}_gen_op.so, which contains the kernels
    and ops and can be loaded via `tf.load_op_library`.

    Args:
      name: The name of the py_library.
      out: The name of the python file.  Use "gen_{name}_ops.py".
      kernel_lib: A cc_kernel_library kernel target to generate for.
      ops_lib: A cc_kernel_library ops target to generate for.
      linkopts: Forwarded to the `cc_binary` internal target.
      **kwargs: Any args to the `cc_binary` and `py_library` internal rules.
    """
    if not out.endswith(".py"):
        fail("Argument out must end with '.py', but saw: {}".format(out))

    module_name = "lib{}_gen_op".format(name)
    exported_symbols_file = "%s-exported-symbols.lds" % module_name

    # gen_client_ops -> reverb_client
    symbol = "reverb_{}".format(name.split("_")[1])
    native.genrule(
        name = module_name + "_exported_symbols",
        outs = [exported_symbols_file],
        cmd = "echo '*%s*' >$@" % symbol,
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
    )
    version_script_file = "%s-version-script.lds" % module_name
    native.genrule(
        name = module_name + "_version_script",
        outs = [version_script_file],
        cmd = "echo '{global:\n *%s*;\n local: *;};' >$@" % symbol,
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
    )
    cc_binary(
        name = "{}.so".format(module_name),
        deps = [kernel_lib] + [ops_lib] if ops_lib else [],
        copts = tf_copts() + [
            "-fno-strict-aliasing",  # allow a wider range of code [aliasing] to compile.
            "-fvisibility=hidden",  # avoid symbol clashes between DSOs.
        ],
        additional_linker_inputs = [
            exported_symbols_file,
            version_script_file,
        ],
        dynamic_deps = ["//reverb:libreverb"],
        linkshared = 1,
        linkopts = linkopts + _rpath_linkopts(module_name) + select({
            "@platforms//os:macos": [
                "-Wl,-exported_symbols_list,$(location %s)" % exported_symbols_file,
            ],
            "//conditions:default": [
                "-Wl,--version-script,$(location %s)" % version_script_file,
            ],
        }),
        **kwargs
    )
    native.genrule(
        name = "{}_genrule".format(out),
        outs = [out],
        cmd = """echo 'import tensorflow as _tf
from reverb.platform.default import load_op_library as _load_op_library

try:
  _reverb_gen_op = _tf.load_op_library(
    _tf.compat.v1.resource_loader.get_path_to_datafile("lib{}_gen_op.so"))
except _tf.errors.NotFoundError as e:
  _load_op_library.reraise_wrapped_error(e)
_locals = locals()
for k in dir(_reverb_gen_op):
  _locals[k] = getattr(_reverb_gen_op, k)
del _locals' > $@""".format(name),
    )
    deps = kwargs.pop("deps", [])
    deps.append("//reverb/platform/default:load_op_library")
    native.py_library(
        name = name,
        srcs = [out],
        data = [":lib{}_gen_op.so".format(name)],
        deps = deps,
        **kwargs
    )

def reverb_py_proto_deps():
    return []

def reverb_pytype_library(deps = [], **kwargs):
    if "strict_deps" in kwargs:
        kwargs.pop("strict_deps")
    py_library(
        deps = deps + reverb_py_standard_imports() + reverb_py_proto_deps(),
        **kwargs
    )

reverb_pytype_strict_library = reverb_pytype_library

def reverb_pytype_binary(deps = [], **kwargs):
    if "strict_deps" in kwargs:
        kwargs.pop("strict_deps")
    py_binary(
        deps = deps + reverb_py_standard_imports() + reverb_py_proto_deps(),
        **kwargs
    )

reverb_pytype_strict_binary = reverb_pytype_binary

def _make_search_paths(prefix, levels_to_root):
    return ",".join(
        [
            "-rpath,%s/%s" % (prefix, "/".join([".."] * search_level))
            for search_level in range(levels_to_root + 1)
        ],
    )

def _rpath_linkopts(name):
    # Search parent directories up to the TensorFlow root directory for shared
    # object dependencies, even if this op shared object is deeply nested
    # (e.g. tensorflow/contrib/package:python/ops/_op_lib.so). tensorflow/ is then
    # the root and tensorflow/libtensorflow_framework.so should exist when
    # deployed. Other shared object dependencies (e.g. shared between contrib/
    # ops) are picked up as long as they are in either the same or a parent
    # directory in the tensorflow/ tree.
    levels_to_root = native.package_name().count("/") + name.count("/")
    return select({
        "@platforms//os:macos": [
            "-Wl,%s" % (_make_search_paths("@loader_path", levels_to_root),),
        ],
        "//conditions:default": [
            "-Wl,%s" % (_make_search_paths("$$ORIGIN", levels_to_root),),
        ],
    })

def reverb_pybind_extension(
        name,
        srcs,
        module_name,
        hdrs = [],
        features = [],
        srcs_version = "PY3",
        data = [],
        copts = [],
        linkopts = [],
        deps = [],
        defines = [],
        visibility = None,
        testonly = None,
        licenses = None,
        compatible_with = None,
        restricted_to = None,
        deprecation = None,
        pytype_srcs = None):
    """Builds a generic Python extension module.

    The module can be loaded in python by performing "import ${name}.".

    Args:
      name: Name.
      srcs: cc files.
      module_name: The name of the hidden module.  It should be different
        from `name`, and *must* match the MODULE declaration in the .cc file.
      hdrs: h files.
      features: see bazel docs.
      srcs_version: srcs_version for py_library.
      data: data deps.
      copts: compilation opts.
      linkopts: linking opts.
      deps: cc_library deps.
      defines: cc_library defines.
      visibility: visibility.
      testonly: whether the rule is testonly.
      licenses: see bazel docs.
      compatible_with: see bazel docs.
      restricted_to: see bazel docs.
      deprecation:  see bazel docs.
      pytype_srcs: Unused list of pytype stub files.
    """
    if name == module_name:
        fail(
            "Must have name != module_name ({} vs. {}) because the python ".format(name, module_name) +
            "wrapper $name.py needs to add extra logic loading tensorflow.",
        )
    py_file = "%s.py" % name
    so_file = "%s.so" % module_name
    pyd_file = "%s.pyd" % module_name
    symbol = "init%s" % module_name
    symbol2 = "init_%s" % module_name
    symbol3 = "PyInit_%s" % module_name
    exported_symbols_file = "%s-exported-symbols.lds" % module_name
    version_script_file = "%s-version-script.lds" % module_name
    native.genrule(
        name = module_name + "_exported_symbols",
        outs = [exported_symbols_file],
        cmd = "echo '_%s\n' >$@" % (symbol3),
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
        testonly = testonly,
    )
    native.genrule(
        name = module_name + "_version_script",
        outs = [version_script_file],
        cmd = "echo '{global:\n %s;\n local: *;};' >$@" % (symbol3),
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
        testonly = testonly,
    )
    cc_binary(
        name = so_file,
        srcs = srcs + hdrs,
        data = data,
        copts = copts + [
            "-fno-strict-aliasing",  # allow a wider range of code [aliasing] to compile.
            "-fexceptions",  # pybind relies on exceptions, required to compile.
            "-fvisibility=hidden",  # avoid pybind symbol clashes between DSOs.
        ],
        linkopts = linkopts + _rpath_linkopts(module_name) + select({
            "@platforms//os:macos": [
                "-Wl,-exported_symbols_list,$(location %s)" % exported_symbols_file,
            ],
            "//conditions:default": [
                "-Wl,--version-script,$(location %s)" % version_script_file,
            ],
        }),
        deps = deps,
        additional_linker_inputs = [
            exported_symbols_file,
            version_script_file,
        ],
        dynamic_deps = ["//reverb:libreverb"],
        defines = defines,
        features = features + ["-use_header_modules"],
        linkshared = 1,
        testonly = testonly,
        licenses = licenses,
        visibility = visibility,
        deprecation = deprecation,
        restricted_to = restricted_to,
        compatible_with = compatible_with,
    )
    native.genrule(
        name = module_name + "_pyd_copy",
        srcs = [so_file],
        outs = [pyd_file],
        cmd = "cp $< $@",
        output_to_bindir = True,
        visibility = visibility,
        deprecation = deprecation,
        restricted_to = restricted_to,
        compatible_with = compatible_with,
    )
    native.genrule(
        name = name + "_py_file",
        outs = [py_file],
        # ponytail: 去 `import tensorflow` —— libpybind.so 已无 TF 符号(内嵌
        # numpy 模式),无需 ensure_tf_install / load_op_library 的符号检查,
        # 使 `import reverb.pybind` 在无 TF 环境下也可用。
        cmd = "echo 'try:\n  from .%s import *\nexcept ImportError as e:\n  raise e' >$@" % module_name,
        output_licenses = ["unencumbered"],
        visibility = visibility,
        testonly = testonly,
    )
    py_library(
        name = name,
        data = [so_file, "//reverb:libreverb"],
        deps = [],
        srcs = [py_file],
        srcs_version = srcs_version,
        licenses = licenses,
        testonly = testonly,
        visibility = visibility,
        deprecation = deprecation,
        restricted_to = restricted_to,
        compatible_with = compatible_with,
    )

def reverb_py_standard_imports():
    # ponytail: tf_nightly 已移除——内嵌 numpy 模式不依赖 TF。
    # 若恢复 gRPC TF ops 集成,需重新加 @pypi//tf_nightly。
    return [
        "@pypi//absl_py",
        "@pypi//dm_tree",
        "@pypi//portpicker",
        "@pypi//numpy",
        "@pypi//packaging",
        "@pypi//protobuf",
    ]

def reverb_py_test(
        name,
        srcs = [],
        deps = [],
        paropts = [],
        python_version = "PY3",
        **kwargs):
    size = kwargs.pop("size", "small")
    if "enable_dashboard" in kwargs:
        kwargs.pop("enable_dashboard")
    py_test(
        name = name,
        size = size,
        srcs = srcs,
        deps = deps + reverb_py_standard_imports() + reverb_py_proto_deps(),
        python_version = python_version,
        **kwargs
    )
    return

def reverb_pybind_deps():
    return [
        "@pybind11",
        "@pypi//numpy:numpy_headers",
    ]

def reverb_tf_ops_visibility():
    return [
        "//reverb:__subpackages__",
    ]

def reverb_tf_deps():
    # ponytail: TF 依赖已移除——内嵌 numpy 模式不需要 TF。保留函数签名
    # 兼容 reverb_cc_library/reverb_cc_test 的调用点。client.cc 等仍含 TF
    # include 的文件不在编译路径内,故返回空列表不破坏内嵌链路。
    return []

def reverb_grpc_deps():
    return ["@com_github_grpc_grpc//:grpc++"]
