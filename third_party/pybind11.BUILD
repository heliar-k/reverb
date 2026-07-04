package(default_visibility = ["//visibility:public"])

# ponytail: reverb 自有 BUILD,覆盖 tf_workspace 注册的 @local_xla 版本,去掉 @local_xla 依赖。
# Python.h 来自 rules_python 的 py_cc 工具链(由 local_xla python_init_toolchains 注册),
# 而非 @local_xla//third_party/python_runtime:headers,故 @pybind11 闭包不再含 @local_xla。

cc_library(
    name = "pybind11",
    hdrs = glob(
        include = [
            "include/pybind11/*.h",
            "include/pybind11/detail/*.h",
        ],
        exclude = [
            "include/pybind11/common.h",
            "include/pybind11/eigen.h",
        ],
    ),
    copts = select({
        ":msvc_compiler": [],
        "//conditions:default": [
            "-fexceptions",
            "-Wno-undefined-inline",
            "-Wno-pragma-once-outside-header",
        ],
    }),
    includes = ["include"],
    strip_include_prefix = "include",
    deps = [
        "@rules_python//python/cc:current_py_cc_headers",
    ],
)

# Needed by copts select above (mirrors upstream @local_xla BUILD).
config_setting(
    name = "msvc_compiler",
    flag_values = {"@bazel_tools//tools/cpp:compiler": "msvc-cl"},
    visibility = ["//visibility:public"],
)
