"""Main Starlark code for platform-specific build rules."""

load(
    "//reverb/cc/platform/default:build_rules.bzl",
    _reverb_cc_grpc_library = "reverb_cc_grpc_library",
    _reverb_cc_library = "reverb_cc_library",
    _reverb_cc_proto_library = "reverb_cc_proto_library",
    _reverb_cc_shared_library = "reverb_cc_shared_library",
    _reverb_cc_test = "reverb_cc_test",
    _reverb_embed_py_test = "reverb_embed_py_test",
    _reverb_grpc_deps = "reverb_grpc_deps",
    _reverb_py_proto_library = "reverb_py_proto_library",
    _reverb_py_standard_imports = "reverb_py_standard_imports",
    _reverb_py_test = "reverb_py_test",
    _reverb_pybind_deps = "reverb_pybind_deps",
    _reverb_pybind_extension = "reverb_pybind_extension",
    _reverb_pytype_library = "reverb_pytype_library",
    _reverb_pytype_strict_binary = "reverb_pytype_strict_binary",
    _reverb_pytype_strict_library = "reverb_pytype_strict_library",
)

reverb_cc_library = _reverb_cc_library
reverb_cc_test = _reverb_cc_test
reverb_embed_py_test = _reverb_embed_py_test
reverb_cc_grpc_library = _reverb_cc_grpc_library
reverb_cc_proto_library = _reverb_cc_proto_library
reverb_grpc_deps = _reverb_grpc_deps
reverb_py_proto_library = _reverb_py_proto_library
reverb_py_standard_imports = _reverb_py_standard_imports
reverb_py_test = _reverb_py_test
reverb_pybind_deps = _reverb_pybind_deps
reverb_pybind_extension = _reverb_pybind_extension
reverb_pytype_library = _reverb_pytype_library
reverb_pytype_strict_library = _reverb_pytype_strict_library
reverb_pytype_strict_binary = _reverb_pytype_strict_binary
reverb_cc_shared_library = _reverb_cc_shared_library
