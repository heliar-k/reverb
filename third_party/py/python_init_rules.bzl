"""rules_cc / com_google_protobuf / rules_python registration (reverb-local).

ponytail: 替代 @local_xla//third_party/py:python_init_rules.bzl。urls/sha 与
原版一致;patches vendored 到 third_party/(py|patches)。此文件不 load
@rules_python(@rules_python 由本文件的 python_init_rules() 注册,需在
调用 python_init_repositories.bzl 之前运行)。
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _tf_mirror_urls(url):
    if not url.startswith("https://"):
        return [url]
    return [
        "https://storage.googleapis.com/mirror.tensorflow.org/%s" % url[8:],
        url,
    ]

def python_init_rules(extra_rules_python_patches = []):
    """Defines rules_cc, com_google_protobuf, rules_python (no toolchain setup)."""
    http_archive(
        name = "rules_cc",
        urls = ["https://github.com/bazelbuild/rules_cc/archive/refs/tags/0.1.0.tar.gz"],
        strip_prefix = "rules_cc-0.1.0",
        sha256 = "4b12149a041ddfb8306a8fd0e904e39d673552ce82e4296e96fac9cbf0780e59",
        patches = [
            Label("//third_party/py:rules_cc_protobuf.patch"),
        ],
        patch_args = ["-p1"],
    )

    http_archive(
        name = "com_google_protobuf",
        patches = ["//third_party/patches:protobuf.patch"],
        patch_args = ["-p1"],
        sha256 = "6e09bbc950ba60c3a7b30280210cd285af8d7d8ed5e0a6ed101c72aff22e8d88",
        strip_prefix = "protobuf-6.31.1",
        urls = _tf_mirror_urls("https://github.com/protocolbuffers/protobuf/archive/refs/tags/v6.31.1.zip"),
        repo_mapping = {
            "@abseil-cpp": "@com_google_absl",
            "@protobuf_pip_deps": "@pypi",
        },
    )

    http_archive(
        name = "rules_python",
        sha256 = "fa7dd2c6b7d63b3585028dd8a90a6cf9db83c33b250959c2ee7b583a6c130e12",
        strip_prefix = "rules_python-1.6.0",
        url = "https://github.com/bazelbuild/rules_python/releases/download/1.6.0/rules_python-1.6.0.tar.gz",
        patch_args = ["-p1"],
        patches = [
            Label("//third_party/py:rules_python_pip_version.patch"),
            Label("//third_party/py:rules_python_freethreaded.patch"),
            Label("//third_party/py:rules_python_versions.patch"),
        ] + extra_rules_python_patches,
    )
