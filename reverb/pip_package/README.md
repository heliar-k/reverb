# Building reverb from source

This document describes how to build and install Reverb from source.

## System requirements

Reverb supports building on Linux x86_64 and macOS Apple Silicon (ARM). You may
be able to build Reverb on other platforms and architectures but they are not
tested.

## Install dependencies

Building Reverb from source requires the following dependencies to be installed
on your system:

* **Bazel**. Reverb uses bazel as the build system for Python extensions. The
    bazel version used in CI can be found in `.bazelversion`. You may want to
    install [bazelisk](https://github.com/bazelbuild/bazelisk) as the bazel
    binary which automatically ensures that you use the correct version of
    bazel.

* **patchelf** (Linux only). Used to shrink the `.so` rpath after the bazel
    build so the wheel is portable. Optional — if absent, the build continues
    and `$ORIGIN` still resolves `libreverb.so` at runtime.

## Install from source (pip install . / uv add)

The repository ships a custom PEP 517 build backend (`_build_backend.py` +
root `pyproject.toml`) that shells out to bazel to compile the C++ extensions
and assembles a wheel using only the Python standard library. This lets you
install directly from source without a pre-built wheel or an internal PyPI.

> [!NOTE]
> The backend uses only the Python standard library (`requires = []`),
> but invokes `bazel` (and optionally `patchelf`) as subprocesses, so both
> must be on `PATH`. Build isolation works normally — the isolated env has
> stdlib, and bazel/patchelf are system tools reachable from it.

### Clone and install

```shell
git clone <repo-url> reverb
cd reverb
pip install .          # regular install
pip install -e .       # editable install (dev)
```

Or with uv:

```shell
uv pip install .       # regular
uv pip install -e .    # editable
```

### Add as a dependency from git

In another project:

```shell
uv add "dm-reverb-numpy @ git+ssh://git@gitlab.fuxi.netease.com:2222/guankai1/reverb.git"
```

or in `pyproject.toml`:

```toml
dependencies = [
  "dm-reverb-numpy @ git+ssh://git@gitlab.fuxi.netease.com:2222/guankai1/reverb.git",
]
```

### How it works

The backend (`_build_backend.py`):

1. Checks `bazel-bin/reverb/libpybind.so`; if it exists and links the matching
    `libpythonX.Y`, reuses it. Otherwise runs
    `bazel build //reverb/pip_package:{cc_deps,py_deps}` with
    `--repo_env=HERMETIC_PYTHON_VERSION=<target>`.
2. Collects outputs via `bazel cquery --output=files`, filtering out
    `external/` paths (numpy/absl/etc. come from pip dependencies, not the
    wheel).
3. Lays out the wheel (`.so` + `_pb2.py` + `.py` sources), shrinks the
    `libpybind.so` rpath with patchelf, and writes the wheel zip with
    `METADATA`/`WHEEL`/`RECORD`/`entry_points.txt`.

For editable installs, the generated artifacts are copied into the source tree
(gitignored) so `import reverb` resolves directly to the source.

> [!NOTE]
> The `.so` is Python-version-specific (links `libpythonX.Y`). Installing for
> Python 3.12 builds against hermetic Python 3.12; the supported range is
> 3.9–3.13.

## Update PyPI requirements

The bazel build is set up to use pre-built Python packages specified in
`reverb/pip_package/requirements_lock*.txt`.

You can update the requirements by modifying
`reverb/pip_package/requirements.in` and running:

```shell
bazel run --repo_env=HERMETIC_PYTHON_VERSION=3.12 //reverb/pip_package:requirements.update
```

which will update the locked requirements for Python 3.12.

## Notes on dependencies

At runtime Reverb depends only on `absl-py`, `dm-tree`, `portpicker`, `numpy`,
`packaging`, and `protobuf`; TensorFlow is **not**** required.

Dependencies such as abseil-cpp, grpc and protobuf are resolved via Bazel
(`WORKSPACE` / `MODULE.bazel`) at build time.
