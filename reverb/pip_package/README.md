# Building reverb from source

This document describes how to build and install Reverb from source.

## System requirements

Reverb supports building on Linux x86_64 and macOS Apple Silicon (ARM). You may
be able to build Reverb on other platforms and architectures but they are not
tested.

## Install dependencies

Building Reverb Python wheels requires the following dependencies to be
installed on your system:

* **Bazel**. Reverb uses bazel as the build system for Python extensions. The
    bazel version used in CI can be found in `.bazelversion`. You may want to
    install [bazelisk](https://github.com/bazelbuild/bazelisk) as the bazel
    binary which automatically ensures that you use the correct version of
    bazel.

* **patchelf** (Linux only). Used to shrink the `.so` rpath after the bazel
    build so the wheel is portable. Optional — if absent, the build continues
    and `$ORIGIN` still resolves `libreverb.so` at runtime.

* **uv**. Only needed for the `oss_build.sh` distribution-build path (wheel
    repair + test venvs). The `pip install .` path below does not require uv.

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
uv add "dm-reverb @ git+ssh://git@gitlab.fuxi.netease.com:2222/guankai1/reverb.git"
```

or in `pyproject.toml`:

```toml
dependencies = [
  "dm-reverb @ git+ssh://git@gitlab.fuxi.netease.com:2222/guankai1/reverb.git",
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

## Build distribution wheels with oss_build.sh

> [!NOTE] This fork runs **without TensorFlow at runtime**: the installed
> package depends only on numpy (see `reverb/pip_package/requirements.in`).
> The `oss_build.sh` / wheel-packaging scripts still carry historical
> TensorFlow version metadata (`reverb_version.bzl`, `--tf-version` args) that
> has not yet been fully cleaned up, but no TF package is required to import
> or use Reverb.

For building manylinux-compliant distribution wheels (with auditwheel repair),
use `oss_build.sh`. This is the distribution path; for local source installs
prefer `pip install .` above.

You can build Reverb either as the release package `dm_reverb` or the nightly
package `dm_reverb_nightly`. We provide a shell script `oss_build.sh` that
automates building and testing Reverb wheels.

To build the wheel, run the following command from the root of the repository:

```shell
bash oss_build.sh --python '3.11'
```

The script supports the following flags:

* `--python`. The Python version to build the wheel for. You can specify
    multiple Python versions with `--python '3.11 3.12'`.
* `--release`. Whether to build the nightly or release package. This
    determines the name of the wheel (`dm_reverb` or `dm_reverb_nightly`).
* `--python_tests`. Whether to run the Python tests by installing Reverb in a
    virtual environment. Use `--python_tests true` for running the test and
    `--python_tests false` for skipping the tests.
* `--output_dir`. Location to store the wheels. Defaults to `dist`.

You can then install the wheel with:

```shell
python3 -m pip install '<path to .whl file>'
```

## Build with bazel

You can also build the wheels with Bazel:

```shell
# Build release wheel
bazel build \
  --repo_env=HERMETIC_PYTHON_VERSION=3.12 \
  --repo_env=WHEEL_NAME=dm_reverb \
  --repo_env=ML_WHEEL_TYPE=release \
  //reverb/pip_package:wheel

# Build nightly wheel
bazel build \
  --repo_env=HERMETIC_PYTHON_VERSION=3.12 \
  --repo_env=WHEEL_NAME=dm_reverb_nightly \
  --repo_env=ML_WHEEL_TYPE=nightly \
  --repo_env=ML_WHEEL_BUILD_DATE=`date '+%Y%m%d'` \
  //reverb/pip_package:wheel
```

The `--repo_env=HERMETIC_PYTHON_VERSION=3.12` controls the hermetic Python
version used for building the wheel. This should be set to the Python version
that you intend to use the wheel for.

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

At runtime Reverb depends only on numpy (plus `absl-py`, `dm-tree`,
`portpicker`, `packaging`); TensorFlow is **not** required. The historical
packaging scripts still reference TensorFlow version metadata
(`reverb/pip_package/reverb_version.bzl`, the `--tf-version` argument in
`reverb_wheel.bzl`) carried over from upstream; cleaning these up is tracked as
part of the build-system de-TF work and does not affect installed usage.

Dependencies such as abseil-cpp, grpc and protobuf are resolved via Bazel
(`WORKSPACE` / `MODULE.bazel`) at build time.
