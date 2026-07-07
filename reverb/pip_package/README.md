# Building reverb from source

This document describes how to build Reverb wheels from source.

## System requirements

Reverb supports building on Linux x86_64 and macOS Apple Silicon (ARM). You may
be able to build Reverb on other platforms and architectures but they are not
tested.

## Install dependencies

Building Reverb Python wheels requires the following dependencies to be
installed on your system:

*   **Bazel**. Reverb uses bazel as the build system for Python extensions. The
    bazel version used in CI can be found in `.bazelversion`. You may want to
    install [bazelisk](https://github.com/bazelbuild/bazelisk) as the bazel
    binary which automatically ensures that you use the correct version of
    bazel.

*   **uv**. Reverb uses [uv](https://docs.astral.sh/uv/) for managing Python
    virtual environments and several tools used for creating wheels. Follow the
    [installation instruction](https://docs.astral.sh/uv/getting-started/installation/)
    to set up uv on your system.

## Build wheels with oss_build.sh

> [!NOTE] This fork runs **without TensorFlow at runtime**: the installed
> package depends only on numpy (see `reverb/pip_package/requirements.in`).
> The `oss_build.sh` / wheel-packaging scripts still carry historical
> TensorFlow version metadata (`reverb_version.bzl`, `--tf-version` args) that
> has not yet been fully cleaned up, but no TF package is required to import
> or use Reverb.

You can build Reverb either as the release package `dm_reverb` or the nightly
package `dm_reverb_nightly`. We provide a shell script `oss_build.sh` that
automates building and testing Reverb wheels.

To build the wheel, run the following command from the root of the repository:

```shell
bash oss_build.sh --python '3.11'
```

The script supports the following flags:

*   `--python`. The Python version to build the wheel for. You can specify
    multiple Python versions with `--python '3.11 3.12'`.
*   `--release`. Whether to build the nightly or release package. This
    determines the name of the wheel (`dm_reverb` or `dm_reverb_nightly`).
*   `--python_tests`. Whether to run the Python tests by installing Reverb in a
    virtual environment. Use `--python_tests true` for running the test and
    `--python_tests false` for skipping the tests.
*   `--output_dir`. Location to store the wheels. Defaults to `dist`.

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
