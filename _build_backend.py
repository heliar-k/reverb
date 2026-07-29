# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PEP 517 build backend: 从源码安装 reverb via pip install . / uv add git+...

后端 shell out 调 bazel 编译 C++ 产物 (.so + _pb2.py),组装 wheel。
只用 stdlib;bazel/patchelf 作为子进程调用。不依赖 hatchling/setuptools/wheel。

混合模式:若 bazel-bin/reverb/libpybind.so 已存在且链接的 libpython 版本
匹配当前解释器,复用之;否则 bazel build。
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
NAME = "dm-reverb-numpy"  # PyPI 规范化名
DIST_NAME = "dm_reverb_numpy"  # wheel 文件名/dist-info 用下划线
VERSION = "0.15.0"
ENTRY_POINT = "reverb.server_executable.server_main:app_run_main"
SUMMARY = (
    "Reverb is an efficient and easy-to-use data storage and transport system "
    "designed for machine learning research."
)
BAZEL_TARGETS = [
    "//reverb/pip_package:cc_deps",  # libpybind.so, libreverb.so
    "//reverb/pip_package:py_deps",  # .py 源码 + 生成 _pb2.py + pybind.py
]
# 不在 bazel target 里、但需要进 wheel 的源文件。
EXTRA_FILES = [
    ("reverb/pybind.pyi", "reverb/pybind.pyi"),
    ("LICENSE", "LICENSE"),
    ("README.md", "README.md"),
]


def _target_py_version() -> str:
    """当前解释器版本,作为 HERMETIC_PYTHON_VERSION。"""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


# pybind11 模块不直接链接 libpython (Python 符号运行时由宿主解释器注入,
# readelf NEEDED 里永远没有 libpythonX.Y.so), 所以无法从 .so 的动态段探测
# 编译时的 Python 版本。改用 sidecar 标记文件记录上次构建的版本。
# ponytail: 单文件 marker, 升级路径 = 若 bazel 未来把 py 版本编码进产物路径,
# 改读路径即可删此文件。
_PYVER_MARKER = REPO_ROOT / ".bazel_pyver"


def _artifacts_valid() -> bool:
    """bazel-bin/reverb/libpybind.so 存在且 sidecar 标记的版本匹配 → 可复用。"""
    so = REPO_ROOT / "bazel-bin" / "reverb" / "libpybind.so"
    # bazel-bin 是符号链接;resolve() 解析到真实路径再 exists()。
    if not so.exists():
        return False
    try:
        return _PYVER_MARKER.read_text().strip() == _target_py_version()
    except OSError:
        return False


def _run_bazel(args: list[str], **kw) -> subprocess.CompletedProcess:
    """统一 bazel 调用入口:无 bazel 时给友好提示而非裸 FileNotFoundError。"""
    py_ver = _target_py_version()
    cmd = ["bazel", *args, f"--repo_env=HERMETIC_PYTHON_VERSION={py_ver}"]
    try:
        return subprocess.run(cmd, cwd=REPO_ROOT, **kw)
    except FileNotFoundError:
        raise SystemExit(
            "reverb 源码安装需要 bazel,但 PATH 上找不到。"
            "请安装 bazel (https://bazel.build/install) 或用预编译 wheel。"
        ) from None


def _ensure_artifacts() -> None:
    """混合模式:有可用产物就复用,否则 bazel build。"""
    if _artifacts_valid():
        return
    try:
        _run_bazel(["build", *BAZEL_TARGETS], check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"bazel build 失败 (exit {e.returncode})") from e
    _PYVER_MARKER.write_text(_target_py_version())


def _collect_artifacts() -> list[tuple[str, str]]:
    """cquery 收集 bazel 产物,返回 [(绝对源路径, wheel 内相对路径)]。

    过滤 external/(numpy/absl 等第三方 .py,由 pip 依赖提供)。
    生成的文件路径以 bazel-out/.../bin/ 开头,剥离该前缀。
    """
    files: list[str] = []
    for target in BAZEL_TARGETS:
        out = _run_bazel(
            ["cquery", "--output=files", target],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        files.extend(out.strip().splitlines())

    artifacts = []
    for f in files:
        f = f.strip()
        if not f or "external" in f:
            continue
        abs_path = str(REPO_ROOT / f)
        rel = f
        if rel.startswith("bazel-out"):
            rel = rel[rel.index("bin") + 4 :]  # 剥离 bazel-out/.../bin/
        artifacts.append((abs_path, rel))
    return artifacts


def _patch_so(wheel_dir: Path) -> None:
    """收缩 libpybind.so 的 rpath(指向 bazel _solib 的残留路径无效)。

    patchelf 缺失时跳过:$ORIGIN/ 仍能定位同目录的 libreverb.so。
    """
    so = wheel_dir / "reverb" / "libpybind.so"
    if not so.exists() or not shutil.which("patchelf"):
        return
    rpath = (
        subprocess.check_output(["patchelf", "--print-rpath", str(so)]).decode().strip()
    )
    subprocess.run(["patchelf", "--set-rpath", rpath, str(so)], check=True)
    subprocess.run(["patchelf", "--shrink-rpath", str(so)], check=True)


def _platform_tag() -> str:
    """wheel 平台标签。仅支持 linux x86_64 / macos arm64。"""
    if sys.platform == "darwin":
        # macOS ARM;x86 非用户目标,不处理。
        return "macosx_12_0_arm64"
    return "linux_x86_64"


def _metadata() -> str:
    """PEP 621 core metadata (METADATA 文件内容)。"""
    return (
        f"Metadata-Version: 2.1\n"
        f"Name: {NAME}\n"
        f"Version: {VERSION}\n"
        f"Summary: {SUMMARY}\n"
        f"License: Apache-2.0\n"
        f"Requires-Python: >=3.9\n"
        f"Requires-Dist: absl-py\n"
        f"Requires-Dist: dm-tree\n"
        f"Requires-Dist: portpicker\n"
        f"Requires-Dist: numpy\n"
        # >=5.27: generated _pb2.py uses google.protobuf.runtime_version (5.27+).
        f"Requires-Dist: protobuf >= 5.27\n"
    )


def _wheel_meta() -> str:
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return (
        "Wheel-Version: 1.0\n"
        "Generator: reverb_build_backend\n"
        "Root-Is-Purelib: false\n"
        f"Tag: {py_tag}-{py_tag}-{_platform_tag()}\n"
    )


def _record_line(data: bytes, arcname: str) -> str:
    """生成 RECORD 行:arcname,sha256=base64url,size。"""
    h = hashlib.sha256(data).digest()
    import base64

    digest = base64.urlsafe_b64encode(h).rstrip(b"=").decode()
    return f"{arcname},sha256={digest},{len(data)}"


def _assemble_wheel(wheel_dir: Path, output_dir: str) -> str:
    """把 wheel_dir 打包成 wheel zip,写到 output_dir,返回文件名。"""
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    tag = f"{py_tag}-{py_tag}-{_platform_tag()}"
    wheel_name = f"{DIST_NAME}-{VERSION}-{tag}.whl"
    wheel_path = os.path.join(output_dir, wheel_name)
    dist_info = f"{DIST_NAME}-{VERSION}.dist-info"

    records: list[str] = []
    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 包文件
        for root, _dirs, files in os.walk(wheel_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, wheel_dir)
                with open(fpath, "rb") as fh:
                    data = fh.read()
                zf.writestr(arcname, data)
                records.append(_record_line(data, arcname))
        # dist-info 元数据
        for meta_file, content in [
            ("METADATA", _metadata()),
            ("WHEEL", _wheel_meta()),
            ("entry_points.txt", f"[console_scripts]\nreverb_server = {ENTRY_POINT}\n"),
        ]:
            arcname = f"{dist_info}/{meta_file}"
            zf.writestr(arcname, content)
            records.append(_record_line(content.encode(), arcname))
        # RECORD 自身(无 hash)
        record_arcname = f"{dist_info}/RECORD"
        record_content = "\n".join(records) + f"\n{record_arcname},,\n"
        zf.writestr(record_arcname, record_content)
    return wheel_name


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """PEP 517: 构建 wheel。"""
    _ensure_artifacts()
    with tempfile.TemporaryDirectory(prefix="reverb_wheel_") as tmp:
        tmp_dir = Path(tmp)
        # 复制 bazel 产物
        for abs_path, rel_path in _collect_artifacts():
            dst = tmp_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(abs_path, dst)
            os.chmod(dst, 0o644)
        # 复制额外文件
        for src, dst_rel in EXTRA_FILES:
            src_path = REPO_ROOT / src
            if src_path.exists():
                dst = tmp_dir / dst_rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_path, dst)
        # patch .so rpath
        _patch_so(tmp_dir)
        # 组装 wheel
        return _assemble_wheel(tmp_dir, wheel_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """PEP 660: 可编辑安装。

    把生成的产物 (.so/_pb2.py/pybind.py) 复制到源树,使源树直接可 import;
    editable wheel 仅含 .pth 指向源树根。
    """
    _ensure_artifacts()
    generated_suffixes = (".so", "_pb2.py", "pybind.py")
    for abs_path, rel_path in _collect_artifacts():
        if rel_path.endswith(generated_suffixes):
            dst = REPO_ROOT / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(abs_path, dst)
            os.chmod(dst, 0o644)
    # editable wheel: 一个 .pth 指向源树根 + dist-info
    py_tag = "py3-none-any"
    wheel_name = f"{DIST_NAME}-{VERSION}-{py_tag}.whl"
    wheel_path = os.path.join(wheel_directory, wheel_name)
    dist_info = f"{DIST_NAME}-{VERSION}.dist-info"
    pth_content = str(REPO_ROOT) + "\n"
    records: list[str] = []
    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:
        pth_arcname = f"__editable__.{DIST_NAME}-{VERSION}.pth"
        zf.writestr(pth_arcname, pth_content)
        records.append(_record_line(pth_content.encode(), pth_arcname))
        for meta_file, content in [
            ("METADATA", _metadata()),
            (
                "WHEEL",
                _wheel_meta().replace(
                    "Root-Is-Purelib: false", "Root-Is-Purelib: true"
                ),
            ),
        ]:
            arcname = f"{dist_info}/{meta_file}"
            zf.writestr(arcname, content)
            records.append(_record_line(content.encode(), arcname))
        record_arcname = f"{dist_info}/RECORD"
        record_content = "\n".join(records) + f"\n{record_arcname},,\n"
        zf.writestr(record_arcname, record_content)
    return wheel_name
