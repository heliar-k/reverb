#!/usr/bin/env bash
# 生成 reverb C++ 行覆盖率报告(clang source-based coverage)。
#
# 背景:reverb 用 clang source-based coverage(-fprofile-instr-generate/
# -fcoverage-mapping)而非 `bazel coverage`。手动插桩构建→跑测试拿 profraw→
# llvm-profdata merge + llvm-cov export。
#
# 用法:
#   scripts/coverage.sh                  # 跑全量 C++ 测试,生成 lcov
#   scripts/coverage.sh //reverb/cc:writer_test  # 只跑指定 target
#   scripts/coverage.sh --html           # 额外生成 HTML 报告
#
# 产出:/tmp/reverb_cov/coverage.lcov (+ coverage.html/ 若 --html)
# 依赖:系统 llvm 的 llvm-profdata/llvm-cov(`apt install llvm`)。
# ponytail: 原依赖 hermetic llvm18(bazel external,随 rules_ml_toolchain 拉入),
# 去 rules_ml_toolchain 后改用系统 llvm。

set -euo pipefail

TARGET="${1:-//reverb/cc/... //reverb/platform/...}"
GEN_HTML=false
[[ "${2:-}" == "--html" ]] && GEN_HTML=true

OUTDIR=/tmp/reverb_cov
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"

# 定位 llvm 工具:优先系统 PATH,其次 /usr/lib/llvm-*/bin
# ponytail: 去 rules_ml_toolchain 后不再有 bazel external llvm18。
LLVMBIN=""
if command -v llvm-profdata >/dev/null 2>&1; then
  LLVMBIN=$(dirname $(command -v llvm-profdata))
elif ls /usr/lib/llvm-*/bin/llvm-profdata >/dev/null 2>&1; then
  LLVMBIN=$(ls -d /usr/lib/llvm-*/bin | tail -1)
fi
if [[ -z "$LLVMBIN" || ! -x "$LLVMBIN/llvm-profdata" ]]; then
  echo "ERROR: 找不到 llvm-profdata/llvm-cov" >&2
  echo "装系统 llvm: apt install llvm" >&2
  exit 1
fi
export PATH="$LLVMBIN:$PATH"
echo ">> 用 llvm 工具: $LLVMBIN"

# 1. 插桩构建 + 跑测试,profraw 按 pid 分文件落到 $OUTDIR
# 注意:必须 --cache_test_results=no(否则 bazel 跳过测试不产 profraw)
# + --sandbox_writable_path 允许 sandbox 写 profraw 到 OUTDIR
echo ">> 插桩跑测试: $TARGET"
bazel test $TARGET \
  --copt="-fprofile-instr-generate=$OUTDIR/%p.profraw" \
  --copt="-fcoverage-mapping" \
  --linkopt="-fprofile-instr-generate=$OUTDIR/%p.profraw" \
  --linkopt="-fcoverage-mapping" \
  --test_env=LLVM_PROFILE_FILE="$OUTDIR/%p.profraw" \
  --cache_test_results=no \
  --sandbox_writable_path="$OUTDIR" \
  --test_output=errors

# 2. 合并所有 profraw
PROFRAW_COUNT=$(ls $OUTDIR/*.profraw 2>/dev/null | wc -l)
if [[ "$PROFRAW_COUNT" -eq 0 ]]; then
  echo "ERROR: 无 profraw 生成,测试可能未实际执行" >&2
  echo "检查 bazel test 是否被缓存或 sandbox 写权限" >&2
  exit 1
fi
echo ">> 合并 profraw ($PROFRAW_COUNT 个)"
"$LLVMBIN/llvm-profdata" merge -o "$OUTDIR/all.profdata" $OUTDIR/*.profraw

# 3. 对每个测试二进制 + 它依赖的 reverb .so 导出 lcov(取并集)
echo ">> 导出 lcov"
mkdir -p "$OUTDIR/parts"
declare -A OBJ_SET
for bin in $(find bazel-bin/reverb -name "*_test" -type f -executable 2>/dev/null | grep -E "cc/|platform/"); do
  realbin=$(readlink -f "$bin" 2>/dev/null || echo "$bin")
  OBJ_SET["$realbin"]=1
  for so in $(ldd "$realbin" 2>/dev/null | grep -oE '/[^ ]*libreverb[^ ]*\.so|/[^ ]*libtensor[^ ]*\.so|/[^ ]*libthird_Uparty_Sreverb[^ ]*\.so' | sort -u); do
    OBJ_SET["$so"]=1
  done
done
# libreverb.so(单体共享库,含全部 reverb .cc)
[[ -f bazel-bin/reverb/libreverb.so ]] && OBJ_SET["$(readlink -f bazel-bin/reverb/libreverb.so)"]=1
[[ -f bazel-bin/reverb/libpybind.so ]] && OBJ_SET["$(readlink -f bazel-bin/reverb/libpybind.so)"]=1

idx=0
for obj in "${!OBJ_SET[@]}"; do
  [[ -f "$obj" ]] || continue
  idx=$((idx+1))
  "$LLVMBIN/llvm-cov" export -format=lcov -instr-profile="$OUTDIR/all.profdata" "$obj" \
    2>/dev/null > "$OUTDIR/parts/part_${idx}.lcov" || true
done
cat "$OUTDIR"/parts/part_*.lcov > "$OUTDIR/coverage.lcov" 2>/dev/null || true
echo ">> lcov 生成: $OUTDIR/coverage.lcov ($(du -h $OUTDIR/coverage.lcov | cut -f1))"

# 4. 可选 HTML
if $GEN_HTML; then
  echo ">> 生成 HTML 报告"
  if command -v genhtml >/dev/null 2>&1; then
    genhtml -o "$OUTDIR/html" "$OUTDIR/coverage.lcov" 2>/dev/null && \
      echo ">> HTML: $OUTDIR/html/index.html"
  else
    # 无 genhtml,用 llvm-cov 的 line-by-line summary
    "$LLVMBIN/llvm-cov" report -instr-profile="$OUTDIR/all.profdata" \
      bazel-bin/reverb/libreverb.so 2>/dev/null | head -40
    echo "(系统无 genhtml,上面是 llvm-cov 文本 summary。装 lcov 可得 HTML: apt install lcov)"
  fi
fi

# 5. 打印 reverb .cc 覆盖率汇总
echo ""
echo ">> reverb/cc .cc 文件覆盖率(从低到高):"
python3 - <<'PYEOF'
import re
res = {}
cur=None; hit=set(); tot=0
for line in open("/tmp/reverb_cov/coverage.lcov", errors="replace"):
    line=line.rstrip("\n")
    if line.startswith("SF:"):
        p=line[3:].replace("/proc/self/cwd/","").replace("bazel-out/k8-opt/bin/","")
        m=re.search(r"((?:reverb/|third_party/).*)",p)
        cur=m.group(1) if m else p; hit=set(); tot=0
    elif line.startswith("DA:") and cur:
        p=line[3:].split(",")
        try: tot+=1; (int(p[1])>0) and hit.add(int(p[0]))
        except: pass
    elif line=="end_of_record" and cur:
        if cur not in res: res[cur]=[0,0]
        res[cur][0]=max(res[cur][0],len(hit)); res[cur][1]=max(res[cur][1],tot)
        cur=None
files=[(f,h,t) for f,(h,t) in res.items() if f.endswith(".cc") and "_test" not in f and ".pb." not in f and "reverb/cc" in f]
files.sort(key=lambda x: x[1]/x[2] if x[2] else 0)
for f,h,t in files:
    pct=h/t*100 if t else 0
    mark="❌" if pct<30 else ("⚠️" if pct<60 else ("🟡" if pct<75 else "✅"))
    print(f"  {mark} {pct:5.1f}%  {h:>4}/{t:<4}  {f}")
PYEOF
echo ""
echo "完成。lcov: /tmp/reverb_cov/coverage.lcov"
