// Tier 3 协议层拷贝成本微基准(诊断工具,非回归测试)。
//
// 背景:writer Finish→wire→server 链路上,每字节载荷经过的物化拷贝:
//   Concat → SerializeToProto(set_tensor_content) → snappy 压缩 →
//   ChunkData.SerializeToArray → 对端 ParseFromString → DecompressTensorFromProto
// 本基准逐步计时并与 memcpy 地板对比,回答"Tier 3 是否值得优化"。
//
// 运行:bazel run -c opt //reverb/cc/support:tier3_bench

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/tensor_compression.h"

namespace {

using deepmind::reverb::DataType;
using deepmind::reverb::TensorBuffer;
using deepmind::reverb::TensorSpec;

double NowUs() {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// 对 `fn` 跑 iters 次,返回平均 μs/次。fn 返回 bool 防编译器优化掉。
template <typename F>
double TimeOp(int iters, F&& fn) {
  // warmup
  volatile bool sink = fn();
  (void)sink;
  double t0 = NowUs();
  for (int i = 0; i < iters; ++i) {
    volatile bool s = fn();
    (void)s;
  }
  return (NowUs() - t0) / iters;
}

std::string RandomBytes(size_t n) {
  std::mt19937 rng(42);
  std::string s(n, '\0');
  for (size_t i = 0; i < n; ++i) s[i] = static_cast<char>(rng());
  return s;
}

void BenchSize(size_t bytes, int iters) {
  std::printf("\n=== payload %.1f KB (%d iters) ===\n", bytes / 1024.0, iters);
  std::string raw = RandomBytes(bytes);
  TensorSpec spec{DataType::Float32, {1, static_cast<int64_t>(bytes / 4)}};
  TensorBuffer buf(spec, raw);

  // 0. memcpy 地板
  std::string dst;
  double t_memcpy = TimeOp(iters, [&] {
    dst = raw;  // string 拷贝即 memcpy
    return !dst.empty();
  });

  // 1. Concat:单 timestep(benchmark insert 场景)与 20 timestep 分块场景
  double t_concat1 = TimeOp(iters, [&] {
    auto r = TensorBuffer::Concat({buf});
    return r.ok();
  });
  std::vector<TensorBuffer> parts;
  for (int i = 0; i < 20; ++i) {
    TensorSpec ps{DataType::Float32,
                  {1, static_cast<int64_t>(bytes / 4 / 20)}};
    parts.emplace_back(ps, raw.substr(i * bytes / 20, bytes / 20));
  }
  double t_concat20 = TimeOp(iters, [&] {
    auto r = TensorBuffer::Concat(parts);
    return r.ok();
  });

  // 2. SerializeToProto(set_tensor_content 拷贝)
  double t_ser_proto = TimeOp(iters, [&] {
    ::reverb::tensor::TensorProto p;
    return buf.SerializeToProto(&p).ok();
  });

  // 3. CompressTensorAsProto(= 2 + snappy + 再 set 一次)
  double t_compress = TimeOp(iters, [&] {
    ::reverb::tensor::TensorProto p;
    return deepmind::reverb::CompressTensorAsProto(buf, &p).ok();
  });

  // 4. 完整 ChunkData 装配 + SerializeToArray(writer 出向 wire)
  double t_wire_out = TimeOp(iters, [&] {
    deepmind::reverb::ChunkData cd;
    cd.set_chunk_key(1);
    cd.set_data_uncompressed_size(bytes);
    cd.set_data_tensors_len(1);
    bool ok = deepmind::reverb::CompressTensorAsProto(
                  buf, cd.mutable_data()->add_tensors())
                  .ok();
    std::string out;
    out.resize(cd.ByteSizeLong());
    return ok && cd.SerializeToArray(out.data(), out.size());
  });

  // 5. 入向:ParseFromString + DecompressTensorFromProto
  deepmind::reverb::ChunkData cd;
  cd.set_chunk_key(1);
  cd.set_data_uncompressed_size(bytes);
  cd.set_data_tensors_len(1);
  (void)deepmind::reverb::CompressTensorAsProto(
      buf, cd.mutable_data()->add_tensors());
  std::string wire = cd.SerializeAsString();
  double t_wire_in = TimeOp(iters, [&] {
    deepmind::reverb::ChunkData parsed;
    if (!parsed.ParseFromString(wire)) return false;
    auto tb = deepmind::reverb::DecompressTensorFromProto(
        parsed.data().tensors(0));
    return tb.ok();
  });

  // 6. 全链路估算 vs 实测链路
  double t_chain_out = TimeOp(iters, [&] {
    auto c = TensorBuffer::Concat({buf});
    if (!c.ok()) return false;
    deepmind::reverb::ChunkData d;
    d.set_chunk_key(1);
    d.set_data_uncompressed_size(bytes);
    d.set_data_tensors_len(1);
    bool ok = deepmind::reverb::CompressTensorAsProto(
                  *c, d.mutable_data()->add_tensors())
                  .ok();
    std::string out;
    out.resize(d.ByteSizeLong());
    return ok && d.SerializeToArray(out.data(), out.size());
  });

  auto row = [&](const char* name, double us) {
    // bytes/μs 数值上等于 MB/s。
    std::printf("%-34s %9.1f μs   %7.0f MB/s   ×%.1f\n", name, us,
                bytes / us, us / t_memcpy);
  };
  row("memcpy 地板", t_memcpy);
  row("Concat(1 buffer)", t_concat1);
  row("Concat(20 buffers)", t_concat20);
  row("SerializeToProto", t_ser_proto);
  row("CompressTensorAsProto(+snappy)", t_compress);
  row("ChunkData装配+SerializeToArray", t_wire_out);
  row("ParseFromString+Decompress", t_wire_in);
  row("出向全链路(Concat+压缩+wire)", t_chain_out);
}

}  // namespace

int main() {
  BenchSize(112896, 2000);   // med:  float32[84,84,4]
  BenchSize(786432, 500);    // large: float32[256,256,3]
  return 0;
}
