// Copyright 2019 DeepMind Technologies Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "reverb/cc/tensor_compression.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "third_party/reverb_tensor/reverb_tensor.pb.h"

namespace deepmind {
namespace reverb {
namespace {

using ::testing::HasSubstr;
using ::absl_testing::StatusIs;

// Builds a numeric TensorBuffer from a host vector laid out row-major in
// little-endian. T must be a numeric type with a matching DataType.
template <typename T>
TensorBuffer MakeNumeric(DataType dt, const std::vector<int64_t>& shape,
                         const std::vector<T>& values) {
  std::string bytes(values.size() * sizeof(T), '\0');
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return TensorBuffer(TensorSpec{dt, shape}, std::move(bytes));
}

template <typename T>
std::vector<T> ReadNumeric(const TensorBuffer& t) {
  std::vector<T> out(t.NumElements());
  std::memcpy(out.data(), t.bytes().data(), out.size() * sizeof(T));
  return out;
}

// Encodes a list of strings into the [len][bytes] layout TensorBuffer uses.
TensorBuffer MakeString(const std::vector<int64_t>& shape,
                        const std::vector<std::string>& values) {
  std::string bytes;
  for (const std::string& s : values) {
    uint32_t len = static_cast<uint32_t>(s.size());
    char header[4] = {
        static_cast<char>(len & 0xff),
        static_cast<char>((len >> 8) & 0xff),
        static_cast<char>((len >> 16) & 0xff),
        static_cast<char>((len >> 24) & 0xff),
    };
    bytes.append(header, 4);
    bytes.append(s);
  }
  return TensorBuffer(TensorSpec{DataType::String, shape}, std::move(bytes));
}

std::vector<std::string> ReadStrings(const TensorBuffer& t) {
  std::vector<std::string> out;
  size_t pos = 0;
  absl::string_view src = t.bytes();
  while (pos < src.size()) {
    uint32_t len = static_cast<uint8_t>(src[pos]) |
                   (static_cast<uint8_t>(src[pos + 1]) << 8) |
                   (static_cast<uint8_t>(src[pos + 2]) << 16) |
                   (static_cast<uint8_t>(src[pos + 3]) << 24);
    pos += 4;
    out.emplace_back(src.data() + pos, len);
    pos += len;
  }
  return out;
}

template <typename T>
void ExpectTensorBufferEq(const TensorBuffer& x, const TensorBuffer& y) {
  ASSERT_EQ(x.dtype(), y.dtype());
  ASSERT_EQ(x.shape(), y.shape());
  EXPECT_EQ(x.bytes(), y.bytes()) << "byte content differs";
}

// Fills `count` values with a deterministic pattern.
template <typename T>
std::vector<T> Pattern(int count) {
  std::vector<T> v(count);
  for (int i = 0; i < count; i++) v[i] = static_cast<T>(i * 7 + 3);
  return v;
}

TEST(TensorCompressionTest, EncodeMatchesDecodeInt32) {
  auto values = Pattern<int32_t>(16 * 37 * 6);
  TensorBuffer tensor =
      MakeNumeric<int32_t>(DataType::Int32, {16, 37, 6}, values);
  TensorBuffer encoded = DeltaEncode(tensor, true);
  TensorBuffer decoded = DeltaEncode(encoded, false);
  ExpectTensorBufferEq<int32_t>(tensor, decoded);
}

TEST(TensorCompressionTest, EncodeMatchesDecodeInt8) {
  auto values = Pattern<int8_t>(4 * 5 * 3);
  TensorBuffer tensor =
      MakeNumeric<int8_t>(DataType::Int8, {4, 5, 3}, values);
  TensorBuffer encoded = DeltaEncode(tensor, true);
  TensorBuffer decoded = DeltaEncode(encoded, false);
  ExpectTensorBufferEq<int8_t>(tensor, decoded);
}

TEST(TensorCompressionTest, EncodeMatchesDecodeUint64) {
  auto values = Pattern<uint64_t>(2 * 8 * 4);
  TensorBuffer tensor =
      MakeNumeric<uint64_t>(DataType::Uint64, {2, 8, 4}, values);
  TensorBuffer encoded = DeltaEncode(tensor, true);
  TensorBuffer decoded = DeltaEncode(encoded, false);
  ExpectTensorBufferEq<uint64_t>(tensor, decoded);
}

TEST(TensorCompressionTest, EncodeMatchesDecodeUint8) {
  auto values = Pattern<uint8_t>(3 * 4 * 2);
  TensorBuffer tensor =
      MakeNumeric<uint8_t>(DataType::Uint8, {3, 4, 2}, values);
  TensorBuffer encoded = DeltaEncode(tensor, true);
  TensorBuffer decoded = DeltaEncode(encoded, false);
  ExpectTensorBufferEq<uint8_t>(tensor, decoded);
}

TEST(TensorCompressionTest, EncodeLeavesFloatUnchanged) {
  // Float is not an integral type: DeltaEncode returns it unchanged.
  auto values = Pattern<float>(4 * 3);
  TensorBuffer tensor =
      MakeNumeric<float>(DataType::Float32, {4, 3}, values);
  TensorBuffer encoded = DeltaEncode(tensor, true);
  EXPECT_EQ(encoded.bytes(), tensor.bytes());
}

TEST(TensorCompressionTest, EncodeLeavesRank1Unchanged) {
  auto values = Pattern<int32_t>(8);
  TensorBuffer tensor = MakeNumeric<int32_t>(DataType::Int32, {8}, values);
  TensorBuffer encoded = DeltaEncode(tensor, true);
  EXPECT_EQ(encoded.bytes(), tensor.bytes());
}

TEST(TensorCompressionTest, EncodeListMatchesDecode) {
  auto values = Pattern<int32_t>(16 * 37 * 6);
  TensorBuffer tensor =
      MakeNumeric<int32_t>(DataType::Int32, {16, 37, 6}, values);
  std::vector<TensorBuffer> tensors{tensor, tensor};
  std::vector<TensorBuffer> encoded = DeltaEncodeList(tensors, true);
  std::vector<TensorBuffer> decoded = DeltaEncodeList(encoded, false);
  EXPECT_EQ(tensors.size(), decoded.size());
  for (int i = 0; i < tensors.size(); i++) {
    ExpectTensorBufferEq<int32_t>(tensors[i], decoded[i]);
  }
}

TEST(TensorCompressionTest, StringTensor) {
  TensorBuffer tensor = MakeString({2}, {"hello", "world"});

  ::reverb::tensor::TensorProto proto;
  REVERB_ASSERT_OK(CompressTensorAsProto(tensor, &proto));

  absl::StatusOr<TensorBuffer> r = DecompressTensorFromProto(proto);
  REVERB_ASSERT_OK(r);
  TensorBuffer result = std::move(r).value();
  ExpectTensorBufferEq<int>(tensor, result);
  EXPECT_THAT(ReadStrings(result), ::testing::ElementsAre("hello", "world"));
}

TEST(TensorCompressionTest, NonStringTensor) {
  auto values = Pattern<int32_t>(4);
  TensorBuffer tensor =
      MakeNumeric<int32_t>(DataType::Int32, {2, 2}, values);

  ::reverb::tensor::TensorProto proto;
  REVERB_ASSERT_OK(CompressTensorAsProto(tensor, &proto));

  absl::StatusOr<TensorBuffer> r = DecompressTensorFromProto(proto);
  REVERB_ASSERT_OK(r);
  TensorBuffer result = std::move(r).value();
  ExpectTensorBufferEq<int32_t>(tensor, result);
}

TEST(TensorCompressionTest, NonStringTensorWithDeltaEncoding) {
  auto values = Pattern<int32_t>(4);
  TensorBuffer tensor =
      MakeNumeric<int32_t>(DataType::Int32, {2, 2}, values);

  ::reverb::tensor::TensorProto proto;
  REVERB_ASSERT_OK(CompressTensorAsProto(DeltaEncode(tensor, true), &proto));
  absl::StatusOr<TensorBuffer> r2 = DecompressTensorFromProto(proto);
  REVERB_ASSERT_OK(r2);
  TensorBuffer result = std::move(r2).value();
  // result == DeltaEncode(tensor, true); decoding it must recover `tensor`.
  ExpectTensorBufferEq<int32_t>(tensor, DeltaEncode(result, false));
}

TEST(TensorCompressionTest, NonStringTensorWithDeltaRoundTrip) {
  // Full encode -> compress -> decompress -> decode == original, across a
  // larger shape to exercise the per-row delta loop.
  auto values = Pattern<int32_t>(8 * 4);
  TensorBuffer tensor =
      MakeNumeric<int32_t>(DataType::Int32, {8, 4}, values);

  ::reverb::tensor::TensorProto proto;
  REVERB_ASSERT_OK(CompressTensorAsProto(DeltaEncode(tensor, true), &proto));
  absl::StatusOr<TensorBuffer> r = DecompressTensorFromProto(proto);
  REVERB_ASSERT_OK(r);
  TensorBuffer result = std::move(r).value();
  ExpectTensorBufferEq<int32_t>(tensor, DeltaEncode(result, false));
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
