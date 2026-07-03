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

#include "reverb/cc/support/trajectory_util.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/tensor_compression.h"
#include "third_party/reverb_tensor/reverb_tensor.pb.h"

namespace deepmind {
namespace reverb {
namespace internal {
namespace {

using ::testing::ElementsAre;

// ponytail: inlined minimal EqualsProto to avoid depending on
// //reverb/cc/testing:proto_test_util (its .cc still calls the now-renamed
// CompressTensorAsProto(tensorflow::Tensor) and won't compile until migrated).
class ProtoStringMatcher {
 public:
  explicit ProtoStringMatcher(const std::string& expected)
      : expected_str_(expected) {}

  template <typename Message>
  bool MatchAndExplain(const Message& actual,
                       ::testing::MatchResultListener* listener) const {
    Message expected;
    if (!google::protobuf::TextFormat::ParseFromString(expected_str_,
                                                       &expected)) {
      *listener << "failed to parse expected proto";
      return false;
    }
    google::protobuf::util::MessageDifferencer differencer;
    std::string diff;
    differencer.ReportDifferencesToString(&diff);
    if (!differencer.Compare(expected, actual)) {
      *listener << "protos differ:\n" << diff;
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os) const { *os << expected_str_; }
  void DescribeNegationTo(std::ostream* os) const {
    *os << "not equal to: " << expected_str_;
  }

 private:
  std::string expected_str_;
};

inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const std::string& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

// Builds an int32 TensorBuffer from host values (little-endian, row-major).
TensorBuffer MakeInt32(const std::vector<int64_t>& shape,
                       const std::vector<int32_t>& values) {
  std::string bytes(values.size() * sizeof(int32_t), '\0');
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return TensorBuffer(TensorSpec{DataType::Int32, shape}, std::move(bytes));
}

std::vector<int32_t> ReadInt32(const TensorBuffer& t) {
  std::vector<int32_t> out(t.NumElements());
  std::memcpy(out.data(), t.bytes().data(), out.size() * sizeof(int32_t));
  return out;
}

TEST(GetChunkKeys, DedupsTimestepTrajectory) {
  auto trajectory = FlatTimestepTrajectory(
      /*chunk_keys=*/{1, 2, 3}, /*chunk_lengths=*/{2, 2, 2},
      /*num_columns=*/2, /*offset=*/1, /*length=*/4);
  EXPECT_THAT(GetChunkKeys(trajectory), ElementsAre(1, 2, 3));
}

TEST(GetChunkKeys, DedupsNonTimestepTrajectory) {
  FlatTrajectory trajectory;

  auto* first = trajectory.add_columns();
  first->add_chunk_slices()->set_chunk_key(1);
  first->add_chunk_slices()->set_chunk_key(2);
  first->add_chunk_slices()->set_chunk_key(4);

  auto* second = trajectory.add_columns();
  second->add_chunk_slices()->set_chunk_key(2);
  second->add_chunk_slices()->set_chunk_key(3);

  EXPECT_THAT(GetChunkKeys(trajectory), ElementsAre(1, 2, 4, 3));
}

TEST(FlatTimestepTrajectory, CreateTrajectoryFromChunks) {
  ChunkData first;
  first.set_chunk_key(1);
  first.mutable_sequence_range()->set_start(0);
  first.mutable_sequence_range()->set_end(4);
  first.mutable_data()->add_tensors();
  first.mutable_data()->add_tensors();
  first.set_data_tensors_len(2);


  ChunkData second;
  second.set_chunk_key(2);
  second.mutable_sequence_range()->set_start(5);
  second.mutable_sequence_range()->set_end(7);
  second.mutable_data()->add_tensors();
  second.mutable_data()->add_tensors();
  second.set_data_tensors_len(2);

  std::vector<std::shared_ptr<ChunkStore::Chunk>> chunks = {
      std::make_shared<ChunkStore::Chunk>(std::move(first)),
      std::make_shared<ChunkStore::Chunk>(std::move(second)),
  };

  auto trajectory = FlatTimestepTrajectory(chunks, 1, 5);
  EXPECT_THAT(trajectory, EqualsProto(R"(
                columns: {
                  chunk_slices: { chunk_key: 1 offset: 1 length: 4 index: 0 }
                  chunk_slices: { chunk_key: 2 offset: 0 length: 1 index: 0 }
                }
                columns: {
                  chunk_slices: { chunk_key: 1 offset: 1 length: 4 index: 1 }
                  chunk_slices: { chunk_key: 2 offset: 0 length: 1 index: 1 }
                }
              )"));
}

TEST(FlatTimestepTrajectory, CreateTrajectoryFromVectors) {
  auto trajectory = FlatTimestepTrajectory(
      /*chunk_keys=*/{1, 2},
      /*chunk_lengths=*/{4, 4}, /*num_columns=*/2, /*offset=*/2, /*length=*/4);
  EXPECT_THAT(trajectory, EqualsProto(R"(
                columns: {
                  chunk_slices: { chunk_key: 1 offset: 2 length: 2 index: 0 }
                  chunk_slices: { chunk_key: 2 offset: 0 length: 2 index: 0 }
                }
                columns: {
                  chunk_slices: { chunk_key: 1 offset: 2 length: 2 index: 1 }
                  chunk_slices: { chunk_key: 2 offset: 0 length: 2 index: 1 }
                }
              )"));
}

TEST(IsTimestepTrajectory, SingleColumn) {
  FlatTrajectory trajectory;
  auto* col = trajectory.add_columns();

  auto* first = col->add_chunk_slices();
  first->set_chunk_key(1);
  first->set_length(3);
  first->set_offset(2);

  auto* second = col->add_chunk_slices();
  second->set_chunk_key(2);
  second->set_length(2);
  second->set_offset(0);

  EXPECT_TRUE(IsTimestepTrajectory(trajectory));
}

TEST(IsTimestepTrajectory, SingleColumnWithGap) {
  FlatTrajectory trajectory;
  auto* col = trajectory.add_columns();

  auto* first = col->add_chunk_slices();
  first->set_chunk_key(1);
  first->set_length(3);
  first->set_offset(2);

  auto* second = col->add_chunk_slices();
  second->set_chunk_key(2);
  second->set_length(2);
  second->set_offset(1);

  EXPECT_FALSE(IsTimestepTrajectory(trajectory));
}

TEST(IsTimestepTrajectory, MultiColumn) {
  FlatTrajectory trajectory;
  for (int i = 0; i < 3; i++) {
    auto* col = trajectory.add_columns();

    auto* first = col->add_chunk_slices();
    first->set_chunk_key(1);
    first->set_length(3);
    first->set_offset(2);
    first->set_index(i);

    auto* second = col->add_chunk_slices();
    second->set_chunk_key(2);
    second->set_length(2);
    second->set_offset(0);
    second->set_index(i);
  }

  EXPECT_TRUE(IsTimestepTrajectory(trajectory));
}

class IsTimestepTrajectoryTest : public ::testing::Test {
 protected:
  IsTimestepTrajectoryTest() {
    for (int i = 0; i < 3; i++) {
      auto* col = valid_.add_columns();

      auto* first = col->add_chunk_slices();
      first->set_chunk_key(1);
      first->set_length(3);
      first->set_offset(2);
      first->set_index(i);

      auto* second = col->add_chunk_slices();
      second->set_chunk_key(2);
      second->set_length(2);
      second->set_offset(0);
      second->set_index(i);
    }
  }

  FlatTrajectory GetValid() const { return valid_; }

 private:
  FlatTrajectory valid_;
};

TEST_F(IsTimestepTrajectoryTest, KeyChanged) {
  auto trajectory = GetValid();
  trajectory.mutable_columns(1)->mutable_chunk_slices(0)->set_chunk_key(3);
  EXPECT_FALSE(IsTimestepTrajectory(trajectory));
}

TEST_F(IsTimestepTrajectoryTest, LengthChanged) {
  auto trajectory = GetValid();
  trajectory.mutable_columns(1)->mutable_chunk_slices(0)->set_length(5);
  EXPECT_FALSE(IsTimestepTrajectory(trajectory));
}

TEST_F(IsTimestepTrajectoryTest, NumSlicesChanged) {
  auto trajectory = GetValid();
  trajectory.mutable_columns(1)->mutable_chunk_slices()->RemoveLast();
  EXPECT_FALSE(IsTimestepTrajectory(trajectory));
}

TEST_F(IsTimestepTrajectoryTest, IndexIsInvalid) {
  auto trajectory = GetValid();
  trajectory.mutable_columns(1)->mutable_chunk_slices(0)->set_index(3);
  EXPECT_FALSE(IsTimestepTrajectory(trajectory));
}

TEST(TimestepTrajectoryLength, AccumulatesSlices) {
  FlatTrajectory trajectory;
  auto* col = trajectory.add_columns();

  auto* first = col->add_chunk_slices();
  first->set_chunk_key(1);
  first->set_length(3);
  first->set_offset(2);

  auto* second = col->add_chunk_slices();
  second->set_chunk_key(2);
  second->set_length(2);
  second->set_offset(0);

  EXPECT_EQ(TimestepTrajectoryLength(trajectory), 5);

  // Changing the offset should not impact it.
  first->set_offset(5);
  EXPECT_EQ(TimestepTrajectoryLength(trajectory), 5);

  // Changing the lengths of slices should impact the total length.
  first->set_length(5);
  EXPECT_EQ(TimestepTrajectoryLength(trajectory), 7);

  second->set_length(1);
  EXPECT_EQ(TimestepTrajectoryLength(trajectory), 6);
}

TEST(UnpackChunkColumn, SelectsCorrectColumn) {
  TensorBuffer first_col_tensor =
      MakeInt32({1}, {1337});
  TensorBuffer second_col_tensor =
      MakeInt32({1}, {9000});

  ChunkData data;
  REVERB_ASSERT_OK(CompressTensorAsProto(
      first_col_tensor, data.mutable_data()->add_tensors()));
  REVERB_ASSERT_OK(CompressTensorAsProto(
      second_col_tensor, data.mutable_data()->add_tensors()));
  data.set_data_tensors_len(2);

  TensorBuffer first_got;
  REVERB_EXPECT_OK(UnpackChunkColumn(data, 0, &first_got));
  EXPECT_EQ(ReadInt32(first_got), std::vector<int32_t>{1337});

  TensorBuffer second_got;
  REVERB_EXPECT_OK(UnpackChunkColumn(data, 1, &second_got));
  EXPECT_EQ(ReadInt32(second_got), std::vector<int32_t>{9000});
}

TEST(UnpackChunkColumnAndSlice, SlicesTimeRange) {
  // 4 timesteps of 2 ints each. Slice [1, 3) -> 2 timesteps.
  TensorBuffer tensor =
      MakeInt32({4, 2}, {0, 0, 1, 1, 2, 2, 3, 3});
  ChunkData data;
  REVERB_ASSERT_OK(
      CompressTensorAsProto(tensor, data.mutable_data()->add_tensors()));
  data.set_data_tensors_len(1);

  TensorBuffer got;
  REVERB_EXPECT_OK(UnpackChunkColumnAndSlice(data, 0, 1, 2, &got));
  ASSERT_EQ(got.shape(), std::vector<int64_t>({2, 2}));
  EXPECT_EQ(ReadInt32(got), std::vector<int32_t>({1, 1, 2, 2}));
}

TEST(UnpackChunkColumn, DeltaEncodedChunk) {
  // Encode a [3, 2] int32 tensor with delta encoding and round-trip via
  // UnpackChunkColumn (which applies the decode when delta_encoded is set).
  TensorBuffer tensor = MakeInt32({3, 2}, {10, 20, 11, 21, 12, 22});
  TensorBuffer encoded = DeltaEncode(tensor, true);

  ChunkData data;
  REVERB_ASSERT_OK(
      CompressTensorAsProto(encoded, data.mutable_data()->add_tensors()));
  data.set_data_tensors_len(1);
  data.set_delta_encoded(true);

  TensorBuffer got;
  REVERB_EXPECT_OK(UnpackChunkColumn(data, 0, &got));
  EXPECT_EQ(ReadInt32(got), std::vector<int32_t>({10, 20, 11, 21, 12, 22}));
}

TEST(UnpackChunkColumn, ColumnOutOfRangeFails) {
  ChunkData data;
  data.mutable_data()->add_tensors();  // 1 column
  data.set_data_tensors_len(1);
  TensorBuffer got;
  EXPECT_FALSE(UnpackChunkColumn(data, 5, &got).ok());
}

}  // namespace
}  // namespace internal
}  // namespace reverb
}  // namespace deepmind
