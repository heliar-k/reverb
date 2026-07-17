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

// Ticket ③: the first slice where real Reverb data (samples) crosses the SHM
// boundary. A real Table is pre-seeded via the existing in-process insert path;
// ShmServer owns it; ShmClient.NewSampler -> GetNextTrajectory returns the
// trajectory as TensorBuffers whose bytes/dtype/shape match what was inserted.
// Mirrors sampler_test.cc's table setup.
//
// ponytail: we assert on TensorBuffer bytes/shape/dtype (the actual SHM
// transport contract) rather than round-tripping through ToNdArray. The numpy
// conversion is :tensor_proxy's separately-tested job, and embedding the CPython
// interpreter here would link boringssl (via :sampler -> grpc) into the same
// binary as libpython, whose _hashlib expects system OpenSSL — the symbol clash
// segfaults Py_Initialize. Asserting bytes proves the transport without that
// clash. Upgrade: a pybind11-level numpy assertion once grpc is split out of
// :sampler (or the test runs as a separate python-linked binary).

#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/sampler.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/shm/shm_client.h"
#include "reverb/cc/shm/shm_server.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/table.h"
#include "reverb/cc/tensor_compression.h"
#include "third_party/reverb_tensor/reverb_tensor.pb.h"

namespace deepmind {
namespace reverb {
namespace shm {
namespace {

using ::testing::SizeIs;

// ── Table-seeding helpers (mirrors sampler_test.cc) ──

SequenceRange MakeSequenceRange(uint64_t episode_id, int32_t start,
                                int32_t end) {
  REVERB_CHECK_LE(start, end);
  SequenceRange range;
  range.set_episode_id(episode_id);
  range.set_start(start);
  range.set_end(end);
  return range;
}

PrioritizedItem MakePrioritizedItem(uint64_t key, double priority,
                                    const std::vector<ChunkData>& chunks) {
  REVERB_CHECK(!chunks.empty());
  PrioritizedItem item;
  item.set_key(key);
  item.set_priority(priority);
  for (int i = 0; i < chunks.front().data().tensors_size(); ++i) {
    auto* col = item.mutable_flat_trajectory()->add_columns();
    for (const auto& chunk : chunks) {
      auto* slice = col->add_chunk_slices();
      slice->set_chunk_key(chunk.chunk_key());
      slice->set_offset(0);
      slice->set_length(chunk.data().tensors(i).shape().dim(0));
      slice->set_index(i);
    }
  }
  return item;
}

TensorBuffer MakeTensor(int length) {
  std::vector<uint64_t> values(length * 2);
  for (int i = 0; i < length * 2; i++) values[i] = static_cast<uint64_t>(i);
  std::string bytes(values.size() * sizeof(uint64_t), '\0');
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return TensorBuffer(TensorSpec{DataType::Uint64, {length, 2}},
                      std::move(bytes));
}

ChunkData MakeChunkData(uint64_t key, SequenceRange range) {
  ChunkData chunk;
  chunk.set_chunk_key(key);
  auto t = MakeTensor(range.end() - range.start() + 1);
  CHECK_OK(
      CompressTensorAsProto(t, chunk.mutable_data()->add_tensors()));
  *chunk.mutable_sequence_range() = std::move(range);
  return chunk;
}

TableItem MakeItem(uint64_t key, double priority,
                   const std::vector<SequenceRange>& sequences, int32_t offset,
                   int32_t length) {
  std::vector<std::shared_ptr<ChunkStore::Chunk>> chunks;
  std::vector<ChunkData> data(sequences.size());
  for (int i = 0; i < sequences.size(); i++) {
    data[i] = MakeChunkData(key * 100 + i, sequences[i]);
    chunks.push_back(std::make_shared<ChunkStore::Chunk>(data[i]));
  }
  Table::Item item(MakePrioritizedItem(key, priority, data), std::move(chunks));

  int32_t remaining = length;
  for (int slice_index = 0; slice_index < sequences.size(); slice_index++) {
    for (int col_index = 0;
         col_index < item.flat_trajectory().columns_size(); col_index++) {
      auto* col =
          item.unsafe_mutable_flat_trajectory()->mutable_columns(col_index);
      auto* slice = col->mutable_chunk_slices(slice_index);
      slice->set_offset(offset);
      slice->set_length(
          std::min<int32_t>(slice->length() - slice->offset(), remaining));
      slice->set_index(col_index);
    }
    remaining -=
        item.flat_trajectory().columns(0).chunk_slices(slice_index).length();
    offset = 0;
  }
  return item;
}

void InsertItem(Table* table, uint64_t key, double priority,
                std::vector<int> sequence_lengths, int32_t offset = 0,
                int32_t length = 0, bool squeeze = false) {
  REVERB_CHECK(!squeeze || length == 1);
  if (length == 0) {
    length =
        std::accumulate(sequence_lengths.begin(), sequence_lengths.end(), 0) -
        offset;
  }
  std::vector<SequenceRange> ranges(sequence_lengths.size());
  int step_index = 0;
  for (int i = 0; i < sequence_lengths.size(); i++) {
    ranges[i] = MakeSequenceRange(100 * key, step_index,
                                  step_index + sequence_lengths[i] - 1);
    step_index += sequence_lengths[i];
  }
  auto item = MakeItem(key, priority, ranges, offset, length);
  item.unsafe_mutable_flat_trajectory()->mutable_columns(0)->set_squeeze(
      squeeze);
  REVERB_EXPECT_OK(table->InsertOrAssign(std::move(item)));
}

std::shared_ptr<Table> MakeTable(int max_size = 100) {
  return std::make_shared<Table>(
      /*name=*/"queue",
      /*sampler=*/std::make_shared<FifoSelector>(),
      /*remover=*/std::make_shared<FifoSelector>(),
      /*max_size=*/max_size,
      /*max_times_sampled=*/1,
      /*rate_limiter=*/std::make_shared<RateLimiter>(1, 1, 0, max_size));
}

std::string UniqueTag(const std::string& tag) {
  return tag + "_" + std::to_string(getpid()) + "_" +
         std::to_string(reinterpret_cast<uintptr_t>(&tag));
}

void ExpectTensorBufferEqual(const TensorBuffer& x, const TensorBuffer& y) {
  ASSERT_EQ(x.dtype(), y.dtype());
  ASSERT_EQ(x.shape(), y.shape());
  EXPECT_EQ(x.bytes(), y.bytes()) << "byte content differs";
}

// ── Tests ──

// The core ③ test: real data crosses SHM and comes back with the same
// bytes/dtype/shape as what was inserted. Mirrors the local Sampler path's
// expectation for the same seed.
TEST(ShmSampleTest, SampleCrossesShmMatchesInsertedData) {
  auto table = MakeTable();
  InsertItem(table.get(), /*key=*/1, /*priority=*/1.0,
             /*sequence_lengths=*/{5}, /*offset=*/0, /*length=*/5,
             /*squeeze=*/false);

  std::string sock = "/tmp/reverb_shm_sample_" + UniqueTag("s1") + ".sock";
  auto server = ShmServer::Create({table}, sock);
  REVERB_ASSERT_OK(server.status());
  REVERB_ASSERT_OK((*server)->Start());

  auto client = ShmClient::Connect(sock);
  REVERB_ASSERT_OK(client.status());

  std::unique_ptr<ShmSampler> sampler;
  REVERB_ASSERT_OK(
      (*client)->NewSampler("queue", /*options=*/{1}, &sampler));

  std::vector<TensorBuffer> data;
  REVERB_EXPECT_OK(sampler->GetNextTrajectory(&data));
  ASSERT_THAT(data, SizeIs(1));
  // Item 1: full length-5 chunk, shape [5, 2], uint64. The bytes that crossed
  // the SHM pool must match the inserted tensor exactly.
  ExpectTensorBufferEqual(data[0], MakeTensor(5));

  sampler->Close();
  (*server)->Stop();
}

// The squeeze flag must round-trip: a squeezed column returns shape [2], not
// [1, 2] (mirrors LocalSamplerTest.GetNextTrajectorySqueezesColumnsIfSet).
TEST(ShmSampleTest, SqueezedColumnRoundTrips) {
  auto table = MakeTable();
  InsertItem(table.get(), /*key=*/1, /*priority=*/1.0,
             /*sequence_lengths=*/{5}, /*offset=*/2, /*length=*/1,
             /*squeeze=*/true);

  std::string sock = "/tmp/reverb_shm_sample_" + UniqueTag("sq") + ".sock";
  auto server = ShmServer::Create({table}, sock);
  REVERB_ASSERT_OK(server.status());
  REVERB_ASSERT_OK((*server)->Start());

  auto client = ShmClient::Connect(sock);
  REVERB_ASSERT_OK(client.status());

  std::unique_ptr<ShmSampler> sampler;
  REVERB_ASSERT_OK((*client)->NewSampler("queue", {1}, &sampler));

  std::vector<TensorBuffer> data;
  REVERB_EXPECT_OK(sampler->GetNextTrajectory(&data));
  ASSERT_THAT(data, SizeIs(1));
  // Squeezed length-1 sample at offset 2 of a length-4 chunk: shape [2].
  ExpectTensorBufferEqual(data[0], MakeTensor(4).SubSlice(2));

  sampler->Close();
  (*server)->Stop();
}

// Timeout path: sampling from an empty table with a short timeout returns
// DeadlineExceeded (maps to reverb.errors.DeadlineExceededError on the Python
// side). Mirrors LocalSamplerTest.GetNextTrajectoryForwardsFatalServerError.
TEST(ShmSampleTest, EmptyTableTimeoutIsDeadlineExceeded) {
  auto table = MakeTable();  // empty: rate limiter min_size_to_sample = 1

  std::string sock = "/tmp/reverb_shm_sample_" + UniqueTag("to") + ".sock";
  auto server = ShmServer::Create({table}, sock);
  REVERB_ASSERT_OK(server.status());
  REVERB_ASSERT_OK((*server)->Start());

  auto client = ShmClient::Connect(sock);
  REVERB_ASSERT_OK(client.status());

  std::unique_ptr<ShmSampler> sampler;
  Sampler::Options options;
  options.max_samples = 1;
  options.rate_limiter_timeout = absl::Milliseconds(50);
  REVERB_ASSERT_OK((*client)->NewSampler("queue", options, &sampler));

  std::vector<TensorBuffer> data;
  auto status = sampler->GetNextTrajectory(&data);
  EXPECT_EQ(status.code(), absl::StatusCode::kDeadlineExceeded);

  sampler->Close();
  (*server)->Stop();
}

// RELEASE recycles pool blocks: sampling many items must not leak. Each
// sample ([5,2] uint64 = 80 bytes) lands in the pool's 256-byte tier, which
// has kDefaultBlocksPerSlab (256) blocks. We sample 600 times — more than
// the 256-block tier holds — so a broken RELEASE (never Deallocating) would
// exhaust the tier after 256 allocations and the 257th pool_.Allocate would
// block forever on its condvar, hanging the test (bazel kills it as TIMEOUT).
// If RELEASE works, blocks recycle and all 600 samples succeed. Do NOT lower
// the count below 257 without also shrinking the pool (ShmServer::Create has
// no pool-config knob); 600 leaves headroom above the 256-block threshold.
TEST(ShmSampleTest, ReleaseRecyclesPoolBlocks) {
  constexpr int kNumSamples = 600;
  auto table = MakeTable(kNumSamples);
  for (int i = 1; i <= kNumSamples; i++) {
    InsertItem(table.get(), /*key=*/i, /*priority=*/1.0,
               /*sequence_lengths=*/{5}, /*offset=*/0, /*length=*/5);
  }

  std::string sock = "/tmp/reverb_shm_sample_" + UniqueTag("rel") + ".sock";
  auto server = ShmServer::Create({table}, sock);
  REVERB_ASSERT_OK(server.status());
  REVERB_ASSERT_OK((*server)->Start());

  auto client = ShmClient::Connect(sock);
  REVERB_ASSERT_OK(client.status());

  std::unique_ptr<ShmSampler> sampler;
  REVERB_ASSERT_OK(
      (*client)->NewSampler("queue", {kNumSamples}, &sampler));

  for (int i = 0; i < kNumSamples; i++) {
    std::vector<TensorBuffer> data;
    REVERB_EXPECT_OK(sampler->GetNextTrajectory(&data))
        << "sample " << i << " failed (pool likely leaked)";
    ASSERT_THAT(data, SizeIs(1));
    ASSERT_EQ(data[0].shape().size(), 2);
    EXPECT_EQ(data[0].shape()[0], 5);
    EXPECT_EQ(data[0].shape()[1], 2);
  }

  sampler->Close();
  (*server)->Stop();
}

}  // namespace
}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
