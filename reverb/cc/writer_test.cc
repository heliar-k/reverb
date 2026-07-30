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

#include "reverb/cc/writer.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "grpcpp/impl/codegen/call_op_set.h"
#include "grpcpp/impl/codegen/status.h"
#include "grpcpp/impl/codegen/sync_stream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "reverb/cc/client.h"
#include "reverb/cc/platform/default/status_matchers.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/reverb_service_mock.grpc.pb.h"
#include "reverb/cc/support/grpc_util.h"
#include "reverb/cc/support/queue.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/support/trajectory_util.h"
#include "reverb/cc/support/uint128.h"
#include "reverb/cc/testing/proto_test_util.h"
#include "reverb/cc/platform/default/hash_map.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/table.h"
#include "third_party/reverb_tensor/reverb_tensor.pb.h"

namespace deepmind {
namespace reverb {
namespace {

using ::deepmind::reverb::testing::Partially;
using ::testing::ElementsAre;
using ::testing::SizeIs;

// ponytail: alias so the original TEST bodies read as TF-era code.
using Tensor = TensorBuffer;

// Spec used to fill timesteps. name is "" since Append stores per-step specs
// with empty names (see Writer::Append).
const internal::TensorSpec kFloatSpec{"", DataType::Float32, {}};

constexpr auto kNotificationTimeout = absl::Milliseconds(200);

// Builds a TensorBuffer from a spec, filling every element with `value`.
// Mirrors streaming_trajectory_writer_test.cc's MakeTensor but with a constant
// fill (the original TF test filled with 1.0).
Tensor MakeTensor(const internal::TensorSpec& spec, float value = 1.0f) {
  int64_t n = 1;
  for (int64_t d : spec.shape) n *= d;

  std::string bytes;
  if (spec.dtype == DataType::Float32) {
    bytes.resize(static_cast<size_t>(n) * sizeof(float));
    auto* dst = reinterpret_cast<float*>(&bytes[0]);
    for (int64_t i = 0; i < n; ++i) dst[i] = value;
  } else if (spec.dtype == DataType::Int32) {
    bytes.resize(static_cast<size_t>(n) * sizeof(int32_t));
    auto* dst = reinterpret_cast<int32_t*>(&bytes[0]);
    for (int64_t i = 0; i < n; ++i) dst[i] = static_cast<int32_t>(value);
  } else {
    REVERB_LOG(REVERB_FATAL) << "Unexpected dtype";
  }
  return Tensor(TensorSpec{spec.dtype, spec.shape}, std::move(bytes));
}

std::vector<Tensor> MakeTimestep(int num_tensors = 1,
                                 const std::vector<int64_t>& shape = {}) {
  std::vector<Tensor> res;
  res.reserve(num_tensors);
  for (int i = 0; i < num_tensors; ++i) {
    res.push_back(MakeTensor(internal::TensorSpec{"", DataType::Float32, shape}));
  }
  return res;
}

// Builds a SignatureProto with a single tensor_spec named "tensor0".
::reverb::tensor::SignatureProto MakeSignature(
    DataType dtype = DataType::Float32,
    const std::vector<int64_t>& shape = {}) {
  ::reverb::tensor::SignatureProto signature;
  auto* spec = signature.mutable_tensor_spec();
  spec->set_dtype(DataTypeToProto(dtype));
  spec->set_name("tensor0");
  for (int64_t d : shape) spec->mutable_shape()->add_dim(d);
  return signature;
}

// ponytail: BoundedTensorSpec min/max are not representable in the flat
// signature (FlatSignatureFromSignatureProto only reads dtype/shape for
// bounded specs), so we omit them. The signature-validation tests below only
// exercise dtype/shape checks, which are fully covered.
::reverb::tensor::SignatureProto MakeBoundedTensorSpecSignature(
    DataType dtype = DataType::Float32,
    const std::vector<int64_t>& shape = {}) {
  ::reverb::tensor::SignatureProto signature;
  auto* spec = signature.mutable_bounded_tensor_spec();
  spec->set_dtype(DataTypeToProto(dtype));
  spec->set_name("tensor0");
  for (int64_t d : shape) spec->mutable_shape()->add_dim(d);
  return signature;
}

MATCHER(IsChunk, "") {
  if (arg.chunks_size() == 0) {
    return false;
  }
  for (const auto& chunk : arg.chunks()) {
    EXPECT_EQ(chunk.data_tensors_len(), chunk.data().tensors_size());
  }
  return true;
}

MATCHER_P4(IsItemWithRangeAndPriorityAndTable, offset, length, priority, table,
           "") {
  if (arg.items_size() == 0) {
    return false;
  }

  if (arg.items(0).flat_trajectory().columns(0)
      .chunk_slices(0).offset() != offset) {
    return false;
  }

  int total_length = 0;
  for (const auto& slice :
       arg.items(0).flat_trajectory().columns(0).chunk_slices()) {
    total_length += slice.length();
  }
  if (length != total_length) {
    return false;
  }

  if (arg.items(0).priority() != priority) {
    return false;
  }

  if (arg.items(0).table() != table) {
    return false;
  }
  return true;
}

class FakeInsertStream
    : public grpc::ClientReaderWriterInterface<InsertStreamRequest,
                                               InsertStreamResponse> {
 public:
  FakeInsertStream(std::vector<InsertStreamRequest>* requests,
                   int num_success_writes, grpc::Status bad_status,
                   bool automatic_response_ids = true)
      : requests_(requests),
        num_success_writes_(num_success_writes),
        bad_status_(std::move(bad_status)),
        response_ids_(std::make_shared<internal::Queue<uint64_t>>(100)),
        automatic_response_ids_(automatic_response_ids) {}

  bool Write(const InsertStreamRequest& msg,
             grpc::WriteOptions options) override {
    requests_->push_back(msg);
    if (automatic_response_ids_) {
      for (auto& item : msg.items()) {
        response_ids_->Reserve(1);
        response_ids_->PushBatch({item.key()});
      }
    }
    return num_success_writes_-- > 0;
  }

  bool Read(InsertStreamResponse* response) override {
    // If an explicit response IDs queue was provided then we block until it is
    // nonempty.
    response->Clear();

    uint64_t id;
    // There should be at least one id in the response.
    REVERB_CHECK(response_ids_->Pop(&id));
    response->add_keys(id);

    // But in case of batching there might be more.
    while (response_ids_->size() > 0) {
      REVERB_CHECK(response_ids_->Pop(&id));
      response->add_keys(id);
    }
    return true;
  }

  grpc::Status Finish() override {
    return num_success_writes_ >= 0 ? grpc::Status::OK : bad_status_;
  }

  bool WritesDone() override { return num_success_writes_-- > 0; }

  bool NextMessageSize(uint32_t* sz) override {
    if (response_ids_->size() == 0) {
      return false;
    }
    InsertStreamResponse response;
    *sz = response.ByteSizeLong();
    return true;
  }

  void WaitForInitialMetadata() override {}

  std::shared_ptr<internal::Queue<uint64_t>> response_ids() const {
    return response_ids_;
  }

 private:
  std::vector<InsertStreamRequest>* requests_;
  std::queue<uint64_t> written_item_ids_;
  int num_success_writes_;
  grpc::Status bad_status_;
  std::shared_ptr<internal::Queue<uint64_t>> response_ids_;
  bool automatic_response_ids_;
};

class FakeStub : public /* grpc_gen:: */MockReverbServiceStub {
 public:
  explicit FakeStub(std::list<FakeInsertStream*> streams,
                    const ::reverb::tensor::SignatureProto* signature = nullptr)
      : streams_(std::move(streams)) {
    if (signature) {
      *response_.mutable_tables_state_id() =
          Uint128ToMessage(absl::MakeUint128(1, 2));
      auto* table_info = response_.add_table_info();
      table_info->set_name("dist");
      *table_info->mutable_signature() = *signature;
    }
  }
  ~FakeStub() override {
    // Since writers where allocated with New we manually free the memory if
    // the writer hasn't already been passed to the Writer where it is
    // handled as a unique ptr and thus is destroyed with ~Writer.
    while (!streams_.empty()) {
      auto writer = streams_.front();
      delete writer;
      streams_.pop_front();
    }
  }

  grpc::ClientReaderWriterInterface<InsertStreamRequest, InsertStreamResponse>*
  InsertStreamRaw(grpc::ClientContext* context) override {
    auto writer = streams_.front();
    streams_.pop_front();
    return writer;
  }

  grpc::Status ServerInfo(grpc::ClientContext* context,
                          const ServerInfoRequest& request,
                          ServerInfoResponse* response) override {
    *response = response_;
    return grpc::Status::OK;
  }

 private:
  ServerInfoResponse response_;
  std::list<FakeInsertStream*> streams_;
};

std::shared_ptr<FakeStub> MakeGoodStub(
    std::vector<InsertStreamRequest>* requests,
    const ::reverb::tensor::SignatureProto* signature = nullptr) {
  FakeInsertStream* stream = new FakeInsertStream(
      requests, 10000, ToGrpcStatus(absl::InternalError("")));
  return std::make_shared<FakeStub>(std::list<FakeInsertStream*>{stream},
                                    signature);
}

std::shared_ptr<FakeStub> MakeFlakyStub(
    std::vector<InsertStreamRequest>* requests, int num_success, int num_fail,
    grpc::Status error) {
  std::list<FakeInsertStream*> streams;
  streams.push_back(new FakeInsertStream(requests, num_success, error));
  for (int i = 1; i < num_fail; i++) {
    streams.push_back(new FakeInsertStream(requests, 0, error));
  }
  streams.push_back(new FakeInsertStream(
      requests, 10000, ToGrpcStatus(absl::InternalError(""))));
  return std::make_shared<FakeStub>(std::move(streams));
}

TEST(WriterTest, DoesNotSendTimestepsWhenThereAreNoItems) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Writer writer(stub, 2, 10);
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  EXPECT_THAT(requests, SizeIs(0));
}

TEST(WriterTest, OnlySendsChunksWhichAreUsedByItems) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Writer writer(stub, 2, 10);
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  EXPECT_THAT(requests, SizeIs(0));

  REVERB_ASSERT_OK(writer.CreateItem("dist", 3, 1.0));
  ASSERT_THAT(requests, SizeIs(1));
  EXPECT_THAT(requests[0], IsChunk());
  EXPECT_THAT(requests[0],
              IsItemWithRangeAndPriorityAndTable(1, 3, 1.0, "dist"));
  EXPECT_THAT(
      internal::GetChunkKeys(requests[0].items(0).flat_trajectory()),
      ElementsAre(requests[0].chunks(0).chunk_key(),
                  requests[0].chunks(1).chunk_key()));
}

TEST(WriterTest, DoesNotSendAlreadySentChunks) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Writer writer(stub, 2, 10);

  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.CreateItem("dist", 1, 1.5));

  ASSERT_THAT(requests, SizeIs(1));

  EXPECT_THAT(requests[0], IsChunk());
  auto first_chunk_key = requests[0].chunks(0).chunk_key();

  EXPECT_THAT(requests[0],
              IsItemWithRangeAndPriorityAndTable(1, 1, 1.5, "dist"));
  EXPECT_THAT(
      internal::GetChunkKeys(requests[0].items(0).flat_trajectory()),
      ElementsAre(first_chunk_key));

  requests.clear();
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.CreateItem("dist", 3, 1.3));

  ASSERT_THAT(requests, SizeIs(1));
  EXPECT_THAT(requests[0], IsChunk());
  auto second_chunk_key = requests[0].chunks(0).chunk_key();

  EXPECT_THAT(requests[0],
              IsItemWithRangeAndPriorityAndTable(1, 3, 1.3, "dist"));
  EXPECT_THAT(
      internal::GetChunkKeys(requests[0].items(0).flat_trajectory()),
      ElementsAre(first_chunk_key, second_chunk_key));
}

TEST(WriterTest, SendsPendingDataOnClose) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Writer writer(stub, 2, 10);

  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.CreateItem("dist", 1, 1.0));
  EXPECT_THAT(requests, SizeIs(0));

  REVERB_ASSERT_OK(writer.Close());
  ASSERT_THAT(requests, SizeIs(1));
  EXPECT_THAT(requests[0], IsChunk());
  EXPECT_THAT(requests[0],
              IsItemWithRangeAndPriorityAndTable(0, 1, 1.0, "dist"));
  EXPECT_THAT(
      internal::GetChunkKeys(requests[0].items(0).flat_trajectory()),
      ElementsAre(requests[0].chunks(0).chunk_key()));
}

TEST(WriterTest, FailsIfMethodsCalledAfterClose) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Writer writer(stub, 2, 10);

  REVERB_ASSERT_OK(writer.Close());

  EXPECT_FALSE(writer.Close().ok());
  EXPECT_FALSE(writer.Append(MakeTimestep()).ok());
  EXPECT_FALSE(writer.CreateItem("dist", 1, 1.0).ok());
}

TEST(WriterTest, RetriesOnTransientError) {
  std::vector<InsertStreamRequest> requests;
  // 1 fail, then all success.
  auto stub =
      MakeFlakyStub(&requests, 0, 1, ToGrpcStatus(absl::UnavailableError("")));
  Writer writer(stub, 2, 10);

  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.CreateItem("dist", 1, 1.0));

  ASSERT_THAT(requests, SizeIs(2));
  EXPECT_THAT(requests[0], IsChunk());
  EXPECT_THAT(requests[1], IsChunk());
  EXPECT_THAT(requests[0], testing::EqualsProto(requests[1]));
  EXPECT_THAT(requests[1],
              IsItemWithRangeAndPriorityAndTable(1, 1, 1.0, "dist"));
  EXPECT_THAT(
      internal::GetChunkKeys(requests[1].items(0).flat_trajectory()),
      ElementsAre(requests[0].chunks(0).chunk_key()));
}

TEST(WriterTest, DoesNotRetryOnNonTransientError) {
  std::vector<InsertStreamRequest> requests;
  auto stub =
      MakeFlakyStub(&requests, 0, 1, ToGrpcStatus(absl::InternalError("")));
  Writer writer(stub, 2, 10);

  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  EXPECT_FALSE(writer.CreateItem("dist", 1, 1.0).ok());

  EXPECT_THAT(requests, SizeIs(1));  // Tries only once and then gives up.
}

TEST(WriterTest, CloseDoesntRetryIfRetriesDisabled) {
  std::vector<InsertStreamRequest> requests;
  auto stub =
      MakeFlakyStub(&requests, 0, 1, ToGrpcStatus(absl::UnavailableError("")));
  Writer writer(stub, 2, 10);

  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.CreateItem("dist", 1, 1.0));
  REVERB_ASSERT_OK(writer.Close(false));
  EXPECT_THAT(requests, SizeIs(1));  // Tries only once and then gives up.
}

TEST(WriterTest, CallsCloseWhenObjectDestroyed) {
  std::vector<InsertStreamRequest> requests;
  {
    auto stub = MakeGoodStub(&requests);
    Writer writer(stub, 2, 10);
    REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
    REVERB_ASSERT_OK(writer.CreateItem("dist", 1, 1.0));
    EXPECT_THAT(requests, SizeIs(0));
  }
  ASSERT_THAT(requests, SizeIs(1));
}

TEST(WriterTest, ResendsOnlyTheChunksTheRemainingItemsNeedWithNewStream) {
  std::vector<InsertStreamRequest> requests;
  auto stub =
      MakeFlakyStub(&requests, 3, 1, ToGrpcStatus(absl::UnavailableError("")));
  Writer writer(stub, 2, 10);

  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.CreateItem("dist", 3, 1.0));
  REVERB_ASSERT_OK(writer.CreateItem("dist2", 1, 1.0));
  EXPECT_THAT(requests, SizeIs(0));

  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));

  ASSERT_THAT(requests, SizeIs(2));
  EXPECT_THAT(requests[0], IsChunk());
  EXPECT_THAT(requests[1], IsChunk());
  auto first_chunk_key = requests[0].chunks(0).chunk_key();
  auto second_chunk_key = requests[0].chunks(1).chunk_key();

  EXPECT_THAT(requests[0],
              IsItemWithRangeAndPriorityAndTable(0, 3, 1.0, "dist"));
  EXPECT_THAT(
      internal::GetChunkKeys(requests[0].items(0).flat_trajectory()),
      ElementsAre(first_chunk_key, second_chunk_key));

  EXPECT_THAT(requests[1], IsItemWithRangeAndPriorityAndTable(
                               0, 1, 1.0, "dist2"));  // Failed
  EXPECT_THAT(
      internal::GetChunkKeys(requests[1].items(0).flat_trajectory()),
      ElementsAre(second_chunk_key));

  // Stream is opened and only the second chunk is sent again.
  EXPECT_THAT(requests[1], IsChunk());
  EXPECT_THAT(requests[1],
              IsItemWithRangeAndPriorityAndTable(0, 1, 1.0, "dist2"));
  EXPECT_THAT(
      internal::GetChunkKeys(requests[1].items(0).flat_trajectory()),
      ElementsAre(second_chunk_key));
}

TEST(WriterTest, TellsServerToKeepStreamedItemsStillInClient) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Writer writer(stub, 2, 6);

  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.CreateItem("dist", 1, 1.0));

  ASSERT_THAT(requests, SizeIs(1));
  EXPECT_THAT(requests[0], IsChunk());
  auto first_chunk_key = requests[0].chunks(0).chunk_key();

  EXPECT_THAT(requests[0],
              IsItemWithRangeAndPriorityAndTable(1, 1, 1.0, "dist"));
  EXPECT_THAT(requests[0].keep_chunk_keys(), ElementsAre(first_chunk_key));

  requests.clear();

  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));

  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.CreateItem("dist", 1, 1.0));

  ASSERT_THAT(requests, SizeIs(1));
  EXPECT_THAT(requests[0], IsChunk());
  auto third_chunk_key = requests[0].chunks(0).chunk_key();

  EXPECT_THAT(requests[0],
              IsItemWithRangeAndPriorityAndTable(1, 1, 1.0, "dist"));
  EXPECT_THAT(requests[0].keep_chunk_keys(),
              ElementsAre(first_chunk_key, third_chunk_key));

  requests.clear();

  // Now the first chunk will go out of scope
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.CreateItem("dist", 1, 1.0));

  ASSERT_THAT(requests, SizeIs(1));
  EXPECT_THAT(requests[0], IsChunk());
  auto forth_chunk_key = requests[0].chunks(0).chunk_key();

  EXPECT_THAT(requests[0],
              IsItemWithRangeAndPriorityAndTable(1, 1, 1.0, "dist"));
  EXPECT_THAT(requests[0].keep_chunk_keys(),
              ElementsAre(third_chunk_key, forth_chunk_key));
}

TEST(WriterTest, IgnoresCloseErrorsIfAllItemsWritten) {
  std::vector<InsertStreamRequest> requests;
  auto stub =
      MakeFlakyStub(&requests, /*num_success=*/2,
                    /*num_fail=*/1, ToGrpcStatus(absl::InternalError("")));
  Writer writer(stub, /*chunk_length=*/1, /*max_timesteps=*/2);

  // Insert an item and make sure it is flushed to the server.
  REVERB_EXPECT_OK(writer.Append(MakeTimestep()));
  REVERB_EXPECT_OK(writer.CreateItem("dist", 1, 1.0));
  EXPECT_THAT(requests, SizeIs(1));

  // Close the writer without any pending items and check that it swallows
  // the error.
  REVERB_EXPECT_OK(writer.Close());
}

TEST(WriterTest, ReturnsCloseErrorsIfAllItemsNotWritten) {
  std::vector<InsertStreamRequest> requests;
  auto stub =
      MakeFlakyStub(&requests, /*num_success=*/0,
                    /*num_fail=*/1, ToGrpcStatus(absl::InternalError("")));
  Writer writer(stub, /*chunk_length=*/2, /*max_timesteps=*/4);

  // Insert an item which is shorter
  // than the batch and thus should not
  // be automatically flushed.
  REVERB_EXPECT_OK(writer.Append(MakeTimestep()));
  REVERB_EXPECT_OK(writer.CreateItem("dist", 1, 1.0));
  EXPECT_THAT(requests, SizeIs(0));

  // Since not all items where sent
  // before the error should be
  // returned.
  EXPECT_EQ(writer.Close().code(), absl::StatusCode::kInternal);
}

TEST(WriterTest, FlushWritesItem) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Writer writer(stub, /*chunk_length=*/2, /*max_timesteps=*/4);

  // Insert an item which is shorter than the batch and thus should not be
  // automatically flushed.
  REVERB_EXPECT_OK(writer.Append(MakeTimestep()));
  EXPECT_THAT(requests, SizeIs(0));
  // No Item, Flush does nothing.
  REVERB_EXPECT_OK(writer.Flush());
  EXPECT_THAT(requests, SizeIs(0));
  REVERB_EXPECT_OK(writer.CreateItem("dist", 1, 1.0));
  EXPECT_THAT(requests, SizeIs(0));
  // Flush the item and make sure it doesn't result in an error.
  REVERB_EXPECT_OK(writer.Flush());
  EXPECT_THAT(requests, SizeIs(1));
  EXPECT_THAT(requests[0], IsChunk());
  EXPECT_THAT(requests[0],
              IsItemWithRangeAndPriorityAndTable(0, 1, 1.0, "dist"));

  // Repeat.
  REVERB_EXPECT_OK(writer.Append(MakeTimestep()));
  EXPECT_THAT(requests, SizeIs(1));
  REVERB_EXPECT_OK(writer.CreateItem("dist", 1, 1.0));
  EXPECT_THAT(requests, SizeIs(1));
  REVERB_EXPECT_OK(writer.Flush());
  EXPECT_THAT(requests, SizeIs(2));
  EXPECT_THAT(requests[1], IsChunk());
  EXPECT_THAT(requests[1],
              IsItemWithRangeAndPriorityAndTable(0, 1, 1.0, "dist"));
}

TEST(WriterTest, SequenceRangeIsSetOnChunks) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Writer writer(stub, /*chunk_length=*/2,
                /*max_timesteps=*/4);

  REVERB_EXPECT_OK(writer.Append(MakeTimestep()));
  REVERB_EXPECT_OK(writer.Append(MakeTimestep()));
  REVERB_EXPECT_OK(writer.Append(MakeTimestep()));
  REVERB_EXPECT_OK(writer.CreateItem("dist", 3, 1.0));
  REVERB_EXPECT_OK(writer.Append(MakeTimestep()));

  EXPECT_THAT(requests[0].chunks().Get(0),
              Partially(testing::EqualsProto("sequence_range: { end: 1 } ")));
  EXPECT_THAT(
      requests[0].chunks().Get(1),
      Partially(testing::EqualsProto("sequence_range: { start: 2, end: 3 } ")));
  EXPECT_THAT(requests[0],
              IsItemWithRangeAndPriorityAndTable(0, 3, 1.0, "dist"));

  EXPECT_NE(requests[0].chunks(0).sequence_range().episode_id(), 0);
  EXPECT_EQ(requests[0].chunks(0).sequence_range().episode_id(),
            requests[0].chunks(1).sequence_range().episode_id());
}

TEST(WriterTest, DeltaEncode) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Writer writer(stub, /*chunk_length=*/2,
                /*max_timesteps=*/4, /*delta_encoded=*/true);

  REVERB_EXPECT_OK(writer.Append(MakeTimestep()));
  REVERB_EXPECT_OK(writer.Append(MakeTimestep()));
  REVERB_EXPECT_OK(writer.Append(MakeTimestep()));
  REVERB_EXPECT_OK(writer.CreateItem("dist", 3, 1.0));
  REVERB_EXPECT_OK(writer.Append(MakeTimestep()));

  EXPECT_THAT(requests[0],
              Partially(testing::EqualsProto(
                  "chunks: { sequence_range: { start: 0 end: 1 } "
                  "delta_encoded: true } chunks: { sequence_range: { start: 2 "
                  "end: 3 } delta_encoded: true }")));
  EXPECT_THAT(requests[0],
              IsItemWithRangeAndPriorityAndTable(0, 3, 1.0, "dist"));
}

TEST(WriterTest, MultiChunkItemsAreCorrect) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Writer writer(stub, /*chunk_length=*/3,
                /*max_timesteps=*/4, /*delta_encoded=*/false);

  // We create two chunks with 5 time steps (t_0,.., t_4) and 3 sequences
  // (s_0, s_1, s_2):
  // +--- CHUNK0 --+- CHUNK1 -+
  // | t_0 t_1 t_2 | t_3 t_4  |
  // +-------------+----------+
  // | s_0 s_0 s_1 | s_1 s_3  |
  // +-------------+----------+

  // First item: 1 chunk.
  REVERB_EXPECT_OK(writer.Append(MakeTimestep()));
  REVERB_EXPECT_OK(writer.Append(MakeTimestep()));
  REVERB_EXPECT_OK(writer.CreateItem("dist", 2, 1.0));

  // Second item: 2 chunks.
  REVERB_EXPECT_OK(writer.Append(MakeTimestep()));
  REVERB_EXPECT_OK(writer.Append(MakeTimestep()));
  REVERB_EXPECT_OK(writer.CreateItem("dist", 2, 1.0));

  // Third item: 1 chunk.
  REVERB_EXPECT_OK(writer.Append(MakeTimestep()));
  REVERB_EXPECT_OK(writer.CreateItem("dist", 1, 1.0));

  REVERB_EXPECT_OK(writer.Close());

  EXPECT_THAT(requests[0], Partially(testing::EqualsProto(
                               "chunks: { sequence_range: { start: 0 "
                               "end: 2 } delta_encoded: false }")));
  EXPECT_THAT(requests[0],
              IsItemWithRangeAndPriorityAndTable(0, 2, 1.0, "dist"));
  EXPECT_THAT(requests[1], Partially(testing::EqualsProto(
                               "chunks: { sequence_range: { start: 3 "
                               "end: 4 } delta_encoded: false }")));
  EXPECT_THAT(requests[1],
              IsItemWithRangeAndPriorityAndTable(2, 2, 1.0, "dist"));
  EXPECT_THAT(requests[2],
              IsItemWithRangeAndPriorityAndTable(1, 1, 1.0, "dist"));

  EXPECT_THAT(
      internal::GetChunkKeys(requests[0].items(0).flat_trajectory()),
      SizeIs(1));
  EXPECT_THAT(
      internal::GetChunkKeys(requests[1].items(0).flat_trajectory()),
      SizeIs(2));
  EXPECT_THAT(
      internal::GetChunkKeys(requests[2].items(0).flat_trajectory()),
      SizeIs(1));
}

TEST(WriterTest, DataUncompressedSizeIsPopulatedInChunks) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Writer writer(stub, /*chunk_length=*/2,
                /*max_timesteps=*/2, /*delta_encoded=*/true);

  REVERB_EXPECT_OK(writer.Append(MakeTimestep()));
  REVERB_EXPECT_OK(writer.Append(MakeTimestep()));
  REVERB_EXPECT_OK(writer.CreateItem("dist", 2, 1.0));

  ASSERT_THAT(requests, SizeIs(1));
  ASSERT_THAT(requests[0].chunks(), SizeIs(1));
  EXPECT_GT(requests[0].chunks(0).data_uncompressed_size(), 0);
}

TEST(WriterTest, WriteTimeStepsMatchingSignature) {
  std::vector<InsertStreamRequest> requests;
  ::reverb::tensor::SignatureProto signature =
      MakeSignature(DataType::Float32, {});
  auto stub = MakeGoodStub(&requests, &signature);
  Client client(stub);
  std::unique_ptr<Writer> writer;
  REVERB_EXPECT_OK(client.NewWriter(2, 6, /*delta_encoded=*/false, &writer));

  REVERB_ASSERT_OK(writer->Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer->Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer->CreateItem("dist", 2, 1.0));
  ASSERT_THAT(requests, SizeIs(1));
}

TEST(WriterTest, WriteTimeStepsMatchingBoundedSignature) {
  std::vector<InsertStreamRequest> requests;
  ::reverb::tensor::SignatureProto signature =
      MakeBoundedTensorSpecSignature(DataType::Float32, {});
  auto stub = MakeGoodStub(&requests, &signature);
  Client client(stub);
  std::unique_ptr<Writer> writer;
  REVERB_EXPECT_OK(client.NewWriter(2, 6, /*delta_encoded=*/false, &writer));

  REVERB_ASSERT_OK(writer->Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer->Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer->CreateItem("dist", 2, 1.0));
  ASSERT_THAT(requests, SizeIs(1));
}

TEST(WriterTest, WriteTimeStepsNumTensorsDontMatchSignatureError) {
  std::vector<InsertStreamRequest> requests;
  ::reverb::tensor::SignatureProto signature = MakeSignature();
  auto stub = MakeGoodStub(&requests, &signature);
  Client client(stub);
  std::unique_ptr<Writer> writer;
  REVERB_EXPECT_OK(client.NewWriter(2, 6, /*delta_encoded=*/false, &writer));

  REVERB_ASSERT_OK(writer->Append(MakeTimestep(/*num_tensors=*/2)));
  REVERB_ASSERT_OK(writer->Append(MakeTimestep(/*num_tensors=*/2)));
  auto status = writer->CreateItem("dist", 2, 1.0);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "Append for timestep offset 0 was called with 2 tensors, "
                  "but table requires 1 tensors per entry."));
}

TEST(WriterTest, WriteTimeStepsWithoutSignatureTensorShapesNotConsistentError) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Client client(stub);
  std::unique_ptr<Writer> writer;
  REVERB_EXPECT_OK(client.NewWriter(2, 6, /*delta_encoded=*/false, &writer));

  REVERB_ASSERT_OK(writer->Append(MakeTimestep(/*num_tensors=*/1, /*shape=*/{2})));
  auto status = writer->Append(MakeTimestep(/*num_tensors=*/1, /*shape=*/{1}));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      std::string(status.message()),
      ::testing::HasSubstr(
          "Unable to concatenate tensors at index 0 due to mismatched shapes."
          "  Tensor 0 has shape:"));
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("but tensor 1 has shape:"));
}

TEST(WriterTest, WriteTimeStepsNumTensorsDontMatchBoundedSignatureError) {
  std::vector<InsertStreamRequest> requests;
  ::reverb::tensor::SignatureProto signature = MakeBoundedTensorSpecSignature();
  auto stub = MakeGoodStub(&requests, &signature);
  Client client(stub);
  std::unique_ptr<Writer> writer;
  REVERB_EXPECT_OK(client.NewWriter(2, 6, /*delta_encoded=*/false, &writer));

  REVERB_ASSERT_OK(writer->Append(MakeTimestep(/*num_tensors=*/2)));
  REVERB_ASSERT_OK(writer->Append(MakeTimestep(/*num_tensors=*/2)));
  auto status = writer->CreateItem("dist", 2, 1.0);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "Append for timestep offset 0 was called with 2 tensors, "
                  "but table requires 1 tensors per entry."));
}

TEST(WriterTest, WriteTimeStepsInconsistentDtypeError) {
  std::vector<InsertStreamRequest> requests;
  ::reverb::tensor::SignatureProto signature = MakeSignature(DataType::Int32);
  auto stub = MakeGoodStub(&requests, &signature);
  Client client(stub);
  std::unique_ptr<Writer> writer;
  REVERB_EXPECT_OK(client.NewWriter(2, 6, /*delta_encoded=*/false, &writer));

  REVERB_ASSERT_OK(writer->Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer->Append(MakeTimestep()));
  auto status = writer->CreateItem("dist", 2, 1.0);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "timestep offset 0, flattened index 0, saw a tensor of "
                  "dtype Float32"));
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "but expected tensor 'tensor0' of dtype Int32"));
}

TEST(WriterTest, WriteTimeStepsInconsistentDtypeErrorAgainstBoundedSpec) {
  std::vector<InsertStreamRequest> requests;
  ::reverb::tensor::SignatureProto signature =
      MakeBoundedTensorSpecSignature(DataType::Int32);
  auto stub = MakeGoodStub(&requests, &signature);
  Client client(stub);
  std::unique_ptr<Writer> writer;
  REVERB_EXPECT_OK(client.NewWriter(2, 6, /*delta_encoded=*/false, &writer));

  REVERB_ASSERT_OK(writer->Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer->Append(MakeTimestep()));
  auto status = writer->CreateItem("dist", 2, 1.0);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "timestep offset 0, flattened index 0, saw a tensor of "
                  "dtype Float32"));
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "but expected tensor 'tensor0' of dtype Int32"));
}

TEST(WriterTest, WriteTimeStepsInconsistentShapeError) {
  std::vector<InsertStreamRequest> requests;
  ::reverb::tensor::SignatureProto signature =
      MakeSignature(DataType::Float32, {5});
  auto stub = MakeGoodStub(&requests, &signature);
  Client client(stub);
  std::unique_ptr<Writer> writer;
  REVERB_EXPECT_OK(client.NewWriter(2, 6, /*delta_encoded=*/false, &writer));

  REVERB_ASSERT_OK(writer->Append(MakeTimestep(1, {4})));
  REVERB_ASSERT_OK(writer->Append(MakeTimestep(1, {4})));
  auto status = writer->CreateItem("dist", 2, 1.0);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "timestep offset 0, flattened index 0, saw a tensor of "
                  "dtype Float32"));
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "but expected tensor 'tensor0' of dtype Float32 and shape "
                  "compatible with"));
}

TEST(WriterTest, WriteNanPriorityError) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Client client(stub);
  std::unique_ptr<Writer> writer;
  REVERB_EXPECT_OK(client.NewWriter(2, 6, /*delta_encoded=*/false, &writer));

  REVERB_ASSERT_OK(writer->Append(MakeTimestep(/*num_tensors=*/1, /*shape=*/{1})));
  REVERB_ASSERT_OK(writer->Append(MakeTimestep(/*num_tensors=*/1, /*shape=*/{1})));

  auto status = writer->CreateItem("dist", 2, std::nan("1"));

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("`priority` must not be nan."));
}

TEST(WriterTest, WriteTimeStepsInconsistentShapeErrorAgainstBoundedSpec) {
  std::vector<InsertStreamRequest> requests;
  ::reverb::tensor::SignatureProto signature = MakeBoundedTensorSpecSignature(
      DataType::Float32, {3});
  auto stub = MakeGoodStub(&requests, &signature);
  Client client(stub);
  std::unique_ptr<Writer> writer;
  REVERB_EXPECT_OK(client.NewWriter(2, 6, /*delta_encoded=*/false, &writer));

  REVERB_ASSERT_OK(writer->Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer->Append(MakeTimestep()));
  auto status = writer->CreateItem("dist", 2, 1.0);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "timestep offset 0, flattened index 0, saw a tensor of "
                  "dtype Float32"));
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "but expected tensor 'tensor0' of dtype Float32 and shape "
                  "compatible with"));
}

TEST(WriterTest, WriteTrajectoryCompatibleWithSignature) {
  std::vector<InsertStreamRequest> requests;
  ::reverb::tensor::SignatureProto signature =
      MakeSignature(DataType::Float32, {2});
  auto stub = MakeGoodStub(&requests, &signature);
  Client client(stub);
  std::unique_ptr<Writer> writer;
  REVERB_EXPECT_OK(client.NewWriter(3, 6, /*delta_encoded=*/false, &writer));

  REVERB_ASSERT_OK(writer->Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer->Append(MakeTimestep()));

  // Writing an item of steps should be fine as it matches the trajectory.
  REVERB_EXPECT_OK(writer->CreateItem("dist", 2, 1.0));

  // Writing an item of 3 steps does not match the trajectory signature and thus
  // should fail.
  REVERB_ASSERT_OK(writer->Append(MakeTimestep()));
  auto status = writer->CreateItem("dist", 3, 1.0);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "timestep offset 0, flattened index 0, saw a tensor of "
                  "dtype Float32"));
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "but expected tensor 'tensor0' of dtype Float32 and shape "
                  "compatible with"));
}

std::pair<std::shared_ptr<FakeStub>, std::shared_ptr<internal::Queue<uint64_t>>>
MakeStubWithExplicitResponseQueue(std::vector<InsertStreamRequest>* requests) {
  FakeInsertStream* stream = new FakeInsertStream(
      requests, 10000, ToGrpcStatus(absl::InternalError("")),
      /*automatic_response_ids=*/false);
  auto stub = std::make_shared<FakeStub>(std::list<FakeInsertStream*>{stream});
  return {std::move(stub), stream->response_ids()};
}

TEST(WriterTest, CloseBlocksUntilAllItemsConfirmed) {
  std::vector<InsertStreamRequest> requests;
  auto pair = MakeStubWithExplicitResponseQueue(&requests);
  auto response_ids = std::move(pair.second);
  Writer writer(pair.first, 2, 2, false, nullptr, 100);

  // Creating the items should not result in any blocking as the number of in
  // flight items is 100.
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.CreateItem("dist", 1, 1.0));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.CreateItem("dist", 2, 1.0));

  // Attempt to close the writer from a separate thread.
  absl::Notification notification;
  auto close_thread = internal::StartThread("Close", [&writer, &notification] {
    REVERB_CHECK(writer.Close().ok());
    notification.Notify();
  });

  // The close call should not be able to complete as the server hasn't
  // confirmed that all the IDs have been written.
  EXPECT_FALSE(
      notification.WaitForNotificationWithTimeout(kNotificationTimeout));

  // Sending the response for the first item should not be enough.
  ASSERT_TRUE(response_ids->Reserve(1));
  response_ids->PushBatch({1});
  EXPECT_FALSE(
      notification.WaitForNotificationWithTimeout(kNotificationTimeout));

  // Sending response for the second (and last) item should unblock the Close
  // call.
  ASSERT_TRUE(response_ids->Reserve(1));
  response_ids->PushBatch({2});
  notification.WaitForNotification();
}

TEST(WriterTest, FlushBlocksUntilAllItemsConfirmed) {
  std::vector<InsertStreamRequest> requests;
  auto pair = MakeStubWithExplicitResponseQueue(&requests);
  auto response_ids = std::move(pair.second);
  Writer writer(pair.first, 2, 2, false, nullptr, 100);

  // Creating the items should not result in any blocking as the number of in
  // flight items is 100.
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.CreateItem("dist", 1, 1.0));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.CreateItem("dist", 2, 1.0));

  // Attempt to flush the writer from a separate thread.
  absl::Notification notification;
  auto flush_thread = internal::StartThread("Flush", [&writer, &notification] {
    REVERB_CHECK(writer.Flush().ok());
    notification.Notify();
  });

  // The flush call should not be able to complete as the server hasn't
  // confirmed that all the IDs have been written.
  EXPECT_FALSE(
      notification.WaitForNotificationWithTimeout(kNotificationTimeout));

  // Sending the response for the first item should not be enough.
  ASSERT_TRUE(response_ids->Reserve(1));
  response_ids->PushBatch({1});
  EXPECT_FALSE(
      notification.WaitForNotificationWithTimeout(kNotificationTimeout));

  // Sending response for the second (and last) item should unblock the Close
  // call.
  ASSERT_TRUE(response_ids->Reserve(1));
  response_ids->PushBatch({2});
  notification.WaitForNotification();
}

TEST(WriterTest, BlocksWhenMaxInFlighItemsReached) {
  std::vector<InsertStreamRequest> requests;
  auto pair = MakeStubWithExplicitResponseQueue(&requests);
  auto response_ids = std::move(pair.second);
  Writer writer(pair.first, 1, 2, false, nullptr, 2);

  // Creating two items should not result in any blocking as it does not exceed
  // the maximum number of in flight items.
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.CreateItem("dist", 1, 1.0));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.CreateItem("dist", 2, 1.0));

  // Attempt to send one more items in a separate thread.
  absl::Notification notification;
  auto thread = internal::StartThread("InsertItem", [&writer, &notification] {
    REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
    REVERB_ASSERT_OK(writer.CreateItem("dist", 2, 1.0));
    notification.Notify();
  });

  // Initially the third item should be blocked.
  ASSERT_FALSE(
      notification.WaitForNotificationWithTimeout(kNotificationTimeout));

  // Confirming one item should unblock the pending item.
  ASSERT_TRUE(response_ids->Reserve(1));
  response_ids->PushBatch({1});
  notification.WaitForNotification();

  // Send the remaining responses to unblock any pending reads without
  // cancelling the stream.
  ASSERT_TRUE(response_ids->Reserve(2));
  response_ids->PushBatch({2, 3});
}

TEST(WriterTest, HandlesBatchedResponses) {
  std::vector<InsertStreamRequest> requests;
  auto pair = MakeStubWithExplicitResponseQueue(&requests);
  auto response_ids = std::move(pair.second);
  Writer writer(pair.first, 1, 4, false, nullptr, 2);

  ASSERT_TRUE(response_ids->Reserve(2));
  response_ids->PushBatch({2, 3});

  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.CreateItem("dist", 1, 1.0));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.CreateItem("dist", 2, 1.0));

  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.CreateItem("dist", 3, 1.0));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep()));
  REVERB_ASSERT_OK(writer.CreateItem("dist", 4, 1.0));

  ASSERT_TRUE(response_ids->Reserve(2));
  response_ids->PushBatch({3, 4});
}

TEST(WriterTest, AppendSequenceBehavesLikeMutlipleAppendCalls) {
  const auto kBatchSize = 10;
  const auto kChunkLength = 5;
  const auto kTensorsPerStep = 3;

  std::vector<std::vector<Tensor>> steps;
  for (int i = 0; i < kBatchSize; i++) {
    steps.push_back(MakeTimestep(kTensorsPerStep, {1}));
  }

  // Concatenate each column across all steps (mirrors the original
  // Concat usage but via TensorBuffer::Concat).
  std::vector<Tensor> batch(kTensorsPerStep);
  for (int i = 0; i < kTensorsPerStep; i++) {
    std::vector<Tensor> column(kBatchSize);
    for (int j = 0; j < kBatchSize; ++j) column[j] = steps[j][i];
    auto concat_result = TensorBuffer::Concat(column);
    REVERB_ASSERT_OK(concat_result);
    batch[i] = std::move(*concat_result);
  }

  std::vector<InsertStreamRequest> simple_requests;
  std::vector<InsertStreamRequest> batch_requests;

  auto simple_stub = MakeGoodStub(&simple_requests);
  auto batch_stub = MakeGoodStub(&batch_requests);

  Writer simple_writer(simple_stub, kChunkLength, kBatchSize);
  Writer batch_writer(batch_stub, kChunkLength, kBatchSize);

  for (const auto& step : steps) {
    REVERB_ASSERT_OK(simple_writer.Append(step));
  }

  REVERB_ASSERT_OK(batch_writer.AppendSequence(batch));

  EXPECT_EQ(simple_requests.size(), batch_requests.size());
  for (int i = 0; i < simple_requests.size(); i++) {
    EXPECT_THAT(simple_requests[i], testing::EqualsProto(batch_requests[i]));
  }
}

TEST(WriterTest, AppendSequenceCalledWithScalar) {
  std::vector<InsertStreamRequest> requests;
  Writer writer(MakeGoodStub(&requests), 1, 1);
  auto status = writer.AppendSequence({MakeTensor(kFloatSpec)});
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "AppendSequence called with scalar tensor at index 0."));
}

TEST(WriterTest, AppendSequenceCalledWithEmptyBatch) {
  std::vector<InsertStreamRequest> requests;
  Writer writer(MakeGoodStub(&requests), 1, 1);
  auto status = writer.AppendSequence({});
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("AppendSequence called with empty data."));
}

TEST(WriterTest, AppendSequenceCalledWithNonEqualBatchSizes) {
  std::vector<InsertStreamRequest> requests;
  Writer writer(MakeGoodStub(&requests), 1, 1);
  auto status = writer.AppendSequence({
      MakeTensor(internal::TensorSpec{"", DataType::Float32, {2, 2}}),
      MakeTensor(internal::TensorSpec{"", DataType::Float32, {3}}),
  });
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      std::string(status.message()),
      ::testing::HasSubstr(
          "AppendSequence called with tensors of non equal batch dimension:"));
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("dtype: Float32, shape:"));
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("0: Tensor<name: ''"));
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("1: Tensor<name: ''"));
}

// ---------------------------------------------------------------------------
// Local Writer path: writes directly into a {name -> Table} map via
// Table::InsertOrAssignAsync, bypassing gRPC. (D3-a.)
// ---------------------------------------------------------------------------

// Builds a {name -> Table} map for local-Writer tests.
internal::flat_hash_map<std::string, std::shared_ptr<Table>>
MakeLocalWriterTables(std::vector<std::pair<std::string, int>> named_sizes) {
  internal::flat_hash_map<std::string, std::shared_ptr<Table>> tables;
  for (auto& [name, max_size] : named_sizes) {
    tables[name] = std::make_shared<Table>(
        /*name=*/name,
        /*sampler=*/std::make_shared<FifoSelector>(),
        /*remover=*/std::make_shared<FifoSelector>(),
        /*max_size=*/max_size,
        /*max_times_sampled=*/1,
        /*rate_limiter=*/std::make_shared<RateLimiter>(1, 1, 0, max_size));
  }
  return tables;
}

TEST(WriterTest, LocalWriterRoundTrip) {
  auto tables = MakeLocalWriterTables({{"dist", 10}});
  Writer writer(tables, /*chunk_length=*/1, /*max_timesteps=*/1,
                /*delta_encoded=*/false, /*signatures=*/nullptr,
                /*max_in_flight_items=*/Writer::kDefaultMaxInFlightItems);

  REVERB_ASSERT_OK(writer.Append(MakeTimestep(/*num_tensors=*/1, /*shape=*/{1})));
  REVERB_ASSERT_OK(writer.CreateItem("dist", /*num_timesteps=*/1, 1.0));
  REVERB_ASSERT_OK(writer.Close());

  EXPECT_EQ(tables["dist"]->size(), 1);
  Table::SampledItem sampled;
  REVERB_ASSERT_OK(tables["dist"]->Sample(&sampled));
  EXPECT_EQ(sampled.ref->table(), "dist");
  EXPECT_EQ(sampled.ref->priority(), 1.0);
  ASSERT_EQ(sampled.ref->chunks().size(), 1);
}

TEST(WriterTest, LocalWriterDispatchesToMultipleTables) {
  auto tables = MakeLocalWriterTables({{"a", 10}, {"b", 10}});
  Writer writer(tables, /*chunk_length=*/1, /*max_timesteps=*/1,
                /*delta_encoded=*/false, /*signatures=*/nullptr,
                /*max_in_flight_items=*/Writer::kDefaultMaxInFlightItems);

  REVERB_ASSERT_OK(writer.Append(MakeTimestep(/*num_tensors=*/1, /*shape=*/{1})));
  REVERB_ASSERT_OK(writer.CreateItem("a", /*num_timesteps=*/1, 1.0));
  REVERB_ASSERT_OK(writer.Append(MakeTimestep(/*num_tensors=*/1, /*shape=*/{1})));
  REVERB_ASSERT_OK(writer.CreateItem("b", /*num_timesteps=*/1, 2.0));
  REVERB_ASSERT_OK(writer.Close());

  EXPECT_EQ(tables["a"]->size(), 1);
  EXPECT_EQ(tables["b"]->size(), 1);

  Table::SampledItem sa;
  REVERB_ASSERT_OK(tables["a"]->Sample(&sa));
  EXPECT_EQ(sa.ref->table(), "a");
  EXPECT_EQ(sa.ref->priority(), 1.0);

  Table::SampledItem sb;
  REVERB_ASSERT_OK(tables["b"]->Sample(&sb));
  EXPECT_EQ(sb.ref->table(), "b");
  EXPECT_EQ(sb.ref->priority(), 2.0);
}

TEST(WriterTest, LocalWriterConfirmItemsEquivalence) {
  // max_in_flight_items=1 forces the local ConfirmItems backpressure gate to
  // engage: after one in-flight insert, the next CreateItem blocks in
  // ConfirmItems(0) until the table completes the insert (callback decrements
  // num_items_in_flight_). We drain samples from a separate thread to let the
  // table complete inserts, then assert Close() drains everything.
  auto tables = MakeLocalWriterTables({{"dist", 10}});
  Writer writer(tables, /*chunk_length=*/1, /*max_timesteps=*/1,
                /*delta_encoded=*/false, /*signatures=*/nullptr,
                /*max_in_flight_items=*/1);

  absl::Notification done;
  std::vector<absl::Status> statuses;
  internal::StartThread("InsertItems", [&] {
    for (int i = 0; i < 4; ++i) {
      statuses.push_back(writer.Append(MakeTimestep(/*num_tensors=*/1,
                                                   /*shape=*/{1})));
      statuses.push_back(writer.CreateItem("dist", /*num_timesteps=*/1, 1.0));
    }
    done.Notify();
  });

  // Drain samples so the rate limiter allows inserts to complete, unblocking
  // the blocked ConfirmItems(0) calls. Bound the wait so a deadlock fails the
  // test instead of hanging forever.
  absl::Time deadline = absl::Now() + absl::Seconds(10);
  while (!done.WaitForNotificationWithTimeout(absl::Milliseconds(20))) {
    Table::SampledItem sampled;
    tables["dist"]->Sample(&sampled).IgnoreError();
    if (absl::Now() > deadline) {
      ADD_FAILURE() << "Insert thread did not complete within 10s";
      break;
    }
  }
  for (const auto& s : statuses) {
    EXPECT_TRUE(s.ok()) << s;
  }

  REVERB_ASSERT_OK(writer.Close());
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
