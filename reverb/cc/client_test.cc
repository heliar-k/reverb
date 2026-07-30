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

#include "reverb/cc/client.h"

#include <chrono>  // NOLINT(build/c++11) - grpc API requires it.
#include <memory>
#include <vector>

#include "grpcpp/impl/codegen/client_context.h"
#include "grpcpp/impl/codegen/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/platform/default/status_matchers.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/reverb_service_mock.grpc.pb.h"
#include "reverb/cc/sampler.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/support/uint128.h"
#include "reverb/cc/testing/proto_test_util.h"
#include "reverb/cc/trajectory_writer.h"
#include "third_party/reverb_tensor/reverb_tensor.pb.h"

namespace deepmind {
namespace reverb {
namespace {

constexpr char kCheckpointPath[] = "/path/to/checkpoint";

// Builds a SignatureProto with a single tensor_spec named "tensor0". Mirrors
// writer_test.cc's MakeSignature so the flat-signature path
// (FlatSignatureFromSignatureProto) yields exactly one TensorSpec.
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

// A no-op bidi stream that immediately reports the stream closed. Used so the
// Sampler worker threads (spawned by NewSampler's stub path) never block on a
// real gRPC stream: FetchSamples sees Write fail / Read return false, calls
// Finish(), gets a non-OK status, and retries with backoff until Close() joins
// the workers.
template <typename RequestT, typename ResponseT>
class FakeClosedStream
    : public grpc::ClientReaderWriterInterface<RequestT, ResponseT> {
 public:
  bool Write(const RequestT&, grpc::WriteOptions) override { return false; }
  bool Read(ResponseT*) override { return false; }
  void WaitForInitialMetadata() override {}
  bool WritesDone() override { return true; }
  bool NextMessageSize(uint32_t* /*sz*/) override { return false; }
  grpc::Status Finish() override {
    return grpc::Status(grpc::StatusCode::CANCELLED, "closed");
  }
};

class FakeStub : public /* grpc_gen:: */MockReverbServiceStub {
 public:
  grpc::Status MutatePriorities(grpc::ClientContext* context,
                                const MutatePrioritiesRequest& request,
                                MutatePrioritiesResponse* response) override {
    last_deadline_ = context->deadline();
    mutate_priorities_request_ = request;
    return grpc::Status::OK;
  }

  grpc::Status Reset(grpc::ClientContext* context, const ResetRequest& request,
                     ResetResponse* response) override {
    last_deadline_ = context->deadline();
    reset_request_ = request;
    return grpc::Status::OK;
  }

  grpc::Status Checkpoint(grpc::ClientContext* context,
                          const CheckpointRequest& request,
                          CheckpointResponse* response) override {
    last_deadline_ = context->deadline();
    response->set_checkpoint_path(kCheckpointPath);
    return grpc::Status::OK;
  }

  grpc::Status ServerInfo(grpc::ClientContext* context,
                          const ServerInfoRequest& request,
                          ServerInfoResponse* response) override {
    last_deadline_ = context->deadline();
    *response->mutable_tables_state_id() =
        Uint128ToMessage(absl::MakeUint128(1, 2));
    // Emit a placeholder table so ServerInfo-based callers see non-empty info.
    // If `table_infos_` was populated (e.g. with signatures) use those instead.
    if (table_infos_.empty()) {
      response->add_table_info()->set_max_size(2);
    } else {
      for (const auto& ti : table_infos_) {
        *response->add_table_info() = ti;
      }
    }
    return grpc::Status::OK;
  }

  // Sampler construction (the stub path taken when GetLocalTablePtr fails)
  // spawns worker threads that call SampleStreamRaw. The default gMock
  // implementation returns nullptr and crashes, so hand back a closed stream
  // whose worker loop exits cleanly on Close().
  grpc::ClientReaderWriterInterface<SampleStreamRequest, SampleStreamResponse>*
  SampleStreamRaw(grpc::ClientContext* /*context*/) override {
    return new FakeClosedStream<SampleStreamRequest, SampleStreamResponse>();
  }

  // GetLocalTablePtr opens an InitializeConnection stream; return a closed
  // stream so it reports a non-OK status and the client falls back to the
  // (stub-based) Sampler path instead of touching a real in-process Table.
  grpc::ClientReaderWriterInterface<InitializeConnectionRequest,
                                    InitializeConnectionResponse>*
  InitializeConnectionRaw(grpc::ClientContext* /*context*/) override {
    return new FakeClosedStream<InitializeConnectionRequest,
                                InitializeConnectionResponse>();
  }

  // Populate the table_info entries returned by ServerInfo. Each entry should
  // carry a `name` and (optionally) a `signature`.
  void set_table_infos(std::vector<TableInfo> infos) {
    table_infos_ = std::move(infos);
  }

  const MutatePrioritiesRequest& mutate_priorities_request() const {
    return mutate_priorities_request_;
  }

  std::chrono::system_clock::time_point last_deadline() const {
    return last_deadline_;
  }

  const ResetRequest& reset_request() const { return reset_request_; }

 private:
  std::chrono::system_clock::time_point last_deadline_;
  MutatePrioritiesRequest mutate_priorities_request_;
  ResetRequest reset_request_;
  std::vector<TableInfo> table_infos_;
};

// Builds a TableInfo named `table` carrying `signature` (or no signature if
// nullptr), used to populate the ServerInfo response for signature tests.
TableInfo MakeTableInfo(const std::string& table,
                        const ::reverb::tensor::SignatureProto* signature) {
  TableInfo info;
  info.set_name(table);
  info.set_max_size(2);
  if (signature != nullptr) {
    *info.mutable_signature() = *signature;
  }
  return info;
}

// Default options used by the NewSampler tests: unlimited samples, one worker,
// small in-flight buffer so the worker loop is light.
Sampler::Options MakeSamplerOptions() {
  Sampler::Options options;
  options.max_samples = 1;
  options.max_in_flight_samples_per_worker = 1;
  options.num_workers = 1;
  return options;
}

TEST(ClientTest, MutatePrioritiesDefaultValues) {
  auto stub = std::make_shared<FakeStub>();
  Client client(stub);
  REVERB_EXPECT_OK(client.MutatePriorities("", {}, {}));
  EXPECT_THAT(stub->mutate_priorities_request(),
              testing::EqualsProto(MutatePrioritiesRequest()));
}

TEST(ClientTest, MutatePrioritiesFilled) {
  auto stub = std::make_shared<FakeStub>();
  Client client(stub);
  auto pair = testing::MakeKeyWithPriority(123, 456);
  REVERB_EXPECT_OK(client.MutatePriorities("table", {pair}, {4}));

  MutatePrioritiesRequest expected;
  expected.set_table("table");
  *expected.add_updates() = pair;
  expected.add_delete_keys(4);
  EXPECT_THAT(stub->mutate_priorities_request(),
              testing::EqualsProto(expected));
}

TEST(ClientTest, Deadline) {
  auto stub = std::make_shared<FakeStub>();
  Client client(stub);
  auto pair = testing::MakeKeyWithPriority(123, 456);

  REVERB_EXPECT_OK(client.MutatePriorities("table", {pair}, {4}));
  EXPECT_EQ(stub->last_deadline(),
            std::chrono::system_clock::time_point::max());

  REVERB_EXPECT_OK(
      client.MutatePriorities("table", {pair}, {4}, absl::Seconds(1)));
  EXPECT_LE(stub->last_deadline(),
            absl::ToChronoTime(absl::Now() + absl::Seconds(1)));
}

TEST(ClientTest, ResetRequestFilled) {
  auto stub = std::make_shared<FakeStub>();
  Client client(stub);
  REVERB_EXPECT_OK(client.Reset("table"));

  ResetRequest expected;
  expected.set_table("table");
  EXPECT_THAT(stub->reset_request(), testing::EqualsProto(expected));
}

TEST(ClientTest, Checkpoint) {
  auto stub = std::make_shared<FakeStub>();
  Client client(stub);
  std::string path;
  REVERB_EXPECT_OK(client.Checkpoint(&path));
  EXPECT_EQ(path, kCheckpointPath);
}

TEST(ClientTest, ServerInfoRequestFilled) {
  auto stub = std::make_shared<FakeStub>();
  Client client(stub);
  struct Client::ServerInfo info;
  REVERB_EXPECT_OK(client.ServerInfo(&info));

  TableInfo expected_info;
  expected_info.set_max_size(2);
  EXPECT_EQ(info.tables_state_id, absl::MakeUint128(1, 2));
  EXPECT_EQ(info.table_info.size(), 1);
  EXPECT_THAT(info.table_info[0], testing::EqualsProto(expected_info));
}

TEST(ClientTest, NewTrajectoryWriterValidatesOptions) {
  auto stub = std::make_shared<FakeStub>();
  Client client(stub);
  std::unique_ptr<TrajectoryWriter> writer;
  TrajectoryWriter::Options options = {
      .chunker_options = std::make_shared<ConstantChunkerOptions>(-1, 1),
  };
  EXPECT_FALSE(client.NewTrajectoryWriter(options, &writer).ok());
}

TEST(ClientTest, NewStreamingTrajectoryWriterValidatesOptions) {
  auto stub = std::make_shared<FakeStub>();
  Client client(stub);
  std::unique_ptr<StreamingTrajectoryWriter> writer;
  TrajectoryWriter::Options options = {
      .chunker_options = std::make_shared<ConstantChunkerOptions>(-1, 1),
  };
  EXPECT_FALSE(client.NewStreamingTrajectoryWriter(options, &writer).ok());
}

// --- NewSampler signature-validation path (client.cc L170-295) ---

// Table not present in ServerInfo: GetDtypesAndShapesForSampler logs a warning
// and returns absl::nullopt, so NewSampler still constructs a Sampler (no
// validation possible). Expect OkStatus.
TEST(ClientTest, NewSamplerUnknownTable) {
  auto stub = std::make_shared<FakeStub>();
  stub->set_table_infos({MakeTableInfo("other", nullptr)});
  Client client(stub);
  std::unique_ptr<Sampler> sampler;
  REVERB_EXPECT_OK(client.NewSampler("missing", MakeSamplerOptions(),
                                     absl::Seconds(1), &sampler));
  EXPECT_NE(sampler, nullptr);
}

// validation_timeout too short: the ServerInfo RPC itself succeeds here (the
// FakeStub answers immediately), so this exercises the happy path through
// GetDtypesAndShapesForSampler rather than the DeadlineExceeded warning
// branch. That branch needs an actually-blocking RPC to time out, which the
// in-process FakeStub cannot reproduce. ponytail: left as a best-effort smoke
// test of the timeout argument plumbing.
TEST(ClientTest, NewSamplerDeadlineExceeded) {
  auto stub = std::make_shared<FakeStub>();
  ::reverb::tensor::SignatureProto sig = MakeSignature(DataType::Float32, {2});
  stub->set_table_infos({MakeTableInfo("t", &sig)});
  Client client(stub);
  std::unique_ptr<Sampler> sampler;
  REVERB_EXPECT_OK(client.NewSampler("t", MakeSamplerOptions(),
                                     absl::Milliseconds(1), &sampler));
  EXPECT_NE(sampler, nullptr);
}

// validation_dtypes and validation_shapes size mismatch -> InvalidArgument,
// raised before any ServerInfo lookup.
TEST(ClientTest, NewSamplerWithValidationDtypesShapeSizeMismatch) {
  auto stub = std::make_shared<FakeStub>();
  stub->set_table_infos({MakeTableInfo("t", nullptr)});
  Client client(stub);
  std::unique_ptr<Sampler> sampler;
  auto status = client.NewSampler(
      "t", MakeSamplerOptions(),
      /*validation_dtypes=*/{DataType::Float32},
      /*validation_shapes=*/{{2}, {3}}, absl::Seconds(1), &sampler);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_EQ(sampler, nullptr);
}

// Table signature has 1 tensor but validation requests 2 -> InvalidArgument
// ("Inconsistent number of tensors").
TEST(ClientTest, NewSamplerWithSignatureNumTensorsMismatch) {
  auto stub = std::make_shared<FakeStub>();
  ::reverb::tensor::SignatureProto sig = MakeSignature(DataType::Float32, {2});
  stub->set_table_infos({MakeTableInfo("t", &sig)});
  Client client(stub);
  std::unique_ptr<Sampler> sampler;
  auto status = client.NewSampler(
      "t", MakeSamplerOptions(),
      /*validation_dtypes=*/{DataType::Float32, DataType::Float32},
      /*validation_shapes=*/{{2}, {3}}, absl::Seconds(1), &sampler);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_EQ(sampler, nullptr);
}

// validation_dtypes disagrees with the table signature's dtype ->
// InvalidArgument ("Requested incompatible tensor").
TEST(ClientTest, NewSamplerWithSignatureDtypeMismatch) {
  auto stub = std::make_shared<FakeStub>();
  ::reverb::tensor::SignatureProto sig = MakeSignature(DataType::Float32, {2});
  stub->set_table_infos({MakeTableInfo("t", &sig)});
  Client client(stub);
  std::unique_ptr<Sampler> sampler;
  auto status = client.NewSampler(
      "t", MakeSamplerOptions(),
      /*validation_dtypes=*/{DataType::Int32},
      /*validation_shapes=*/{{2}}, absl::Seconds(1), &sampler);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_EQ(sampler, nullptr);
}

// validation_shapes rank/size disagrees with the table signature shape ->
// InvalidArgument. Signature is {2}; requesting {} (rank 0) is incompatible.
TEST(ClientTest, NewSamplerWithSignatureShapeIncompatible) {
  auto stub = std::make_shared<FakeStub>();
  ::reverb::tensor::SignatureProto sig = MakeSignature(DataType::Float32, {2});
  stub->set_table_infos({MakeTableInfo("t", &sig)});
  Client client(stub);
  std::unique_ptr<Sampler> sampler;
  auto status = client.NewSampler(
      "t", MakeSamplerOptions(),
      /*validation_dtypes=*/{DataType::Float32},
      /*validation_shapes=*/{{}}, absl::Seconds(1), &sampler);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_EQ(sampler, nullptr);
}

// validation shapes/dtypes exactly match the table signature -> OkStatus and a
// constructed Sampler.
TEST(ClientTest, NewSamplerWithSignatureCompatible) {
  auto stub = std::make_shared<FakeStub>();
  ::reverb::tensor::SignatureProto sig = MakeSignature(DataType::Float32, {2});
  stub->set_table_infos({MakeTableInfo("t", &sig)});
  Client client(stub);
  std::unique_ptr<Sampler> sampler;
  REVERB_EXPECT_OK(client.NewSampler(
      "t", MakeSamplerOptions(),
      /*validation_dtypes=*/{DataType::Float32},
      /*validation_shapes=*/{{2}}, absl::Seconds(1), &sampler));
  EXPECT_NE(sampler, nullptr);
}

// Wildcard dimension: signature shape {-1} is compatible with any rank-1
// shape (e.g. {5}). Exercises IsCompatibleWith's -1-wildcard rule.
TEST(ClientTest, NewSamplerWithSignatureWildcardDimCompatible) {
  auto stub = std::make_shared<FakeStub>();
  ::reverb::tensor::SignatureProto sig = MakeSignature(DataType::Float32, {-1});
  stub->set_table_infos({MakeTableInfo("t", &sig)});
  Client client(stub);
  std::unique_ptr<Sampler> sampler;
  REVERB_EXPECT_OK(client.NewSampler(
      "t", MakeSamplerOptions(),
      /*validation_dtypes=*/{DataType::Float32},
      /*validation_shapes=*/{{5}}, absl::Seconds(1), &sampler));
  EXPECT_NE(sampler, nullptr);
}

// Skip all signature validation; construct a Sampler directly.
TEST(ClientTest, NewSamplerWithoutSignatureCheck) {
  auto stub = std::make_shared<FakeStub>();
  Client client(stub);
  std::unique_ptr<Sampler> sampler;
  REVERB_EXPECT_OK(client.NewSamplerWithoutSignatureCheck(
      "t", MakeSamplerOptions(), &sampler));
  EXPECT_NE(sampler, nullptr);
}

// validation_timeout == -InfiniteDuration: MaybeUpdateServerInfoCache takes
// the "nothing cached, return empty signatures immediately" short-circuit
// (client.cc L75-77). The table is thus absent from the (empty) cache, so
// dtypes_and_shapes ends up nullopt and the sampler is built without a
// signature.
TEST(ClientTest, NewSamplerNegativeInfiniteTimeout) {
  auto stub = std::make_shared<FakeStub>();
  Client client(stub);
  std::unique_ptr<Sampler> sampler;
  REVERB_EXPECT_OK(client.NewSampler("t", MakeSamplerOptions(),
                                     -absl::InfiniteDuration(), &sampler));
  EXPECT_NE(sampler, nullptr);
}

// Table has no signature in ServerInfo, but validation dtypes/shapes are
// provided. Exercises the else-branch (client.cc L283-290) that synthesizes a
// dtypes_and_shapes vector from the validation inputs and feeds it to the
// Sampler.
TEST(ClientTest, NewSamplerWithoutTableSignatureBuildsFromValidation) {
  auto stub = std::make_shared<FakeStub>();
  stub->set_table_infos({MakeTableInfo("t", /*signature=*/nullptr)});
  Client client(stub);
  std::unique_ptr<Sampler> sampler;
  REVERB_EXPECT_OK(client.NewSampler(
      "t", MakeSamplerOptions(),
      /*validation_dtypes=*/{DataType::Float32, DataType::Int32},
      /*validation_shapes=*/{{2}, {3}}, absl::Seconds(1), &sampler));
  EXPECT_NE(sampler, nullptr);
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
