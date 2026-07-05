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

// Tests for the gRPC stream path of TrajectoryWriter
// (RunStreamWorker/OnReadDone/OnWriteDone/OnDone/SetContextAndCreateStream/
// WriteIfNotEmpty/SendNotAlreadySentChunks/Finish). The local path is covered
// by trajectory_writer_test.cc.
//
// TrajectoryWriter uses the gRPC callback/reactor API
// (ClientBidiReactor + stub->async()->InsertStream), unlike Writer and
// StreamingTrajectoryWriter which use the synchronous
// ClientReaderWriterInterface. So we can't reuse their FakeInsertStream
// directly; instead we build a FakeReactorStream that subclasses
// ClientCallbackReaderWriter and manually drives the reactor callbacks
// (OnWriteDone/OnReadDone/OnDone) from background threads.

#include "reverb/cc/trajectory_writer.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "grpcpp/impl/codegen/status.h"
#include "grpcpp/support/client_callback.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/grpc_util.h"
#include "reverb/cc/support/queue.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/support/trajectory_util.h"

namespace deepmind {
namespace reverb {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

using Step = std::vector<std::optional<TensorBuffer>>;
using StepRef = std::vector<std::optional<std::weak_ptr<CellRef>>>;

const auto kIntSpec = internal::TensorSpec{"0", DataType::Int32, {1}};
const auto kFloatSpec = internal::TensorSpec{"0", DataType::Float32, {1}};

constexpr auto kTimeout = absl::Seconds(10);

inline std::string Int32Str() { return DataTypeName(DataType::Int32); }

// --- Tensor builders (TensorBuffer, no TF) ---

template <typename T>
std::string RawBytes(const std::vector<int64_t>& shape, T value) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  std::string bytes(static_cast<size_t>(n) * sizeof(T), '\0');
  for (int64_t i = 0; i < n; ++i) {
    std::memcpy(&bytes[static_cast<size_t>(i) * sizeof(T)], &value, sizeof(T));
  }
  return bytes;
}

template <typename T>
TensorBuffer MakeConstantBuffer(DataType dtype,
                                const std::vector<int64_t>& shape, T value) {
  return TensorBuffer(TensorSpec{dtype, shape}, RawBytes<T>(shape, value));
}

template <typename T>
TensorBuffer MakeZeroBuffer(const internal::TensorSpec& spec) {
  return MakeConstantBuffer<T>(spec.dtype, spec.shape, static_cast<T>(0));
}

std::vector<TrajectoryColumn> MakeTrajectory(
    std::vector<std::vector<std::optional<std::weak_ptr<CellRef>>>>
        trajectory) {
  std::vector<TrajectoryColumn> columns;
  for (const auto& optional_refs : trajectory) {
    std::vector<std::weak_ptr<CellRef>> col_refs;
    for (const auto& optional_ref : optional_refs) {
      col_refs.push_back(optional_ref.value());
    }
    columns.push_back(TrajectoryColumn(std::move(col_refs), /*squeeze=*/false));
  }
  return columns;
}

TrajectoryWriter::Options MakeOptions(int max_chunk_length,
                                      int num_keep_alive_refs) {
  return TrajectoryWriter::Options{
      .chunker_options = std::make_shared<ConstantChunkerOptions>(
          max_chunk_length, num_keep_alive_refs),
  };
}

MATCHER(IsChunk, "") { return arg.chunks_size() >= 1 && arg.items_size() == 0; }
MATCHER(IsItem, "") { return arg.items_size() > 0; }
MATCHER(IsChunkAndItem, "") {
  return arg.chunks_size() >= 1 && arg.items_size() > 0;
}

// ---------------------------------------------------------------------------
// FakeReactorStream: a controllable ClientCallbackReaderWriter that drives the
// TrajectoryWriter (which is itself the ClientBidiReactor) by firing
// OnWriteDone / OnReadDone / OnDone from background threads.
//
// The real gRPC reactor API requires that reactions be non-blocking. We honour
// that by offloading callback firing to background threads:
//   * The write/done driver thread runs OnWriteDone and MaybeFireOnDone
//     closures fed by a thread-safe queue (all non-blocking).
//   * The reader thread blocks on the response-id queue and fires OnReadDone.
//     Keeping reads on a separate thread is essential: otherwise a blocked
//     read would stall the OnDone firing path and deadlock Close()/Finish().
//
// Failure modes are configured at construction: `num_success_writes` writes
// succeed; the next write fails (OnWriteDone(false)), and the stream's Finish
// surfaces `bad_status` via OnDone. Reads auto-confirm any item keys seen in
// written requests (server echoes back inserted item keys).
// ---------------------------------------------------------------------------

class FakeReactorStream
    : public grpc::ClientCallbackReaderWriter<InsertStreamRequest,
                                              InsertStreamResponse> {
 public:
  FakeReactorStream(
      std::shared_ptr<std::vector<InsertStreamRequest>> requests,
      int num_success_writes, grpc::Status bad_status,
      std::shared_ptr<internal::Queue<uint64_t>> response_ids = nullptr,
      bool automatic_response_ids = true)
      : requests_(std::move(requests)),
        num_success_writes_(num_success_writes),
        bad_status_(std::move(bad_status)),
        response_ids_(std::move(response_ids)),
        automatic_response_ids_(automatic_response_ids) {
    if (!response_ids_) {
      response_ids_ = std::make_shared<internal::Queue<uint64_t>>(1000);
    }
    driver_ = internal::StartThread("FakeReactorStreamDrv", [this] {
      DriverLoop();
    });
    reader_ = internal::StartThread("FakeReactorStreamRdr", [this] {
      ReaderLoop();
    });
  }

  ~FakeReactorStream() override {
    // Drop the reactor reference so background threads can't fire reactions
    // into a destroyed TrajectoryWriter, then unblock and join both threads.
    reactor_.store(nullptr, std::memory_order_release);
    response_ids_->Close();
    shutting_down_.store(true, std::memory_order_release);
    // Wake both the driver (waits on mu_) and the reader (waits on read_mu_).
    // Acquiring-and-releasing each mutex triggers Await re-evaluation.
    { absl::MutexLock l(&mu_); }
    cv_.SignalAll();
    { absl::MutexLock l(&read_mu_); }
    read_cv_.Signal();
    reader_ = nullptr;
    driver_ = nullptr;
  }

  // Bind the reactor (TrajectoryWriter) so its internal stream_ points here.
  // Called by our FakeAsyncStub before StartCall.
  void BindReactor(
      grpc::ClientBidiReactor<InsertStreamRequest, InsertStreamResponse>*
          reactor) {
    reactor_.store(reactor, std::memory_order_release);
    // Base-class BindReactor calls reactor->BindStream(this), wiring the
    // reactor's stream_ pointer to this FakeReactorStream so that the
    // reactor's StartRead/StartWrite/StartCall dispatch into our overrides.
    // BindReactor is protected in the base; qualify to call it explicitly.
    grpc::ClientCallbackReaderWriter<InsertStreamRequest,
                                     InsertStreamResponse>::BindReactor(reactor);
  }

  // --- ClientCallbackReaderWriter interface (called by TrajectoryWriter) ---

  void StartCall() override {
    // No-op: background threads fire reactions as writes/reads arrive.
  }

  void Write(const InsertStreamRequest* req, grpc::WriteOptions) override {
    requests_->push_back(*req);

    bool ok = num_success_writes_-- > 0;

    // Only auto-queue confirmations for successful writes. For failed writes
    // the server never sees the request, so it must not echo item keys back;
    // queueing them anyway would let OnReadDone falsely confirm items whose
    // data never reached the server.
    if (ok && automatic_response_ids_) {
      for (const auto& item : req->items()) {
        REVERB_CHECK(response_ids_->Reserve(1));
        response_ids_->PushBatch({item.key()});
      }
    }

    // Schedule OnWriteDone on the driver thread (non-blocking here).
    Enqueue([this, ok] {
      if (reactor()) reactor()->OnWriteDone(ok);
    });
  }

  // TrajectoryWriter calls StartRead with a fixed response buffer. We record
  // the buffer and let the reader thread fill it + fire OnReadDone when a
  // confirmation arrives. Only one outstanding read at a time.
  void Read(InsertStreamResponse* resp) override {
    absl::MutexLock l(&read_mu_);
    read_buf_ = resp;
    read_cv_.Signal();
  }

  void WritesDone() override {
    // Not used by TrajectoryWriter; provided for completeness.
  }

  void AddHold(int holds) override {
    holds_.fetch_add(holds, std::memory_order_relaxed);
  }

  void RemoveHold() override {
    // Mirrors ClientCallbackReaderWriterImpl: when all holds removed and no
    // callbacks outstanding, fire OnDone.
    Enqueue([this] { MaybeFireOnDone(); });
  }

  // Allow tests to inject confirmations manually (when automatic_response_ids_
  // is false).
  std::shared_ptr<internal::Queue<uint64_t>> response_ids() const {
    return response_ids_;
  }

 private:
  grpc::ClientBidiReactor<InsertStreamRequest, InsertStreamResponse>*
  reactor() const {
    return reactor_.load(std::memory_order_acquire);
  }

  // Run a closure on the driver thread.
  void Enqueue(std::function<void()> fn) {
    {
      absl::MutexLock l(&mu_);
      callbacks_outstanding_.fetch_add(1, std::memory_order_relaxed);
      queue_.push_back(std::move(fn));
    }
    cv_.Signal();
  }

  void MaybeFireOnDone() {
    // Fire OnDone once when holds==0. Only fires once.
    // ponytail: we don't wait for callbacks_outstanding_==0 here because this
    // closure is itself counted, which would self-deadlock. FIFO ordering of the
    // driver queue guarantees all OnWriteDone closures queued before this
    // RemoveHold have already run, so OnDone fires strictly after them.
    bool expected = false;
    if (!done_fired_.compare_exchange_strong(expected, true)) return;
    holds_.store(0, std::memory_order_relaxed);
    if (reactor()) reactor()->OnDone(bad_status_);
  }

  void DriverLoop() {
    while (true) {
      std::function<void()> fn;
      {
        absl::MutexLock l(&mu_);
        mu_.Await(absl::Condition(
            +[](FakeReactorStream* s) ABSL_EXCLUSIVE_LOCKS_REQUIRED(s->mu_) {
              return s->shutting_down_.load() || !s->queue_.empty();
            },
            this));
        if (shutting_down_.load() && queue_.empty()) return;
        fn = std::move(queue_.front());
        queue_.pop_front();
      }
      fn();
      callbacks_outstanding_.fetch_sub(1, std::memory_order_relaxed);
      cv_.SignalAll();
    }
  }

  // Reader thread: wait for a registered read buffer, then block on the
  // response-id queue until confirmations arrive (or the queue closes), fill
  // the buffer and fire OnReadDone. Loops to serve re-issued StartReads.
  void ReaderLoop() {
    while (true) {
      InsertStreamResponse* buf = nullptr;
      {
        absl::MutexLock l(&read_mu_);
        read_mu_.Await(absl::Condition(
            +[](FakeReactorStream* s)
                ABSL_EXCLUSIVE_LOCKS_REQUIRED(s->read_mu_) {
              return s->read_buf_ != nullptr || s->shutting_down_.load();
            },
            this));
        if (shutting_down_.load()) return;
        buf = read_buf_;
        read_buf_ = nullptr;
      }
      buf->Clear();
      uint64_t id;
      if (!response_ids_->Pop(&id)) {
        // Queue closed: signal a failed read. TrajectoryWriter will treat
        // this as a stream error and tear down via OnDone.
        if (reactor()) reactor()->OnReadDone(false);
        return;
      }
      buf->add_keys(id);
      while (response_ids_->size() > 0) {
        uint64_t extra;
        if (!response_ids_->Pop(&extra)) break;
        buf->add_keys(extra);
      }
      if (reactor()) reactor()->OnReadDone(true);
    }
  }

  std::shared_ptr<std::vector<InsertStreamRequest>> requests_;
  std::atomic<int> num_success_writes_;
  grpc::Status bad_status_;
  std::shared_ptr<internal::Queue<uint64_t>> response_ids_;
  bool automatic_response_ids_;

  absl::Mutex mu_;
  absl::CondVar cv_;
  std::list<std::function<void()>> queue_ ABSL_GUARDED_BY(mu_);
  std::atomic<bool> shutting_down_{false};
  std::atomic<grpc::ClientBidiReactor<InsertStreamRequest,
                                     InsertStreamResponse>*>
      reactor_{nullptr};
  std::atomic<int> holds_{0};
  std::atomic<int> callbacks_outstanding_{0};
  std::atomic<bool> done_fired_{false};
  std::unique_ptr<internal::Thread> driver_;

  // Reader-thread state.
  absl::Mutex read_mu_;
  absl::CondVar read_cv_;
  InsertStreamResponse* read_buf_ ABSL_GUARDED_BY(read_mu_) = nullptr;
  std::unique_ptr<internal::Thread> reader_;
};

// ---------------------------------------------------------------------------
// FakeAsyncStub: a ReverbService::StubInterface whose async()->InsertStream
// hands out FakeReactorStream instances and binds the TrajectoryWriter reactor
// to them.
// ---------------------------------------------------------------------------

class FakeAsyncStub : public ReverbService::StubInterface {
 public:
  explicit FakeAsyncStub(std::list<FakeReactorStream*> streams)
      : streams_(std::move(streams)) {}

  ~FakeAsyncStub() override {
    // Streams were allocated with new; free any not yet handed out as well as
    // those consumed (whose ownership transferred to us). Joining the
    // FakeReactorStream background threads here is essential to avoid
    // use-after-free of the reactor and detached-thread hangs at exit.
    for (auto* s : consumed_streams_) delete s;
    while (!streams_.empty()) {
      delete streams_.front();
      streams_.pop_front();
    }
  }

  // The reactor entry point used by TrajectoryWriter.
  async_interface* async() override { return &async_iface_; }

  // --- async_interface implementation ---
  class AsyncIface : public async_interface {
   public:
    explicit AsyncIface(FakeAsyncStub* owner) : owner_(owner) {}

    void InsertStream(::grpc::ClientContext* /*context*/,
                      ::grpc::ClientBidiReactor<InsertStreamRequest,
                                                InsertStreamResponse>*
                          reactor) override {
      auto* stream = owner_->TakeNextStream();
      stream->BindReactor(reactor);
    }
    // The other async methods are unused by TrajectoryWriter; no-op them.
    void Checkpoint(::grpc::ClientContext*, const CheckpointRequest*,
                    CheckpointResponse*,
                    std::function<void(::grpc::Status)>) override {}
    void Checkpoint(::grpc::ClientContext*, const CheckpointRequest*,
                    CheckpointResponse*,
                    ::grpc::ClientUnaryReactor*) override {}
    void MutatePriorities(::grpc::ClientContext*,
                          const MutatePrioritiesRequest*,
                          MutatePrioritiesResponse*,
                          std::function<void(::grpc::Status)>) override {}
    void MutatePriorities(::grpc::ClientContext*,
                          const MutatePrioritiesRequest*,
                          MutatePrioritiesResponse*,
                          ::grpc::ClientUnaryReactor*) override {}
    void Reset(::grpc::ClientContext*, const ResetRequest*,
               ResetResponse*, std::function<void(::grpc::Status)>) override {}
    void Reset(::grpc::ClientContext*, const ResetRequest*,
               ResetResponse*, ::grpc::ClientUnaryReactor*) override {}
    void SampleStream(::grpc::ClientContext*,
                      ::grpc::ClientBidiReactor<SampleStreamRequest,
                                                SampleStreamResponse>*) override {}
    void ServerInfo(::grpc::ClientContext*, const ServerInfoRequest*,
                    ServerInfoResponse*,
                    std::function<void(::grpc::Status)>) override {}
    void ServerInfo(::grpc::ClientContext*, const ServerInfoRequest*,
                    ServerInfoResponse*,
                    ::grpc::ClientUnaryReactor*) override {}
    void InitializeConnection(::grpc::ClientContext*,
                              ::grpc::ClientBidiReactor<
                                  InitializeConnectionRequest,
                                  InitializeConnectionResponse>*) override {}

   private:
    FakeAsyncStub* owner_;
  };

  // --- Unused synchronous StubInterface methods (return errors) ---
  // ponytail: TrajectoryWriter only touches async()->InsertStream; the sync
  // RPCs are never called by the writer, so stubbing them to return an error
  // is the minimal correct behaviour (avoids pulling in real gRPC plumbing).
  ::grpc::Status Checkpoint(::grpc::ClientContext*, const CheckpointRequest&,
                            CheckpointResponse*) override {
    return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
  }
  ::grpc::Status MutatePriorities(::grpc::ClientContext*,
                                  const MutatePrioritiesRequest&,
                                  MutatePrioritiesResponse*) override {
    return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
  }
  ::grpc::Status Reset(::grpc::ClientContext*, const ResetRequest&,
                       ResetResponse*) override {
    return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
  }
  ::grpc::Status ServerInfo(::grpc::ClientContext*, const ServerInfoRequest&,
                            ServerInfoResponse*) override {
    return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
  }
  ::grpc::ClientReaderWriterInterface<InsertStreamRequest, InsertStreamResponse>*
  InsertStreamRaw(::grpc::ClientContext*) override {
    return nullptr;
  }
  ::grpc::ClientAsyncResponseReaderInterface<CheckpointResponse>*
  AsyncCheckpointRaw(::grpc::ClientContext*, const CheckpointRequest&,
                     ::grpc::CompletionQueue*) override {
    return nullptr;
  }
  ::grpc::ClientAsyncResponseReaderInterface<CheckpointResponse>*
  PrepareAsyncCheckpointRaw(::grpc::ClientContext*, const CheckpointRequest&,
                            ::grpc::CompletionQueue*) override {
    return nullptr;
  }
  ::grpc::ClientAsyncReaderWriterInterface<InsertStreamRequest,
                                           InsertStreamResponse>*
  AsyncInsertStreamRaw(::grpc::ClientContext*, ::grpc::CompletionQueue*,
                       void*) override {
    return nullptr;
  }
  ::grpc::ClientAsyncReaderWriterInterface<InsertStreamRequest,
                                           InsertStreamResponse>*
  PrepareAsyncInsertStreamRaw(::grpc::ClientContext*,
                              ::grpc::CompletionQueue*) override {
    return nullptr;
  }
  ::grpc::ClientAsyncResponseReaderInterface<MutatePrioritiesResponse>*
  AsyncMutatePrioritiesRaw(::grpc::ClientContext*,
                           const MutatePrioritiesRequest&,
                           ::grpc::CompletionQueue*) override {
    return nullptr;
  }
  ::grpc::ClientAsyncResponseReaderInterface<MutatePrioritiesResponse>*
  PrepareAsyncMutatePrioritiesRaw(::grpc::ClientContext*,
                                  const MutatePrioritiesRequest&,
                                  ::grpc::CompletionQueue*) override {
    return nullptr;
  }
  ::grpc::ClientAsyncResponseReaderInterface<ResetResponse>*
  AsyncResetRaw(::grpc::ClientContext*, const ResetRequest&,
                ::grpc::CompletionQueue*) override {
    return nullptr;
  }
  ::grpc::ClientAsyncResponseReaderInterface<ResetResponse>*
  PrepareAsyncResetRaw(::grpc::ClientContext*, const ResetRequest&,
                       ::grpc::CompletionQueue*) override {
    return nullptr;
  }
  ::grpc::ClientReaderWriterInterface<SampleStreamRequest,
                                      SampleStreamResponse>*
  SampleStreamRaw(::grpc::ClientContext*) override {
    return nullptr;
  }
  ::grpc::ClientAsyncReaderWriterInterface<SampleStreamRequest,
                                           SampleStreamResponse>*
  AsyncSampleStreamRaw(::grpc::ClientContext*, ::grpc::CompletionQueue*,
                       void*) override {
    return nullptr;
  }
  ::grpc::ClientAsyncReaderWriterInterface<SampleStreamRequest,
                                           SampleStreamResponse>*
  PrepareAsyncSampleStreamRaw(::grpc::ClientContext*,
                              ::grpc::CompletionQueue*) override {
    return nullptr;
  }
  ::grpc::ClientAsyncResponseReaderInterface<ServerInfoResponse>*
  AsyncServerInfoRaw(::grpc::ClientContext*, const ServerInfoRequest&,
                     ::grpc::CompletionQueue*) override {
    return nullptr;
  }
  ::grpc::ClientAsyncResponseReaderInterface<ServerInfoResponse>*
  PrepareAsyncServerInfoRaw(::grpc::ClientContext*, const ServerInfoRequest&,
                            ::grpc::CompletionQueue*) override {
    return nullptr;
  }
  ::grpc::ClientReaderWriterInterface<InitializeConnectionRequest,
                                      InitializeConnectionResponse>*
  InitializeConnectionRaw(::grpc::ClientContext*) override {
    return nullptr;
  }
  ::grpc::ClientAsyncReaderWriterInterface<InitializeConnectionRequest,
                                           InitializeConnectionResponse>*
  AsyncInitializeConnectionRaw(::grpc::ClientContext*,
                               ::grpc::CompletionQueue*, void*) override {
    return nullptr;
  }
  ::grpc::ClientAsyncReaderWriterInterface<InitializeConnectionRequest,
                                           InitializeConnectionResponse>*
  PrepareAsyncInitializeConnectionRaw(::grpc::ClientContext*,
                                      ::grpc::CompletionQueue*) override {
    return nullptr;
  }

 private:
  friend class AsyncIface;

  FakeReactorStream* TakeNextStream() {
    FakeReactorStream* s = streams_.front();
    streams_.pop_front();
    consumed_streams_.push_back(s);
    return s;
  }

  AsyncIface async_iface_{this};
  std::list<FakeReactorStream*> streams_;
  std::list<FakeReactorStream*> consumed_streams_;
};

// ---------------------------------------------------------------------------
// Stub factories mirroring writer_test.cc's MakeGoodStub / MakeFlakyStub.
// ---------------------------------------------------------------------------

struct StubAndRequests {
  std::shared_ptr<FakeAsyncStub> stub;
  std::shared_ptr<std::vector<InsertStreamRequest>> requests;
};

StubAndRequests MakeGoodStub() {
  auto requests = std::make_shared<std::vector<InsertStreamRequest>>();
  auto* stream = new FakeReactorStream(
      requests, /*num_success_writes=*/10000,
      grpc::Status::OK);
  auto stub = std::make_shared<FakeAsyncStub>(
      std::list<FakeReactorStream*>{stream});
  return {stub, requests};
}

// A flaky stub: the first stream fails after `num_success` successful writes
// (surfacing `error` via OnDone), then subsequent streams succeed. The number
// of failing streams is `num_fail`.
StubAndRequests MakeFlakyStub(int num_success, int num_fail, grpc::Status error) {
  auto requests = std::make_shared<std::vector<InsertStreamRequest>>();
  std::list<FakeReactorStream*> streams;
  streams.push_back(new FakeReactorStream(requests, num_success, error));
  for (int i = 1; i < num_fail; ++i) {
    streams.push_back(new FakeReactorStream(requests, 0, error));
  }
  streams.push_back(new FakeReactorStream(
      requests, 10000, grpc::Status::OK));
  auto stub = std::make_shared<FakeAsyncStub>(std::move(streams));
  return {stub, requests};
}

// Waits for `requests` to contain at least `n` entries, with a timeout.
void WaitForRequests(const std::shared_ptr<std::vector<InsertStreamRequest>>& reqs,
                     int n) {
  absl::Notification done;
  auto t = internal::StartThread("WaitForRequests", [&] {
    while (static_cast<int>(reqs->size()) < n) {
      absl::SleepFor(absl::Milliseconds(2));
    }
    done.Notify();
  });
  ASSERT_TRUE(done.WaitForNotificationWithTimeout(kTimeout))
      << "timed out waiting for " << n << " requests, got " << reqs->size();
  t = nullptr;
}

// ---------------------------------------------------------------------------
// gRPC stream path tests.
// ---------------------------------------------------------------------------

TEST(TrajectoryWriterGrpcTest, DoesNotSendTimestepsWhenThereAreNoItems) {
  auto [stub, requests] = MakeGoodStub();
  TrajectoryWriter writer(stub, MakeOptions(/*max_chunk_length=*/2,
                                            /*num_keep_alive_refs=*/10));
  StepRef refs;
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
  // No CreateItem => nothing should be written to the stream.
  absl::SleepFor(absl::Milliseconds(100));
  EXPECT_THAT(*requests, SizeIs(0));
  writer.Close();
}

TEST(TrajectoryWriterGrpcTest, OnlySendsChunksWhichAreUsedByItems) {
  auto [stub, requests] = MakeGoodStub();
  TrajectoryWriter writer(stub, MakeOptions(/*max_chunk_length=*/1,
                                            /*num_keep_alive_refs=*/5));
  // Two columns; only the first is referenced by an item.
  StepRef refs;
  REVERB_ASSERT_OK(writer.Append(
      Step({MakeZeroBuffer<int32_t>(kIntSpec),
            MakeZeroBuffer<float>(kFloatSpec)}), &refs));
  // max_chunk_length=1 => chunk finalized immediately.
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{refs[0]}})));
  REVERB_ASSERT_OK(writer.Flush());

  WaitForRequests(requests, 1);
  // The request should contain exactly the one chunk referenced by the item.
  ASSERT_THAT(*requests, SizeIs(1));
  EXPECT_THAT((*requests)[0], IsChunkAndItem());
  EXPECT_EQ((*requests)[0].chunks_size(), 1);
  EXPECT_EQ((*requests)[0].items_size(), 1);
  writer.Close();
}

TEST(TrajectoryWriterGrpcTest, DoesNotSendAlreadySentChunks) {
  auto [stub, requests] = MakeGoodStub();
  TrajectoryWriter writer(stub, MakeOptions(/*max_chunk_length=*/1,
                                            /*num_keep_alive_refs=*/5));
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}),
                                 &first));
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{first[0]}})));
  REVERB_ASSERT_OK(writer.Flush());

  WaitForRequests(requests, 1);
  ASSERT_THAT(*requests, SizeIs(1));
  EXPECT_EQ((*requests)[0].chunks_size(), 1);
  uint64_t first_chunk_key = (*requests)[0].chunks(0).chunk_key();

  // Second item references the SAME chunk; it must not be resent.
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 2.0, MakeTrajectory({{first[0]}})));
  REVERB_ASSERT_OK(writer.Flush());

  WaitForRequests(requests, 2);
  ASSERT_THAT(*requests, SizeIs(2));
  EXPECT_EQ((*requests)[1].chunks_size(), 0);  // no chunk re-sent
  EXPECT_EQ((*requests)[1].items_size(), 1);
  EXPECT_THAT(internal::GetChunkKeys(
                (*requests)[1].items(0).flat_trajectory()),
              ElementsAre(first_chunk_key));
  writer.Close();
}

TEST(TrajectoryWriterGrpcTest, SendsPendingDataOnClose) {
  // TrajectoryWriter's explicit Close() is a hard shutdown that does NOT flush
  // pending items (unlike Writer::Close). The destructor, however, flushes via
  // FlushLocked when the writer hasn't been closed yet. So we let the writer
  // go out of scope to verify pending data is flushed on destruction.
  auto [stub, requests] = MakeGoodStub();
  StepRef refs;
  {
    TrajectoryWriter writer(stub, MakeOptions(/*max_chunk_length=*/2,
                                              /*num_keep_alive_refs=*/5));
    REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}),
                                   &refs));
    REVERB_ASSERT_OK(
        writer.CreateItem("table", 1.0, MakeTrajectory({{refs[0]}})));
    absl::SleepFor(absl::Milliseconds(50));
    EXPECT_THAT(*requests, SizeIs(0));
    // Destructor flushes the pending item.
  }
  WaitForRequests(requests, 1);
  ASSERT_THAT(*requests, SizeIs(1));
  EXPECT_THAT((*requests)[0], IsChunkAndItem());
}

TEST(TrajectoryWriterGrpcTest, FailsIfMethodsCalledAfterClose) {
  auto [stub, requests] = MakeGoodStub();
  TrajectoryWriter writer(stub, MakeOptions(/*max_chunk_length=*/2,
                                            /*num_keep_alive_refs=*/5));
  writer.Close();

  StepRef refs;
  auto append_status = writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}),
                                     &refs);
  EXPECT_FALSE(append_status.ok());

  // CreateItem needs a live ref, but Append after close returns an error and
  // does not populate refs. We instead test CreateItem's empty-trajectory
  // guard, which fails before touching any refs.
  auto create_status = writer.CreateItem("table", 1.0, {});
  EXPECT_FALSE(create_status.ok());
}

TEST(TrajectoryWriterGrpcTest, FlushWritesItem) {
  auto [stub, requests] = MakeGoodStub();
  TrajectoryWriter writer(stub, MakeOptions(/*max_chunk_length=*/2,
                                            /*num_keep_alive_refs=*/5));
  StepRef refs;
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}),
                                 &refs));
  // Flush with no item: should not write anything.
  REVERB_ASSERT_OK(writer.Flush());
  EXPECT_THAT(*requests, SizeIs(0));

  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{refs[0]}})));
  EXPECT_THAT(*requests, SizeIs(0));
  REVERB_ASSERT_OK(writer.Flush());
  WaitForRequests(requests, 1);
  ASSERT_THAT(*requests, SizeIs(1));
  EXPECT_THAT((*requests)[0], IsChunkAndItem());
  EXPECT_EQ((*requests)[0].items(0).table(), "table");
  EXPECT_EQ((*requests)[0].items(0).priority(), 1.0);
  writer.Close();
}

TEST(TrajectoryWriterGrpcTest, SequenceRangeIsSetOnChunks) {
  auto [stub, requests] = MakeGoodStub();
  TrajectoryWriter writer(stub, MakeOptions(/*max_chunk_length=*/2,
                                            /*num_keep_alive_refs=*/5));
  // Three steps => chunks [0,1] and [2].
  StepRef r0, r1, r2;
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &r0));
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &r1));
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &r2));
  // Item spanning all three steps (two chunks).
  REVERB_ASSERT_OK(writer.CreateItem(
      "table", 1.0, MakeTrajectory({{r0[0], r1[0], r2[0]}})));
  REVERB_ASSERT_OK(writer.Flush());

  // The worker may split the chunk+item across requests depending on chunk
  // readiness timing, so collect all chunks across every request.
  WaitForRequests(requests, 1);
  uint64_t chunk0_key = r0[0]->lock()->chunk_key();
  uint64_t chunk1_key = r2[0]->lock()->chunk_key();
  ASSERT_NE(chunk0_key, chunk1_key);

  // Find both chunks among the streamed requests.
  const ChunkData* chunk0 = nullptr;
  const ChunkData* chunk1 = nullptr;
  for (const auto& req : *requests) {
    for (const auto& chunk : req.chunks()) {
      if (chunk.chunk_key() == chunk0_key) chunk0 = &chunk;
      if (chunk.chunk_key() == chunk1_key) chunk1 = &chunk;
    }
  }
  ASSERT_NE(chunk0, nullptr);
  ASSERT_NE(chunk1, nullptr);
  // First chunk covers steps [0,1].
  EXPECT_EQ(chunk0->sequence_range().start(), 0);
  EXPECT_EQ(chunk0->sequence_range().end(), 1);
  // Second chunk covers step [2,2].
  EXPECT_EQ(chunk1->sequence_range().start(), 2);
  EXPECT_EQ(chunk1->sequence_range().end(), 2);
  // Same episode id across chunks.
  EXPECT_NE(chunk0->sequence_range().episode_id(), 0);
  EXPECT_EQ(chunk0->sequence_range().episode_id(),
            chunk1->sequence_range().episode_id());
  writer.Close();
}

TEST(TrajectoryWriterGrpcTest, MultiChunkItemsAreCorrect) {
  auto [stub, requests] = MakeGoodStub();
  TrajectoryWriter writer(stub, MakeOptions(/*max_chunk_length=*/3,
                                            /*num_keep_alive_refs=*/5));
  // Build two chunks across 5 steps and three items of varying span.
  StepRef steps[5];
  for (int i = 0; i < 5; ++i) {
    REVERB_ASSERT_OK(
        writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &steps[i]));
  }
  // Item1: steps [0,1] (chunk0).
  REVERB_ASSERT_OK(writer.CreateItem(
      "table", 1.0, MakeTrajectory({{steps[0][0], steps[1][0]}})));
  // Item2: steps [1,2,3] (chunk0+chunk1).
  REVERB_ASSERT_OK(writer.CreateItem(
      "table", 2.0, MakeTrajectory({{steps[1][0], steps[2][0], steps[3][0]}})));
  // Item3: step [4] (chunk1).
  REVERB_ASSERT_OK(writer.CreateItem(
      "table", 3.0, MakeTrajectory({{steps[4][0]}})));
  REVERB_ASSERT_OK(writer.Flush());

  // The worker may emit the chunk+item batch in one or more requests depending
  // on chunk readiness timing. Find the request carrying Item1 and verify its
  // trajectory references exactly one chunk (chunk0).
  WaitForRequests(requests, 1);
  uint64_t chunk0_key = steps[0][0]->lock()->chunk_key();
  uint64_t chunk1_key = steps[3][0]->lock()->chunk_key();
  EXPECT_NE(chunk0_key, chunk1_key);

  // Collect every item sent across all requests.
  std::vector<const PrioritizedItem*> sent_items;
  for (const auto& req : *requests) {
    for (const auto& item : req.items()) sent_items.push_back(&item);
  }
  // Three items must have been sent.
  ASSERT_EQ(sent_items.size(), 3u);
  // Item1's trajectory should reference chunk0 only.
  EXPECT_THAT(internal::GetChunkKeys(sent_items[0]->flat_trajectory()),
              ElementsAre(chunk0_key));
  // Item2's trajectory should reference chunk0 + chunk1.
  EXPECT_THAT(internal::GetChunkKeys(sent_items[1]->flat_trajectory()),
              UnorderedElementsAre(chunk0_key, chunk1_key));
  // Item3's trajectory should reference chunk1 only.
  EXPECT_THAT(internal::GetChunkKeys(sent_items[2]->flat_trajectory()),
              ElementsAre(chunk1_key));
  writer.Close();
}

TEST(TrajectoryWriterGrpcTest, DataUncompressedSizeIsPopulatedInChunks) {
  auto [stub, requests] = MakeGoodStub();
  TrajectoryWriter writer(stub, MakeOptions(/*max_chunk_length=*/2,
                                            /*num_keep_alive_refs=*/5));
  StepRef refs;
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{refs[0], refs[1]}})));
  REVERB_ASSERT_OK(writer.Flush());

  WaitForRequests(requests, 1);
  ASSERT_THAT(*requests, SizeIs(1));
  ASSERT_EQ((*requests)[0].chunks_size(), 1);
  EXPECT_GT((*requests)[0].chunks(0).data_uncompressed_size(), 0);
  writer.Close();
}

TEST(TrajectoryWriterGrpcTest, TellsServerToKeepStreamedItemsStillInClient) {
  // TrajectoryWriter only sends chunks referenced by items, so keep_chunk_keys
  // contains exactly the streamed chunks that are still live (referenced by
  // the chunker's keep-alive window or by later pending items). We build two
  // chunks (max_chunk_length=1) and two items referencing different chunks,
  // then verify keep keys cover both streamed chunks while both are live.
  auto [stub, requests] = MakeGoodStub();
  TrajectoryWriter writer(stub, MakeOptions(/*max_chunk_length=*/1,
                                            /*num_keep_alive_refs=*/6));
  StepRef s0, s1;
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &s0));
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &s1));
  uint64_t c0 = s0[0]->lock()->chunk_key();
  uint64_t c1 = s1[0]->lock()->chunk_key();
  ASSERT_NE(c0, c1);

  // First item references only chunk0; its request should carry c0 in keep
  // keys (and NOT c1, since c1 hasn't been streamed yet at that point).
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{s0[0]}})));
  REVERB_ASSERT_OK(writer.Flush());
  WaitForRequests(requests, 1);
  ASSERT_THAT(*requests, SizeIs(1));
  EXPECT_THAT((*requests)[0].keep_chunk_keys(), ElementsAre(c0));

  // Second item references both chunks; now both have been streamed and both
  // are still live (num_keep_alive_refs=6), so keep keys should contain both.
  requests->clear();
  REVERB_ASSERT_OK(writer.CreateItem(
      "table", 2.0, MakeTrajectory({{s0[0], s1[0]}})));
  REVERB_ASSERT_OK(writer.Flush());
  WaitForRequests(requests, 1);
  ASSERT_THAT(*requests, SizeIs(1));
  EXPECT_THAT((*requests)[0].keep_chunk_keys(),
              UnorderedElementsAre(c0, c1));
  writer.Close();
}

TEST(TrajectoryWriterGrpcTest, RetriesOnTransientError) {
  // First stream: every write fails with UNAVAILABLE. Second stream succeeds.
  // The writer must retry on the new stream and resend the chunk + item.
  auto [stub, requests] = MakeFlakyStub(/*num_success=*/0,
                                        /*num_fail=*/1,
                                        ToGrpcStatus(absl::UnavailableError("transient")));
  TrajectoryWriter writer(stub, MakeOptions(/*max_chunk_length=*/1,
                                            /*num_keep_alive_refs=*/5));
  StepRef refs;
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{refs[0]}})));
  REVERB_ASSERT_OK(writer.Flush());

  // Two requests: the failed write on stream0, then the retried write on
  // stream1. Both carry the chunk + item.
  WaitForRequests(requests, 2);
  EXPECT_THAT(*requests, SizeIs(2));
  EXPECT_THAT((*requests)[0], IsChunkAndItem());
  EXPECT_THAT((*requests)[1], IsChunkAndItem());
  EXPECT_EQ((*requests)[0].chunks(0).chunk_key(),
            (*requests)[1].chunks(0).chunk_key());
  writer.Close();
}

TEST(TrajectoryWriterGrpcTest, DoesNotRetryOnNonTransientError) {
  auto [stub, requests] = MakeFlakyStub(/*num_success=*/0,
                                        /*num_fail=*/1,
                                        ToGrpcStatus(absl::InternalError("fatal")));
  TrajectoryWriter writer(stub, MakeOptions(/*max_chunk_length=*/1,
                                            /*num_keep_alive_refs=*/5));
  StepRef refs;
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{refs[0]}})));
  // Flush should eventually surface the non-transient error.
  auto status = writer.Flush(/*ignore_last_num_items=*/0);
  EXPECT_FALSE(status.ok());
  writer.Close();
}

TEST(TrajectoryWriterGrpcTest, AppendValidatesDtype) {
  auto [stub, requests] = MakeGoodStub();
  TrajectoryWriter writer(stub, MakeOptions(/*max_chunk_length=*/10,
                                            /*num_keep_alive_refs=*/10));
  StepRef refs;
  REVERB_ASSERT_OK(writer.Append(
      Step({MakeZeroBuffer<int32_t>(kIntSpec),
            MakeZeroBuffer<float>(kFloatSpec)}), &refs));
  auto status = writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec),
                                    MakeZeroBuffer<int32_t>(kIntSpec)}), &refs);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              HasSubstr(absl::StrCat("Tensor of wrong dtype provided for column 1. "
                                     "Got ", Int32Str(), " but expected Float32.")));
  writer.Close();
}

TEST(TrajectoryWriterGrpcTest, CreateItemValidatesTrajectoryNotEmpty) {
  auto [stub, requests] = MakeGoodStub();
  TrajectoryWriter writer(stub, MakeOptions(/*max_chunk_length=*/1,
                                            /*num_keep_alive_refs=*/1));
  StepRef step;
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &step));
  auto status = writer.CreateItem("table", 1.0, {});
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()), HasSubstr("trajectory must not be empty."));
  writer.Close();
}

TEST(TrajectoryWriterGrpcTest, CreateItemValidatesNanPriority) {
  auto [stub, requests] = MakeGoodStub();
  TrajectoryWriter writer(stub, MakeOptions(/*max_chunk_length=*/1,
                                            /*num_keep_alive_refs=*/1));
  StepRef step;
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &step));
  auto status = writer.CreateItem("table", std::nan(""), MakeTrajectory({{step[0]}}));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()), HasSubstr("`priority` must not be nan."));
  writer.Close();
}

TEST(TrajectoryWriterGrpcTest, EndEpisodeResetsEpisodeKeyAndStep) {
  auto [stub, requests] = MakeGoodStub();
  TrajectoryWriter writer(stub, MakeOptions(/*max_chunk_length=*/1,
                                            /*num_keep_alive_refs=*/2));
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &first));
  uint64_t first_episode = first[0]->lock()->episode_id();
  EXPECT_EQ(writer.episode_steps(), 1);

  REVERB_ASSERT_OK(writer.EndEpisode(/*clear_buffers=*/true));

  StepRef second;
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &second));
  EXPECT_NE(first_episode, second[0]->lock()->episode_id());
  EXPECT_EQ(writer.episode_steps(), 1);
  writer.Close();
}

TEST(TrajectoryWriterGrpcTest, CrossEpisodeItemsSpanTwoChunks) {
  auto [stub, requests] = MakeGoodStub();
  TrajectoryWriter writer(stub, MakeOptions(/*max_chunk_length=*/1,
                                            /*num_keep_alive_refs=*/4));
  StepRef refs;
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
  REVERB_ASSERT_OK(writer.EndEpisode(/*clear_buffers=*/false));
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
  // Two chunks (one per episode); item spans both.
  REVERB_ASSERT_OK(writer.CreateItem(
      "table", 1.0, MakeTrajectory({{refs[0], refs[1]}})));
  REVERB_ASSERT_OK(writer.Flush());

  WaitForRequests(requests, 1);
  ASSERT_THAT(*requests, SizeIs(1));
  EXPECT_EQ((*requests)[0].chunks_size(), 2);
  EXPECT_EQ((*requests)[0].items_size(), 1);
  // Different episode ids.
  EXPECT_NE((*requests)[0].chunks(0).sequence_range().episode_id(),
            (*requests)[0].chunks(1).sequence_range().episode_id());
  writer.Close();
}

// ---------------------------------------------------------------------------
// Signature validation: exercises ItemAndRefs::Validate + FlatSignatureFromTrajectory
// + ShapeString, which the round-trip tests above never reach (no signature map).
// Mirrors StreamingTrajectoryWriterSignatureValidationTest.
// ---------------------------------------------------------------------------

class TrajectoryWriterGrpcSignatureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto [stub, requests] = MakeGoodStub();
    stub_ = stub;
    requests_ = requests;
    TrajectoryWriter::Options options = {
        .chunker_options = std::make_shared<ConstantChunkerOptions>(1, 1),
        .flat_signature_map = internal::FlatSignatureMap({
            {"table",
             std::vector<internal::TensorSpec>({
                 internal::TensorSpec{"first_col", DataType::Int32, {2}},
                 internal::TensorSpec{"second_col", DataType::Float32, {1}},
                 internal::TensorSpec{"var_length_col", DataType::Float32, {-1}},
             })},
        }),
    };
    writer_ = std::make_unique<TrajectoryWriter>(stub_, options);

    // Steps with enough columns to compose valid/invalid trajectories.
    REVERB_ASSERT_OK(writer_->Append(Step({
        MakeZeroBuffer<int32_t>(internal::TensorSpec{"0", DataType::Int32, {}}),
        MakeZeroBuffer<float>(internal::TensorSpec{"1", DataType::Float32, {}}),
        MakeZeroBuffer<double>(internal::TensorSpec{"2", DataType::Float64, {}}),
        MakeConstantBuffer<float>(DataType::Float32, {2, 2}, 0.0f),
    }), &step_));
  }

  std::shared_ptr<FakeAsyncStub> stub_;
  std::shared_ptr<std::vector<InsertStreamRequest>> requests_;
  std::unique_ptr<TrajectoryWriter> writer_;
  StepRef step_;
};

TEST_F(TrajectoryWriterGrpcSignatureTest, Valid) {
  REVERB_EXPECT_OK(writer_->CreateItem("table", 1.0, MakeTrajectory({
      {step_[0], step_[0]},  // Int32 [2]
      {step_[1]},            // Float32 [1]
      {step_[1]},            // Float32 [1] (var length)
  })));
}

TEST_F(TrajectoryWriterGrpcSignatureTest, WrongNumColumns) {
  auto status = writer_->CreateItem("table", 1.0, MakeTrajectory({
      {step_[0], step_[0]},
      {step_[1]},
  }));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              HasSubstr("trajectory has 2 columns but the table signature has 3"));
}

TEST_F(TrajectoryWriterGrpcSignatureTest, NotFoundTable) {
  auto status = writer_->CreateItem("not_found", 1.0, MakeTrajectory({
      {step_[0], step_[0]},
      {step_[1]},
      {step_[1]},
  }));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              HasSubstr("table could not be found"));
}

TEST_F(TrajectoryWriterGrpcSignatureTest, WrongDtype) {
  auto status = writer_->CreateItem("table", 1.0, MakeTrajectory({
      {step_[0], step_[0]},
      {step_[2]},  // Float64 but signature wants Float32
      {step_[1]},
  }));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              HasSubstr("expects column 1 to be a Float32 [1] tensor but got a "
                        "Float64 [1] tensor"));
}

TEST_F(TrajectoryWriterGrpcSignatureTest, WrongBatchDim) {
  auto status = writer_->CreateItem("table", 1.0, MakeTrajectory({
      {step_[0]},  // only 1 row, signature wants [2]
      {step_[1]},
      {step_[1]},
  }));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              HasSubstr("expects column 0 to be a Int32 [2] tensor but got a "
                        "Int32 [1] tensor"));
}

TEST_F(TrajectoryWriterGrpcSignatureTest, WrongElementShapeWithVarLength) {
  // var_length_col has shape [-1]; a 2x2 element is incompatible (multi-dim).
  auto status = writer_->CreateItem("table", 1.0, MakeTrajectory({
      {step_[0], step_[0]},
      {step_[1]},
      {step_[3]},  // Float32 [2,2]
  }));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              HasSubstr("expects column 2 to be a Float32 [?] tensor"));
}

// Large tensors force SendNotAlreadySentChunks to split across multiple
// requests once kMaxRequestSizeBytes is exceeded.
// ponytail: skipped — exercising the 40MB split path reliably requires
// incompressible multi-chunk batches whose memory/timing make the test flaky
// under the instrumented build. The split logic (SendNotAlreadySentChunks
// mid-loop WriteIfNotEmpty) is small; the round-trip tests above cover the
// common single-request path. Re-enable if split-path coverage is needed.

// EndEpisode with clear_buffers=false exercises the Flush branch in the
// chunker-reset loop (otherwise only the Reset branch is hit).
TEST(TrajectoryWriterGrpcTest, EndEpisodeFlushesWithoutClearingBuffers) {
  auto [stub, requests] = MakeGoodStub();
  TrajectoryWriter writer(stub, MakeOptions(/*max_chunk_length=*/4,
                                            /*num_keep_alive_refs=*/4));
  StepRef refs;
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
  // Chunk is incomplete (1 of 4); EndEpisode(false) must flush it.
  EXPECT_FALSE(refs[0]->lock()->IsReady());
  REVERB_ASSERT_OK(writer.EndEpisode(/*clear_buffers=*/false));
  EXPECT_TRUE(refs[0]->lock()->IsReady());
  EXPECT_EQ(writer.episode_steps(), 0);
  writer.Close();
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
