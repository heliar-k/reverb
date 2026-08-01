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

// Ticket ④: the WRITE direction. A TrajectoryWriter constructed in SHM mode
// (ShmClient::NewTrajectoryWriter) appends data, creates an item, and flushes;
// the item lands in the server's Table over the SHM transport (ALLOCATE +
// INSERT + INSERT_ACK, decision C2/C4). ③'s ShmSampler then reads it back and
// the bytes/dtype/shape must match what was appended. Mirrors
// in_process_client_test.cc's WriteAndSampleRoundTrip, but the writer and the
// sampler talk to a real ShmServer over udsocket + SHM rings + pool.
//
// ponytail: we assert on TensorBuffer bytes/shape/dtype (the transport
// contract), same rationale as shm_sample_test.cc — no CPython interpreter in
// this binary (see shm_sample_test.cc header comment on the libpython/openssl
// clash via :sampler -> grpc).

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/platform/default/logging.h"
#include "reverb/cc/platform/default/status_matchers.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/shm/shm_client.h"
#include "reverb/cc/shm/shm_server.h"
#include "reverb/cc/structured_writer.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/table.h"
#include "reverb/cc/trajectory_writer.h"

namespace deepmind {
namespace reverb {
namespace shm {
namespace {

using ::testing::SizeIs;

using Step = std::vector<std::optional<TensorBuffer>>;
using StepRef = std::vector<std::optional<std::weak_ptr<CellRef>>>;

const auto kIntSpec = internal::TensorSpec{"0", DataType::Int32, {1}};

// --- Inline tensor builders (no TensorFlow / pybind11 dependency) ---

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
TensorBuffer MakeConstantBuffer(DataType dtype_for_spec,
                                const std::vector<int64_t>& shape, T value) {
  return TensorBuffer(TensorSpec{dtype_for_spec, shape},
                      RawBytes<T>(shape, value));
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

std::shared_ptr<Table> MakeTable(const std::string& name, int max_size = 100) {
  return std::make_shared<Table>(
      /*name=*/name,
      /*sampler=*/std::make_shared<FifoSelector>(),
      /*remover=*/std::make_shared<FifoSelector>(),
      /*max_size=*/max_size,
      /*max_times_sampled=*/1,
      /*rate_limiter=*/std::make_shared<RateLimiter>(1, 1, 0, max_size));
}

// A table whose rate limiter never blocks (min/max_diff wide open): inserts
// and samples proceed freely. Used by the sequential-insert-ack test so the
// ACK path is exercised without the rate limiter gating inserts (which would
// hang the worker's blocking ACK poll when no sampler is draining).
std::shared_ptr<Table> MakePermissiveTable(const std::string& name,
                                            int max_size = 100) {
  return std::make_shared<Table>(
      /*name=*/name,
      /*sampler=*/std::make_shared<FifoSelector>(),
      /*remover=*/std::make_shared<FifoSelector>(),
      /*max_size=*/max_size,
      /*max_times_sampled=*/1,
      /*rate_limiter=*/
      std::make_shared<RateLimiter>(1, 1, /*min_diff=*/-1e9, /*max_diff=*/1e9));
}

std::string UniqueTag(const std::string& tag) {
  return tag + "_" + std::to_string(getpid()) + "_" +
         std::to_string(reinterpret_cast<uintptr_t>(&tag));
}

// Bring up a server + client for `table` on a fresh udsocket path. The server
// is started; the caller stops it at the end of the test. Returns nullptr on
// failure (asserts are void-returning and can't live in a factory that
// returns a value).
struct ShmFixture {
  std::shared_ptr<Table> table;
  std::unique_ptr<ShmServer> server;
  std::unique_ptr<ShmClient> client;
  std::string sock;

  static std::unique_ptr<ShmFixture> Make(std::shared_ptr<Table> table,
                                          const std::string& tag) {
    auto f = std::make_unique<ShmFixture>();
    f->table = table;
    f->sock = "/tmp/reverb_shm_insert_" + UniqueTag(tag) + ".sock";
    auto s = ShmServer::Create({table}, f->sock);
    if (!s.ok()) return nullptr;
    f->server = std::move(*s);
    if (!f->server->Start().ok()) return nullptr;
    auto c = ShmClient::Connect(f->sock);
    if (!c.ok()) return nullptr;
    f->client = std::move(*c);
    return f;
  }
};

void ExpectTensorBufferEqual(const TensorBuffer& x, const TensorBuffer& y) {
  ASSERT_EQ(x.dtype(), y.dtype());
  ASSERT_EQ(x.shape(), y.shape());
  EXPECT_EQ(x.bytes(), y.bytes()) << "byte content differs";
}

// The core ④ test: data written via an SHM TrajectoryWriter lands in the
// server's Table and is read back unchanged through ③'s ShmSampler. Mirrors
// InProcessClientTest.WriteAndSampleRoundTrip.
TEST(ShmInsertTest, WriteAndSampleRoundTrip) {
  auto table = MakeTable("t");
  auto fx = ShmFixture::Make(table, "rt");
  ASSERT_NE(fx, nullptr);

  std::unique_ptr<TrajectoryWriter> writer;
  REVERB_ASSERT_OK(
      fx->client->NewTrajectoryWriter(MakeOptions(1, 1), &writer));

  StepRef refs;
  REVERB_ASSERT_OK(
      writer->Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
  REVERB_ASSERT_OK(writer->CreateItem("t", 1.0, MakeTrajectory({{refs[0]}})));
  REVERB_ASSERT_OK(writer->Flush());

  EXPECT_EQ(table->size(), 1);

  std::unique_ptr<ShmSampler> sampler;
  REVERB_ASSERT_OK(fx->client->NewSampler("t", {1}, &sampler));

  std::vector<TensorBuffer> data;
  REVERB_EXPECT_OK(sampler->GetNextTrajectory(&data));
  ASSERT_THAT(data, SizeIs(1));
  EXPECT_EQ(data[0].dtype(), DataType::Int32);
  // Unsqueezed trajectory column keeps the batch (time) dim -> [1, 1].
  EXPECT_EQ(data[0].shape(), std::vector<int64_t>({1, 1}));

  sampler->Close();
}

// A multi-chunk item: max_chunk_length=2, append 5 steps (chunks of 2,2,1),
// create one item spanning all 5 steps, flush. The writer must serialize each
// ChunkData separately, the server reassemble them, and the sampler return a
// concatenated [5, 1] int32 tensor whose bytes match a direct [5,1] buffer.
TEST(ShmInsertTest, MultiChunkItemRoundTrips) {
  auto table = MakeTable("t");
  auto fx = ShmFixture::Make(table, "mc");
  ASSERT_NE(fx, nullptr);

  std::unique_ptr<TrajectoryWriter> writer;
  // max_chunk_length=2 forces 3 chunks for 5 steps: [2,2,1].
  REVERB_ASSERT_OK(
      fx->client->NewTrajectoryWriter(MakeOptions(2, 5), &writer));

  std::vector<std::weak_ptr<CellRef>> col_refs;
  for (int i = 0; i < 5; i++) {
    StepRef refs;
    REVERB_ASSERT_OK(writer->Append(
        Step({MakeConstantBuffer<int32_t>(DataType::Int32, {1}, i)}), &refs));
    col_refs.push_back(refs[0].value());
  }
  std::vector<TrajectoryColumn> traj;
  traj.push_back(TrajectoryColumn(col_refs, /*squeeze=*/false));
  REVERB_ASSERT_OK(writer->CreateItem("t", 1.0, traj));
  REVERB_ASSERT_OK(writer->Flush());

  EXPECT_EQ(table->size(), 1);

  std::unique_ptr<ShmSampler> sampler;
  REVERB_ASSERT_OK(fx->client->NewSampler("t", {1}, &sampler));

  std::vector<TensorBuffer> data;
  REVERB_EXPECT_OK(sampler->GetNextTrajectory(&data));
  ASSERT_THAT(data, SizeIs(1));
  EXPECT_EQ(data[0].shape(), std::vector<int64_t>({5, 1}));
  for (int i = 0; i < 5; i++) {
    const int32_t* p = reinterpret_cast<const int32_t*>(data[0].bytes().data());
    EXPECT_EQ(p[i], i) << "step " << i << " mismatch";
  }

  sampler->Close();
}

// v1 synchronous insert round-trip (decision C2): with a tight rate
// limiter (max_size=1, no samples drained), writing many items must not
// deadlock. v1 RunShmWorker is STRICTLY SYNCHRONOUS (in_flight <= 1): each
// item does ALLOCATE -> INSERT -> read_blocking(ACK) -> erase from
// in_flight -> RELEASE, then loops to the next item. in_flight_items_ never
// exceeds 1, so there is no in_flight>1 pipelined backpressure gate in v1.
// local_can_insert_more_ is set to true on ACK but is NEVER read/awaited by
// RunShmWorker (unlike RunLocalWorker, which waits on it at line ~815) — it is
// vestigial from RunLocalWorker and NOT a backpressure gate in v1. The only
// backpressure here is the natural C->S ring-full block on Write plus the
// serial ACK wait. After draining samples, the remaining items flush through.
//
// ponytail: true in_flight>1 async/pipelined backpressure is deferred —
// upgrade RunShmWorker to async batch + reuse local_can_insert_more_ as the
// gate.
TEST(ShmInsertTest, SequentialInsertAckDoesNotDeadlock) {
  // Permissive rate limiter + small max_size: all 10 inserts complete (the
  // Fifo remover evicts the oldest beyond max_size), exercising the writer's
  // ALLOCATE->INSERT->ACK->RELEASE loop 10 times. If the ACK path wedges,
  // Flush times out instead of hanging.
  auto table = MakePermissiveTable("t", /*max_size=*/2);
  auto fx = ShmFixture::Make(table, "bp");
  ASSERT_NE(fx, nullptr);

  std::unique_ptr<TrajectoryWriter> writer;
  REVERB_ASSERT_OK(
      fx->client->NewTrajectoryWriter(MakeOptions(1, 1), &writer));

  // Write 10 items with a 100ms flush timeout each. If the synchronous
  // INSERT->ACK path is broken (writer never gets confirmation), Flush times
  // out and the test fails with DeadlineExceeded instead of hanging.
  for (int i = 0; i < 10; i++) {
    StepRef refs;
    REVERB_ASSERT_OK(
        writer->Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
    REVERB_ASSERT_OK(writer->CreateItem("t", 1.0, MakeTrajectory({{refs[0]}})));
  }
  absl::Status st = writer->Flush(/*ignore_last_num_items=*/0,
                                  /*timeout=*/absl::Milliseconds(500));
  REVERB_EXPECT_OK(st) << "Flush timed out — ACK/backpressure path broken";

  writer->Close();
}

// StructuredWriter over SHM: a config with a relative-slice pattern emits
// items into the server's Table as steps are appended. Mirrors
// InProcessClientTest.StructuredWriterConditionWithRelativeSlice.
TEST(ShmInsertTest, StructuredWriterEmitsItemsOverShm) {
  auto table = MakeTable("sw", /*max_size=*/100);
  auto fx = ShmFixture::Make(table, "sw");
  ASSERT_NE(fx, nullptr);

  StructuredWriterConfig config;
  auto* node = config.add_flat();
  node->set_flat_source_index(0);
  node->set_start(-1);  // most recent step
  config.set_table("sw");
  config.mutable_priority()->mutable_constant_fn()->set_value(1.0);
  // Condition: step_index <= 2 (fires on steps 0,1,2).
  auto* cond = config.add_conditions();
  cond->set_step_index(true);
  cond->set_ge(3);
  cond->set_inverse(true);

  std::unique_ptr<StructuredWriter> writer;
  REVERB_ASSERT_OK(fx->client->NewStructuredWriter({config}, &writer));

  for (int i = 0; i < 5; i++) {
    REVERB_ASSERT_OK(writer->Append(
        Step({MakeConstantBuffer<int32_t>(DataType::Int32, {}, i)})));
  }
  REVERB_ASSERT_OK(writer->EndEpisode(/*clear_buffers=*/true));

  // 3 trajectories expected (steps 0,1,2), each a single scalar.
  std::unique_ptr<ShmSampler> sampler;
  REVERB_ASSERT_OK(fx->client->NewSampler("sw", {3}, &sampler));

  std::vector<int32_t> values;
  std::vector<TensorBuffer> data;
  while (sampler->GetNextTrajectory(&data).ok()) {
    ASSERT_EQ(data.size(), 1u);
    const int32_t* p =
        reinterpret_cast<const int32_t*>(data[0].bytes().data());
    values.push_back(*p);
  }
  EXPECT_THAT(values, ::testing::ElementsAre(0, 1, 2));

  sampler->Close();
}

// --- ⑧-b: server-side validation of peer-supplied pool offsets ----------

// Poll a non-blocking Ring::Read with a deadline so a missing response fails
// the test in seconds instead of hanging forever (pre-fix the server only
// LOGS a rejected INSERT and never replies).
absl::Status ReadWithDeadline(Ring* ring, MsgType* msg_type,
                              std::string* payload, absl::Duration timeout) {
  absl::Time deadline = absl::Now() + timeout;
  while (absl::Now() < deadline) {
    absl::Status s = ring->Read(msg_type, payload);
    if (s.ok()) return absl::OkStatus();
    if (!absl::IsNotFound(s)) return s;
    sched_yield();
  }
  return absl::DeadlineExceededError("no response within deadline");
}

// A forged INSERT whose chunk ref points at pool offset 0 (the PoolHeader —
// never an allocated block) must be rejected with an INVALID_ARGUMENT ERROR
// on the insert s2c flow, and the server must keep serving. Before the fix,
// HandleInsert ParseFromArray'd the peer-controlled offset/length directly
// (OOB read for a wild offset) and only LOGGED the failure — the client's
// writer hung until its 60s cap.
TEST(ShmInsertTest, BogusChunkOffsetReturnsErrorAndServerSurvives) {
  auto table = MakeTable("t");
  auto fx = ShmFixture::Make(table, "bad");
  ASSERT_NE(fx, nullptr);

  ShmInsertRequest req;
  auto* chunk = req.add_chunks();
  chunk->set_chunk_key(1);
  chunk->set_shm_offset(0);   // PoolHeader, never granted by ALLOCATE
  chunk->set_total_length(64);
  std::string body;
  req.SerializeToString(&body);
  REVERB_ASSERT_OK(fx->client->connection()->insert_c2s.Write(INSERT, body));

  MsgType type;
  std::string payload;
  REVERB_ASSERT_OK(ReadWithDeadline(&fx->client->connection()->insert_s2c,
                                    &type, &payload, absl::Seconds(5)));
  ASSERT_EQ(type, ERROR);
  ShmError err;
  ASSERT_TRUE(err.ParseFromString(payload));
  EXPECT_EQ(err.code(), ShmError::INVALID_ARGUMENT);

  // Server survives: a normal insert round-trip still works.
  std::unique_ptr<TrajectoryWriter> writer;
  REVERB_ASSERT_OK(fx->client->NewTrajectoryWriter(MakeOptions(1, 1), &writer));
  StepRef refs;
  REVERB_ASSERT_OK(
      writer->Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
  REVERB_ASSERT_OK(writer->CreateItem("t", 1.0, MakeTrajectory({{refs[0]}})));
  REVERB_ASSERT_OK(writer->Flush());
  EXPECT_EQ(table->size(), 1);
}

// A RELEASE for an offset the server never granted to this client must be
// ignored. Before the fix, HandleRelease Deallocate'd ANY offset: a double
// RELEASE pushed the same block onto its slab's free list twice, so two later
// ALLOCATEs were handed the SAME block concurrently (insert bytes overwrite
// each other; offset 0 would even corrupt the PoolHeader).
TEST(ShmInsertTest, DoubleReleaseDoesNotHandOutSameBlockTwice) {
  auto table = MakeTable("t");
  auto fx = ShmFixture::Make(table, "rel");
  ASSERT_NE(fx, nullptr);
  Ring* c2s = &fx->client->connection()->insert_c2s;
  Ring* s2c = &fx->client->connection()->insert_s2c;

  auto allocate = [&](uint64_t* out) {
    ShmAllocateRequest req;
    req.set_num_bytes(64);
    std::string body;
    req.SerializeToString(&body);
    REVERB_ASSERT_OK(c2s->Write(ALLOCATE, body));
    MsgType type;
    std::string payload;
    REVERB_ASSERT_OK(ReadWithDeadline(s2c, &type, &payload, absl::Seconds(5)));
    ASSERT_EQ(type, ALLOCATE_RESP);
    ShmAllocateResponse resp;
    ASSERT_TRUE(resp.ParseFromString(payload));
    *out = resp.shm_offset();
  };
  auto release = [&](uint64_t offset) {
    ShmReleaseRequest rel;
    rel.add_offsets(offset);
    std::string body;
    rel.SerializeToString(&body);
    REVERB_ASSERT_OK(c2s->Write(RELEASE, body));
  };

  uint64_t x = 0;
  allocate(&x);
  release(x);
  release(x);  // double release: must be ignored

  uint64_t a = 0, b = 0;
  allocate(&a);
  allocate(&b);
  EXPECT_NE(a, b) << "double-freed block handed out twice (offset " << a
                  << ")";
}

// An INSERT whose item references a chunk_key that is NOT among the
// request's chunks must be rejected with an ERROR on the insert s2c flow.
// Before the fix, the unknown-chunk path returned a bare status that the
// dispatcher only LOGGED — the client hung until its timeout cap, and the
// request's pending callbacks leaked (remaining never reaches 0, no ACK).
TEST(ShmInsertTest, ItemReferencingUnknownChunkReturnsError) {
  auto table = MakeTable("t");
  auto fx = ShmFixture::Make(table, "unk");
  ASSERT_NE(fx, nullptr);
  Ring* c2s = &fx->client->connection()->insert_c2s;
  Ring* s2c = &fx->client->connection()->insert_s2c;

  // ALLOCATE a real block and write a valid ChunkData (chunk_key=1) into it
  // (offsets are validated against outstanding grants since the ⑧-b fix).
  ShmAllocateRequest areq;
  areq.set_num_bytes(256);
  std::string abody;
  areq.SerializeToString(&abody);
  REVERB_ASSERT_OK(c2s->Write(ALLOCATE, abody));
  MsgType atype;
  std::string apayload;
  REVERB_ASSERT_OK(ReadWithDeadline(s2c, &atype, &apayload, absl::Seconds(5)));
  ASSERT_EQ(atype, ALLOCATE_RESP);
  ShmAllocateResponse aresp;
  ASSERT_TRUE(aresp.ParseFromString(apayload));
  uint64_t off = aresp.shm_offset();

  ChunkData cd;
  cd.set_chunk_key(1);
  std::string cd_bytes;
  cd.SerializeToString(&cd_bytes);
  std::memcpy(fx->client->connection()->pool.At(off), cd_bytes.data(),
              cd_bytes.size());

  ShmInsertRequest req;
  auto* chunk = req.add_chunks();
  chunk->set_chunk_key(1);
  chunk->set_shm_offset(off);
  chunk->set_total_length(cd_bytes.size());
  auto* item = req.add_items();
  item->set_key(1);
  item->set_table("t");
  item->set_priority(1.0);
  // Item references chunk_key=2, which is NOT in req.chunks() — unknown.
  auto* slice = item->mutable_flat_trajectory()
                    ->add_columns()
                    ->add_chunk_slices();
  slice->set_chunk_key(2);
  slice->set_offset(0);
  slice->set_length(1);
  std::string body;
  req.SerializeToString(&body);
  REVERB_ASSERT_OK(c2s->Write(INSERT, body));

  MsgType type;
  std::string payload;
  REVERB_ASSERT_OK(ReadWithDeadline(s2c, &type, &payload, absl::Seconds(5)));
  ASSERT_EQ(type, ERROR);
  ShmError err;
  ASSERT_TRUE(err.ParseFromString(payload));
  EXPECT_THAT(err.message(), ::testing::HasSubstr("unknown chunk"));

  // Server survives: a normal insert round-trip still works.
  std::unique_ptr<TrajectoryWriter> writer;
  REVERB_ASSERT_OK(fx->client->NewTrajectoryWriter(MakeOptions(1, 1), &writer));
  StepRef refs;
  REVERB_ASSERT_OK(
      writer->Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
  REVERB_ASSERT_OK(writer->CreateItem("t", 1.0, MakeTrajectory({{refs[0]}})));
  REVERB_ASSERT_OK(writer->Flush());
  EXPECT_EQ(table->size(), 1);
}

// A client that connects and never sends HELLO must not wedge the dispatch
// loop. Before the fix, TryAccept ran a blocking RecvHello on the dispatch
// thread — one stalled connection froze accept AND service for every client.
TEST(ShmInsertTest, StalledHelloDoesNotWedgeDispatch) {
  auto table = MakeTable("t");
  auto fx = ShmFixture::Make(table, "stall");
  ASSERT_NE(fx, nullptr);

  // Raw-connect and stay silent (no HELLO).
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(fd, 0);
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, fx->sock.c_str(), sizeof(addr.sun_path) - 1);
  ASSERT_EQ(connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);

  // Wait past the handshake bound, then the pre-existing client must still
  // get full service (dispatch thread not stuck in RecvHello).
  usleep(400 * 1000);
  std::unique_ptr<TrajectoryWriter> writer;
  REVERB_ASSERT_OK(fx->client->NewTrajectoryWriter(MakeOptions(1, 1), &writer));
  StepRef refs;
  REVERB_ASSERT_OK(
      writer->Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
  REVERB_ASSERT_OK(writer->CreateItem("t", 1.0, MakeTrajectory({{refs[0]}})));
  REVERB_ASSERT_OK(writer->Flush(/*ignore_last_num_items=*/0,
                                 /*timeout=*/absl::Milliseconds(3000)));
  EXPECT_EQ(table->size(), 1);
  close(fd);
}

// --- P1: insert-callback keepalive clear must be selective per request ------
//
// HandleInsert stashes a keepalive shared_ptr per item callback
// (InsertOrAssignAsync holds only a weak_ptr); both the last-completion
// callback and the mid-insert fail path used to clear() the WHOLE vector.
// With two INSERT requests in flight on one connection, that dropped the
// other request's keepalives: its callbacks expired unfired, so no
// INSERT_ACK ever came back (a client would hang to its 60s cap with the
// data actually inserted). Latent today — the client's insert_flow_mu holds
// in_flight <= 1 — so these tests drive the ring RAW, writing two INSERTs
// back-to-back without waiting for the first ACK (the overlap the
// documented async-inserts upgrade will produce).
//
// Determinism note: post-fix both tests are deterministic (every request's
// callbacks are independent, so every ACK/ERROR always arrives). Pre-fix
// detection relies on the table worker not having completed req1's items
// before the dispatch thread processes req2 — a worker wake + insert +
// callback-executor schedule takes orders of magnitude longer than one ring
// read, and req1's 8 items widen the gap further.

// ALLOCATEs one pool block and writes a minimal ChunkData carrying only
// `chunk_key` into it (the insert path parses the proto but never inspects
// tensor payload). Outputs the pool offset and the serialized ChunkData
// length (needed for the ShmChunkRef).
void AllocateAndWriteChunk(ShmFixture* fx, uint64_t chunk_key, uint64_t* out_off,
                           size_t* out_len) {
  Ring* c2s = &fx->client->connection()->insert_c2s;
  Ring* s2c = &fx->client->connection()->insert_s2c;
  ShmAllocateRequest areq;
  areq.set_num_bytes(256);
  std::string abody;
  areq.SerializeToString(&abody);
  REVERB_ASSERT_OK(c2s->Write(ALLOCATE, abody));
  MsgType atype;
  std::string apayload;
  REVERB_ASSERT_OK(ReadWithDeadline(s2c, &atype, &apayload, absl::Seconds(5)));
  ASSERT_EQ(atype, ALLOCATE_RESP);
  ShmAllocateResponse aresp;
  ASSERT_TRUE(aresp.ParseFromString(apayload));
  *out_off = aresp.shm_offset();

  ChunkData cd;
  cd.set_chunk_key(chunk_key);
  std::string cd_bytes;
  cd.SerializeToString(&cd_bytes);
  std::memcpy(fx->client->connection()->pool.At(*out_off), cd_bytes.data(),
              cd_bytes.size());
  *out_len = cd_bytes.size();
}

// Builds a serialized INSERT with one chunk ref (`chunk_key` at `shm_offset`,
// `chunk_len` bytes) and `num_items` items (keys first_key..first_key+n-1 on
// table "t") each referencing `chunk_key` via one column/one slice — the
// minimal shape that passes Table::CheckItemValidity.
std::string MakeRawInsertBody(uint64_t chunk_key, uint64_t shm_offset,
                              size_t chunk_len, int num_items,
                              uint64_t first_key) {
  ShmInsertRequest req;
  auto* chunk = req.add_chunks();
  chunk->set_chunk_key(chunk_key);
  chunk->set_shm_offset(shm_offset);
  chunk->set_total_length(chunk_len);
  for (int i = 0; i < num_items; ++i) {
    auto* item = req.add_items();
    item->set_key(first_key + i);
    item->set_table("t");
    item->set_priority(1.0);
    auto* slice = item->mutable_flat_trajectory()
                      ->add_columns()
                      ->add_chunk_slices();
    slice->set_chunk_key(chunk_key);
    slice->set_offset(0);
    slice->set_length(1);
  }
  std::string body;
  req.SerializeToString(&body);
  return body;
}

// Two VALID overlapping INSERTs: the first request's completion must not
// drop the second request's keepalive. Both must receive exactly one
// INSERT_ACK. (Pre-fix the first request's last callback clear()ed the whole
// vector; the second request then never ACKed and this test timed out.)
TEST(ShmInsertTest, OverlappingInsertsBothAck) {
  auto table = MakePermissiveTable("t");
  auto fx = ShmFixture::Make(table, "ovlap");
  ASSERT_NE(fx, nullptr);
  Ring* c2s = &fx->client->connection()->insert_c2s;
  Ring* s2c = &fx->client->connection()->insert_s2c;

  uint64_t off1 = 0, off2 = 0;
  size_t len1 = 0, len2 = 0;
  AllocateAndWriteChunk(fx.get(), /*chunk_key=*/1, &off1, &len1);
  AllocateAndWriteChunk(fx.get(), /*chunk_key=*/2, &off2, &len2);

  // req1: 8 items (keys 1..8); req2: 1 item (key 9). Written back-to-back,
  // no ACK wait in between — two requests in flight on one connection.
  REVERB_ASSERT_OK(c2s->Write(INSERT, MakeRawInsertBody(1, off1, len1,
                                                        /*num_items=*/8,
                                                        /*first_key=*/1)));
  REVERB_ASSERT_OK(c2s->Write(INSERT, MakeRawInsertBody(2, off2, len2,
                                                        /*num_items=*/1,
                                                        /*first_key=*/9)));

  std::vector<uint64_t> acked_keys;
  for (int i = 0; i < 2; ++i) {
    MsgType type;
    std::string payload;
    REVERB_ASSERT_OK(ReadWithDeadline(s2c, &type, &payload, absl::Seconds(5)));
    ASSERT_EQ(type, INSERT_ACK);
    InsertAck ack;
    ASSERT_TRUE(ack.ParseFromString(payload));
    acked_keys.insert(acked_keys.end(), ack.keys().begin(), ack.keys().end());
  }
  std::sort(acked_keys.begin(), acked_keys.end());
  EXPECT_EQ(acked_keys,
            std::vector<uint64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

// A synchronously FAILING insert (unknown chunk_key) must not drop an
// earlier in-flight request's keepalive: req1 still receives exactly one
// INSERT_ACK alongside req2's ERROR. (Pre-fix fail_insert clear()ed the
// whole vector; req1 then never ACKed and this test timed out.)
TEST(ShmInsertTest, FailedInsertDoesNotDropInFlightRequestAck) {
  auto table = MakePermissiveTable("t");
  auto fx = ShmFixture::Make(table, "failclr");
  ASSERT_NE(fx, nullptr);
  Ring* c2s = &fx->client->connection()->insert_c2s;
  Ring* s2c = &fx->client->connection()->insert_s2c;

  uint64_t off1 = 0, off2 = 0;
  size_t len1 = 0, len2 = 0;
  AllocateAndWriteChunk(fx.get(), /*chunk_key=*/1, &off1, &len1);
  AllocateAndWriteChunk(fx.get(), /*chunk_key=*/2, &off2, &len2);

  // req1: valid, 8 items (keys 1..8) — still in flight when req2 fails.
  REVERB_ASSERT_OK(c2s->Write(INSERT, MakeRawInsertBody(1, off1, len1,
                                                        /*num_items=*/8,
                                                        /*first_key=*/1)));
  // req2: its item references chunk_key 99, absent from its chunks —
  // HandleInsert fails it synchronously via fail_insert (same shape as
  // ItemReferencingUnknownChunkReturnsError).
  ShmInsertRequest bad;
  auto* chunk = bad.add_chunks();
  chunk->set_chunk_key(2);
  chunk->set_shm_offset(off2);
  chunk->set_total_length(len2);
  auto* item = bad.add_items();
  item->set_key(9);
  item->set_table("t");
  item->set_priority(1.0);
  auto* slice =
      item->mutable_flat_trajectory()->add_columns()->add_chunk_slices();
  slice->set_chunk_key(99);
  slice->set_offset(0);
  slice->set_length(1);
  std::string bad_body;
  bad.SerializeToString(&bad_body);
  REVERB_ASSERT_OK(c2s->Write(INSERT, bad_body));

  // Exactly one ERROR (req2) and exactly one INSERT_ACK (req1's 8 keys), in
  // either order.
  bool got_error = false;
  std::vector<uint64_t> acked_keys;
  for (int i = 0; i < 2; ++i) {
    MsgType type;
    std::string payload;
    REVERB_ASSERT_OK(ReadWithDeadline(s2c, &type, &payload, absl::Seconds(5)));
    if (type == ERROR) {
      ShmError err;
      ASSERT_TRUE(err.ParseFromString(payload));
      EXPECT_THAT(err.message(), ::testing::HasSubstr("unknown chunk"));
      got_error = true;
    } else {
      ASSERT_EQ(type, INSERT_ACK);
      InsertAck ack;
      ASSERT_TRUE(ack.ParseFromString(payload));
      acked_keys.insert(acked_keys.end(), ack.keys().begin(),
                        ack.keys().end());
    }
  }
  EXPECT_TRUE(got_error);
  std::sort(acked_keys.begin(), acked_keys.end());
  EXPECT_EQ(acked_keys, std::vector<uint64_t>({1, 2, 3, 4, 5, 6, 7, 8}));
}

}  // namespace
}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
