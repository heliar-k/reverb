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

#ifndef REVERB_CC_SHM_SHM_CLIENT_H_
#define REVERB_CC_SHM_SHM_CLIENT_H_

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/patterns.pb.h"
#include "reverb/cc/sampler.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/shm/byte_pool.h"
#include "reverb/cc/shm/ring.h"
#include "reverb/cc/shm/shm_connection.h"
#include "reverb/cc/shm/shm_protocol.pb.h"
#include "reverb/cc/structured_writer.h"
#include "reverb/cc/support/queue.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/trajectory_writer.h"
#include "reverb/cc/writer.h"

namespace deepmind {
namespace reverb {
namespace shm {

// ShmSampler is the client-side sampler over the SHM transport. Per decision
// C5 it reuses the existing Sampler architecture (a worker thread feeding a
// `samples_` Queue, popped by GetNextTrajectory) — only the per-sample fetch
// swaps the gRPC/local Table::Sample call for an SHM ring round-trip.
//
// GIL discipline matches the existing Sampler: the worker thread builds
// TensorBuffers (no GIL) from pool bytes; GetNextTrajectory runs on the caller
// thread where ToNdArray() may acquire the GIL.
//
// ponytail: a thin class mirroring Sampler's public API rather than a third
// SamplerWorker injected into Sampler (that constructor is private). diff is
// smaller than teaching Sampler about a third transport.
class ShmSampler {
 public:
  // `conn` is borrowed (owned by ShmClient); must outlive the sampler.
  // `table_name` is the server-side table to sample from. `options` mirrors
  // Sampler::Options (only max_samples + rate_limiter_timeout matter for v1).
  static absl::StatusOr<std::unique_ptr<ShmSampler>> Create(
      ShmConnection* conn, const std::string& table_name,
      const Sampler::Options& options);

  ~ShmSampler();

  // Blocks until a complete sample is retrieved. Populates `data` with the
  // full (flattened) trajectory as TensorBuffers (one per column). The caller
  // may call TensorBuffer::ToNdArray() on the caller thread (GIL).
  absl::Status GetNextTrajectory(
      std::vector<TensorBuffer>* data,
      std::shared_ptr<const SampleInfo>* info = nullptr);

  // Cancels the worker and joins its thread.
  void Close();

 private:
  ShmSampler(ShmConnection* conn, std::string table_name,
             int64_t max_samples, absl::Duration rate_limiter_timeout);

  // Worker thread main loop: fetch samples until max_samples or cancelled.
  void RunWorker();

  // One SHM round-trip: send SAMPLE, poll for SAMPLE_RESP (or ERROR), build a
  // Sample from pool bytes, send RELEASE. Returns the sample or an error.
  absl::StatusOr<std::unique_ptr<Sample>> FetchOne();

  ShmConnection* conn_;  // borrowed
  const std::string table_name_;
  const int64_t max_samples_;
  const absl::Duration rate_limiter_timeout_;

  internal::Queue<std::unique_ptr<Sample>> samples_;

  std::atomic<bool> closed_{false};
  absl::Status worker_status_ ABSL_GUARDED_BY(mu_);
  int64_t requested_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t returned_ ABSL_GUARDED_BY(mu_) = 0;
  std::unique_ptr<internal::Thread> worker_thread_;
  mutable absl::Mutex mu_;
};

// ShmClient connects to a ShmServer over a udsocket, handshakes, and mmaps the
// three shared segments (pool + two rings). NewSampler returns a ShmSampler.
class ShmClient {
 public:
  static absl::StatusOr<std::unique_ptr<ShmClient>> Connect(
      const std::string& socket_path);

  ~ShmClient();

  // Returns a sampler over `table_name`. `options` mirrors Sampler::Options
  // (max_samples + rate_limiter_timeout are the v1-relevant fields).
  absl::Status NewSampler(const std::string& table_name,
                          const Sampler::Options& options,
                          std::unique_ptr<ShmSampler>* sampler);

  // ---- Writer path (ticket ④) ----

  // Constructs a TrajectoryWriter in SHM mode: the chunker/column/backpressure
  // logic runs client-side, but inserts go over SHM to the server's Table
  // (appendix A4). `options.flat_signature_map` is left as-is (no
  // ServerInfo round-trip in v1); pass a populated map if you want
  // ItemAndRefs::Validate to check trajectory signatures against a known
  // table signature.
  absl::Status NewTrajectoryWriter(const TrajectoryWriter::Options& options,
                                   std::unique_ptr<TrajectoryWriter>* writer);

  // Mirrors InProcessClient::NewStructuredWriter: runs
  // PrepareStructuredWriterConfigs to compute max_num_keep_alive_refs, builds
  // AutoTunedChunkerOptions, and wraps a SHM TrajectoryWriter. Each config's
  // `table` field routes its item to the server-side table.
  absl::Status NewStructuredWriter(
      std::vector<StructuredWriterConfig> configs,
      std::unique_ptr<StructuredWriter>* writer);

  // ponytail: the plain Writer (writer.h) has no SHM seam — its local ctor
  // takes a tables map, but an SHM client holds no tables (the server does).
  // Adding an SHM transport to Writer would duplicate RunShmWorker's logic for
  // a legacy API. Deferred: callers that need SHM inserts should use
  // NewTrajectoryWriter / NewStructuredWriter. Upgrade: either add an SHM
  // ctor to Writer mirroring TrajectoryWriter's, or deprecate Writer for SHM
  // clients. TODO(④): implement if a caller needs it.
  absl::Status NewWriter(int chunk_length, int max_timesteps,
                         bool delta_encoded, int max_in_flight_items,
                         std::unique_ptr<Writer>* writer);

  // Returns the bootstrap-time snapshot of the server's TableInfo (one entry
  // per table the server held at Connect time). ticket ⑧ step 1: this is
  // piggybacked on the SHM bootstrap handshake — there is no on-demand
  // SERVER_INFO ring round-trip (that is step 2, deferred).
  // ponytail: bootstrap-time snapshot only; does NOT reflect mid-session
  // Table.replace / signature changes. Ceiling: a long-lived client whose
  // table is replaced mid-session sees stale info. Upgrade path: add a
  // SERVER_INFO/SERVER_INFO_RESP MsgType + HandleServerInfo for an on-demand
  // round-trip (ticket ⑧ step 2).
  absl::Status ServerInfo(std::vector<TableInfo>* table_info);

  // ticket ⑩: control-plane ops, mirroring InProcessClient::MutatePriorities /
  // Reset. These ride the INSERT flow (insert_c2s/insert_s2c) under
  // conn_.insert_flow_mu so they never race RunShmWorker as a second producer
  // on insert_c2s. Unknown table -> absl::NotFoundError (server replies
  // ShmError::NOT_FOUND), surfaced as Python FileNotFoundError. Reuses the
  // existing reverb_service.proto MutatePrioritiesRequest/ResetRequest types.
  absl::Status MutatePriorities(const std::string& table,
                                const std::vector<KeyWithPriority>& updates,
                                const std::vector<uint64_t>& deletes);
  absl::Status Reset(const std::string& table);

  ShmConnection* connection() { return &conn_; }

 private:
  explicit ShmClient(ShmConnection conn,
                     std::vector<TableInfo> cached_server_info);

  ShmConnection conn_;
  std::vector<TableInfo> cached_server_info_;
};

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SHM_SHM_CLIENT_H_
