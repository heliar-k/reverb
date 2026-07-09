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
#include "reverb/cc/sampler.h"
#include "reverb/cc/shm/byte_pool.h"
#include "reverb/cc/shm/ring.h"
#include "reverb/cc/shm/shm_connection.h"
#include "reverb/cc/shm/shm_protocol.pb.h"
#include "reverb/cc/support/queue.h"
#include "reverb/cc/support/tensor_proxy.h"

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

  ShmConnection* connection() { return &conn_; }

 private:
  explicit ShmClient(ShmConnection conn);

  ShmConnection conn_;
};

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SHM_SHM_CLIENT_H_
