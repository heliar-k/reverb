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

#include "reverb/cc/trajectory_writer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sched.h>
#include <string>
#include <utility>
#include <vector>

#include "grpcpp/client_context.h"
#include "grpcpp/impl/codegen/sync_stream.h"
#include "grpcpp/support/status.h"
#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/platform/default/hash_map.h"
#include "reverb/cc/platform/default/hash_set.h"
#include "reverb/cc/platform/default/logging.h"
#include "reverb/cc/platform/default/status_macros.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/grpc_util.h"
#include "reverb/cc/support/key_generators.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/support/trajectory_util.h"
#include "reverb/cc/shm/shm_connection.h"
#include "reverb/cc/shm/shm_protocol.pb.h"

namespace deepmind {
namespace reverb {

// If the total number of pending items (waiting to be sent + waiting for
// confirmation from server) grows beyond this value then we'll start logging
// warning messages to catch the attention of the user.
const int kPendingItemsWarningThreshold = 50;

class ArenaOwnedRequest {
 public:
  ~ArenaOwnedRequest() { Clear(); }

  void Clear() {
    while (!r_.chunks().empty()) {
      r_.mutable_chunks()->UnsafeArenaReleaseLast();
    }
    while (!r_.items().empty()) {
      r_.mutable_items()->UnsafeArenaReleaseLast();
    }
    r_.clear_keep_chunk_keys();
    request_size_bytes_ = 0;
  }
  inline const InsertStreamRequest& Request() { return r_; }
  inline void AddAllocatedChunks(ChunkData* data) {
    r_.mutable_chunks()->UnsafeArenaAddAllocated(data);
    request_size_bytes_ += data->ByteSizeLong();
  }
  inline int64_t RequestSize() { return request_size_bytes_; }
  inline void AddKeepChunkKeys(uint64_t keep_key) {
    r_.add_keep_chunk_keys(keep_key);
    request_size_bytes_ += sizeof(uint64_t);
  }
  inline void AddItem(const PrioritizedItem& item) {
    r_.mutable_items()->UnsafeArenaAddAllocated(
        const_cast<PrioritizedItem*>(&item));
    request_size_bytes_ += item.ByteSizeLong();
    request_size_bytes_ -= r_.keep_chunk_keys_size() * sizeof(uint64_t);
    r_.clear_keep_chunk_keys();
  }

 private:
  InsertStreamRequest r_;
  int64_t request_size_bytes_ = 0;
};

namespace {

std::vector<FlatTrajectory::ChunkSlice> MergeAdjacent(
    const std::vector<std::weak_ptr<CellRef>>& refs) {
  std::vector<FlatTrajectory::ChunkSlice> slices;
  for (const std::weak_ptr<CellRef>& ref : refs) {
    // Caller (TrajectoryWriter) is responsible for ensuring that all of the
    // weak pointers are alive.
    std::shared_ptr<CellRef> ref_sp = ref.lock();
    REVERB_CHECK(ref_sp);

    if (slices.empty() || slices.back().chunk_key() != ref_sp->chunk_key()) {
      FlatTrajectory::ChunkSlice slice;
      slice.set_chunk_key(ref_sp->chunk_key());
      slice.set_offset(ref_sp->offset());
      slice.set_length(1);
      slices.push_back(std::move(slice));
    } else {
      slices.back().set_length(slices.back().length() + 1);
    }
  }
  return slices;
}

// Returns true if all references `refs` are ready.
bool AllReady(absl::Span<const std::shared_ptr<CellRef>> refs) {
  return absl::c_all_of(refs, [](const auto& ref) { return ref->IsReady(); });
}

// Formats a shape vector as "[d0,d1,...]" with -1 shown as "?", matching the
// diagnostic strings used elsewhere in the codebase.
std::string ShapeString(const std::vector<int64_t>& shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) absl::StrAppend(&s, ",");
    absl::StrAppend(&s, shape[i] == -1 ? "?" : std::to_string(shape[i]));
  }
  absl::StrAppend(&s, "]");
  return s;
}

// Returns true if `set` contains all chunk keys references by `refs`.
bool ContainsAll(const internal::flat_hash_set<uint64_t>& set,
                 absl::Span<const std::shared_ptr<CellRef>> refs) {
  return absl::c_all_of(
      refs, [&set](const auto& ref) { return set.contains(ref->chunk_key()); });
}

std::vector<internal::TensorSpec> FlatSignatureFromTrajectory(
    const FlatTrajectory& trajectory,
    absl::Span<const std::shared_ptr<CellRef>> refs) {
  auto get_spec = [&](uint64_t chunk_key) -> internal::TensorSpec {
    for (const auto& ref : refs) {
      if (ref->chunk_key() == chunk_key) {
        return ref->chunker().lock()->spec();
      }
    }
    REVERB_CHECK(false) << "Invalid trajectory";
    return internal::TensorSpec{};  // unreachable
  };

  std::vector<internal::TensorSpec> specs;
  for (int col_idx = 0; col_idx < trajectory.columns_size(); col_idx++) {
    const FlatTrajectory::Column& col = trajectory.columns(col_idx);
    internal::TensorSpec spec = get_spec(col.chunk_slices(0).chunk_key());
    spec.name = std::to_string(col_idx);
    if (!col.squeeze()) {
      spec.shape.insert(spec.shape.begin(),
                        internal::ColumnLength(trajectory, col_idx));
    }
    specs.push_back(std::move(spec));
  }
  return specs;
}

}  // namespace

bool TrajectoryWriter::WriteIfNotEmpty(
    const internal::flat_hash_set<uint64_t>& keep_keys,
    ArenaOwnedRequest* request) {
  if (request->RequestSize() == 0) {
    return true;
  }
  for (uint64_t keep_key : keep_keys) {
    request->AddKeepChunkKeys(keep_key);
  }
  {
    absl::MutexLock lock(mu_);
    write_inflight_ = true;
  }
  grpc::WriteOptions options;
  options.set_no_compression();
  StartWrite(&request->Request(), options);
  {
    absl::MutexLock lock(mu_);
    auto trigger = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      return !write_inflight_ || closed_ || !stream_ok_;
    };
    mu_.Await(absl::Condition(&trigger));
    request->Clear();
    return !write_inflight_;
  }
}

bool TrajectoryWriter::SendNotAlreadySentChunks(
    internal::flat_hash_set<uint64_t>* streamed_chunk_keys,
    absl::Span<const std::shared_ptr<CellRef>> refs,
    ArenaOwnedRequest* request) {
  // Send referenced chunks which haven't already been sent.
  for (const std::shared_ptr<CellRef>& ref : refs) {
    if (!ref->IsReady() || streamed_chunk_keys->contains(ref->chunk_key())) {
      continue;
    }
    ChunkData* chunk_data = const_cast<ChunkData*>(ref->GetChunk()->get());
    request->AddAllocatedChunks(chunk_data);
    streamed_chunk_keys->insert(ref->chunk_key());

    // If the message has grown beyond the cutoff point then we send it.
    if (request->RequestSize() >= TrajectoryWriter::kMaxRequestSizeBytes) {
      if (!WriteIfNotEmpty(*streamed_chunk_keys, request)) {
        return false;
      }

      // There (might) still be chunks which can be transmitted so continue with
      // the remaining references.
    }
  }
  // Remaining chunks will be sent together with the Item.
  return true;
}

absl::Status TrajectoryWriter::Options::Validate() const {
  if (chunker_options == nullptr) {
    return absl::InvalidArgumentError("chunker_options must be set.");
  }
  return ValidateChunkerOptions(chunker_options.get());
}

absl::Status TrajectoryWriter::ItemAndRefs::Validate(
    const TrajectoryWriter::Options& options) const {
  if (!options.flat_signature_map.has_value()) {
    return absl::OkStatus();
  }

  const std::string& table = item.table();
  const internal::FlatSignatureMap& signature_map =
      options.flat_signature_map.value();
  auto it = signature_map.find(table);
  if (it == signature_map.end()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unable to create item in table '%s' since the table "
                        "could not be found.",
                        table));
  }
  if (!it->second.has_value()) {
    return absl::OkStatus();
  }
  const std::vector<internal::TensorSpec>& table_signature = it->second.value();

  const FlatTrajectory& trajectory = item.flat_trajectory();
  std::vector<internal::TensorSpec> trajectory_signature =
      FlatSignatureFromTrajectory(trajectory, refs);

  if (table_signature.size() != trajectory_signature.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Unable to create item in table '%s' since the provided trajectory "
        "is inconsistent with the table signature. The trajectory has %d "
        "columns but the table signature has %d columns."
        "\n\nThe table signature is:\n\t%s"
        "\n\nThe provided trajectory signature was:\n\t%s.\n",
        table, trajectory_signature.size(), table_signature.size(),
        internal::DtypesShapesString(table_signature),
        internal::DtypesShapesString(trajectory_signature)));
  }

  for (int i = 0; i < table_signature.size(); i++) {
    const internal::TensorSpec& want = table_signature[i];
    const internal::TensorSpec& got = trajectory_signature[i];

    if (want.dtype != got.dtype || !want.IsCompatibleWith(got)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unable to create item in table '%s' since the provided trajectory "
          "is inconsistent with the table signature. The table expects column "
          "%d to be a %s %s tensor but got a %s %s tensor."
          "\n\nThe table signature is:\n\t%s"
          "\n\nThe provided trajectory signature is:\n\t%s.\n",
          table, i, DataTypeName(want.dtype),
          ShapeString(want.shape), DataTypeName(got.dtype),
          ShapeString(got.shape),
          internal::DtypesShapesString(table_signature),
          internal::DtypesShapesString(trajectory_signature)));
    }
  }

  return absl::OkStatus();
}

TrajectoryWriter::TrajectoryWriter(
    std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub,
    const Options& options)
    : stub_(std::move(stub)),
      options_(options),
      episode_id_(key_generator_.Generate()),
      episode_step_(0),
      closed_(false),
      stream_worker_(
          internal::StartThread("TrajectoryWriter_StreamWorker", [this] {
            absl::Duration retry_backoff = absl::Milliseconds(1);
            while (true) {
              absl::Time start_time = absl::Now();
              absl::Status status = RunStreamWorker();

              absl::MutexLock lock(mu_);

              if (closed_) {
                unrecoverable_status_ = absl::CancelledError(
                    "TrajectoryWriter::Close has been called.");
                return;
              }

              if (!status.ok() && !absl::IsUnavailable(status) &&
                  !absl::IsCancelled(status)) {
                unrecoverable_status_ = status;
                return;
              }

              // If we lose the connection then we'll never receive the item
              // confirmations so we add them back to the front of the queue.
              // Note that the internal ordering of the items added to the front
              // of the queue is undefined and thus may differ from how they
              // were originally transmitted.
              for (auto& [_, item_and_refs] : in_flight_items_) {
                write_queue_.push_front(std::move(item_and_refs));
              }
              in_flight_items_.clear();

              if (absl::Now() - start_time < absl::Seconds(2)) {
                retry_backoff = std::min(absl::Seconds(1), 2 * retry_backoff);
              } else {
                retry_backoff = absl::Milliseconds(1);
              }
              auto trigger = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                return closed_;
              };
              mu_.AwaitWithTimeout(absl::Condition(&trigger), retry_backoff);
            }
          })) {
  REVERB_CHECK_OK(options.Validate());
}

TrajectoryWriter::TrajectoryWriter(
    internal::flat_hash_map<std::string, std::shared_ptr<Table>> tables,
    const Options& options)
    : is_local_(true),
      tables_(std::move(tables)),
      options_(options),
      episode_id_(key_generator_.Generate()),
      episode_step_(0),
      closed_(false),
      stream_worker_(
          internal::StartThread("TrajectoryWriter_LocalWorker",
                                [this] { (void)RunLocalWorker(); })),
      stream_ok_(true) {
  REVERB_CHECK_OK(options.Validate());
  REVERB_CHECK(!tables_.empty());
}

TrajectoryWriter::TrajectoryWriter(shm::ShmConnection* conn,
                                   const Options& options)
    : shm_conn_(conn),
      options_(options),
      episode_id_(key_generator_.Generate()),
      episode_step_(0),
      closed_(false),
      stream_worker_(
          internal::StartThread("TrajectoryWriter_ShmWorker",
                                [this] { (void)RunShmWorker(); })),
      stream_ok_(true) {
  REVERB_CHECK_OK(options.Validate());
  REVERB_CHECK(shm_conn_ != nullptr);
}

TrajectoryWriter::~TrajectoryWriter() {
  {
    absl::MutexLock lock(mu_);
    if (closed_) return;

    absl::Status status = FlushLocked(/*ignore_last_num_items=*/0,
                                      /*timeout=*/absl::InfiniteDuration());
    REVERB_LOG_IF(REVERB_WARNING, !status.ok())
        << "TrajectoryWriter destroyed before content finalized. Encountered "
           "error when trying to finalize content: "
        << status;
  }
  Close();
}

absl::Status TrajectoryWriter::Append(
    std::vector<std::optional<TensorBuffer>> data,
    std::vector<std::optional<std::weak_ptr<CellRef>>>* refs) {
  return AppendInternal(std::move(data), /*increment_episode_step=*/true, refs);
}

absl::Status TrajectoryWriter::AppendPartial(
    std::vector<std::optional<TensorBuffer>> data,
    std::vector<std::optional<std::weak_ptr<CellRef>>>* refs) {
  return AppendInternal(std::move(data), /*increment_episode_step=*/false,
                        refs);
}

absl::Status TrajectoryWriter::AppendInternal(
    std::vector<std::optional<TensorBuffer>> data,
    bool increment_episode_step,
    std::vector<std::optional<std::weak_ptr<CellRef>>>* refs) {
  CellRef::EpisodeInfo episode_info;
  {
    absl::MutexLock lock(mu_);
    REVERB_RETURN_IF_ERROR(unrecoverable_status_);
    episode_info = {episode_id_, episode_step_};
  }

  // If this is the first time the column has been present in the data then
  // create a chunker using the spec of the item.
  for (int i = 0; i < data.size(); i++) {
    if (data[i].has_value() && !chunkers_.contains(i)) {
      const TensorBuffer& tensor = data[i].value();
      // If the new column has been configured with `ConfigureChunker` then we
      // use the overrided options. If not then we use the default in
      // `options_.chunker_options`.
      const std::shared_ptr<ChunkerOptions>& chunker_options =
          options_override_.contains(i) ? options_override_[i]
                                        : options_.chunker_options;
      chunkers_[i] = std::make_shared<Chunker>(
          internal::TensorSpec{std::to_string(i), tensor.dtype(),
                               tensor.shape()},
          chunker_options->Clone());
    }
  }

  // Append data to respective column chunker.
  for (int i = 0; i < data.size(); i++) {
    if (!data[i].has_value()) {
      refs->push_back(std::nullopt);
      continue;
    }

    std::weak_ptr<CellRef> ref;
    absl::Status status =
        chunkers_[i]->Append(data[i].value(), episode_info, &ref);
    if (absl::IsFailedPrecondition(status)) {
      return absl::FailedPreconditionError(
          "Append/AppendPartial called with data containing column that was "
          "present in previous AppendPartial call.");
    }
    REVERB_RETURN_IF_ERROR(status);
    refs->push_back(std::move(ref));
  }

  absl::MutexLock lock(mu_);

  // Sanity check that `Append`, `AppendPartial` or `EndEpisode` wasn't called
  // concurrently.
  REVERB_CHECK_EQ(episode_info.episode_id, episode_id_);
  REVERB_CHECK_EQ(episode_info.step, episode_step_);

  if (increment_episode_step) {
    episode_step_++;
  }

  // Wake up stream worker in case it was blocked on items referencing
  // incomplete chunks
  data_cv_.Signal();

  return absl::OkStatus();
}

absl::Status TrajectoryWriter::CreateItem(
    absl::string_view table, double priority,
    absl::Span<const TrajectoryColumn> trajectory) {
  if (trajectory.empty() ||
      std::all_of(trajectory.begin(), trajectory.end(),
                  [](const TrajectoryColumn& col) { return col.empty(); })) {
    return absl::InvalidArgumentError("trajectory must not be empty.");
  }
  if (std::isnan(priority)) {
    return absl::InvalidArgumentError("`priority` must not be nan.");
  }

  {
    absl::MutexLock lock(mu_);
    REVERB_RETURN_IF_ERROR(unrecoverable_status_);
  }

  auto item_and_refs = std::make_unique<ItemAndRefs>();

  // Lock all the references to ensure that the underlying data is not
  // deallocated before the worker has successfully written the item (and data)
  // to the gRPC stream.
  for (int col_idx = 0; col_idx < trajectory.size(); ++col_idx) {
    if (absl::Status status = trajectory[col_idx].Validate(); !status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Error in column ", col_idx, ": ", status.message()));
    }
    if (!trajectory[col_idx].LockReferences(&item_and_refs->refs)) {
      return absl::InternalError("CellRef unexpectedly expired in CreateItem.");
    }
  }

  item_and_refs->item.set_key(key_generator_.Generate());
  item_and_refs->item.set_table(table.data(), table.size());
  item_and_refs->item.set_priority(priority);

  for (const TrajectoryColumn& column : trajectory) {
    column.ToProto(
        item_and_refs->item.mutable_flat_trajectory()->add_columns());
  }

  REVERB_RETURN_IF_ERROR(item_and_refs->Validate(options_));

  {
    absl::MutexLock lock(mu_);
    write_queue_.push_back(std::move(item_and_refs));
  }

  return absl::OkStatus();
}

void TrajectoryWriter::Close() {
  {
    absl::MutexLock lock(mu_);
    if (closed_) return;

    // This will unblock the worker if it is waiting for new items to be sent.
    closed_ = true;

    // This will unblock the worker if it is blocked on a Write-call. It will
    // also unblock the reader thread as it definetely is blocked on a
    // Read-call.
    if (context_ != nullptr) {
      context_->TryCancel();
    }

    // This will unblock the worker if the front pending item is referencing
    // incomplete chunks and the worker is waiting for that to change.
    data_cv_.Signal();
  }

  // Join the worker thread.
  stream_worker_ = nullptr;
}

void TrajectoryWriter::OnReadDone(bool ok) {
  absl::MutexLock lock(mu_);
  data_cv_.Signal();
  if (!ok) {
    // BUGFIX: a failed gRPC read kills the stream but previously left both
    // unrecoverable_status_ and stream_status_ as OK, so FlushLocked's await
    // condition (which only checked unrecoverable_status_) never became true
    // -> permanent deadlock. Record the failure so FlushLocked can return a
    // real error and wake up.
    stream_ok_ = false;
    if (stream_status_.ok()) {
      stream_status_ = absl::InternalError("TrajectoryWriter stream read failed");
    }
    return;
  }
  for (uint64_t key : response_.keys()) {
    in_flight_items_.erase(key);
  }
  StartRead(&response_);
}

void TrajectoryWriter::OnWriteDone(bool ok) {
  absl::MutexLock lock(mu_);
  if (ok) {
    write_inflight_ = false;
  } else {
    // BUGFIX: same deadlock class as OnReadDone — a failed write kills the
    // stream but left no status behind for FlushLocked to return.
    stream_ok_ = false;
    if (stream_status_.ok()) {
      stream_status_ = absl::InternalError("TrajectoryWriter stream write failed");
    }
  }
}

void TrajectoryWriter::OnDone(const ::grpc::Status& s) {
  absl::MutexLock lock(mu_);
  stream_ok_ = false;
  stream_done_ = true;
  stream_status_ = FromGrpcStatus(s);
}

absl::Status TrajectoryWriter::Finish() {
  absl::MutexLock lock(mu_);
  // Release a hold from SetContextAndCreateStream.
  RemoveHold();
  auto trigger = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return stream_done_;
  };
  mu_.Await(absl::Condition(&trigger));
  return stream_status_;
}

absl::Status TrajectoryWriter::SetContextAndCreateStream() {
  absl::MutexLock lock(mu_);
  REVERB_RETURN_IF_ERROR(unrecoverable_status_);
  if (closed_) {
    return absl::CancelledError("TrajectoryWriter::Close has been called.");
  }
  context_ = std::make_unique<grpc::ClientContext>();
  context_->set_wait_for_ready(false);
  stub_->async()->InsertStream(context_.get(), this);
  stream_ok_ = true;
  stream_done_ = false;
  // Use a hold since some StartWrites are invoked indirectly rather than
  // directly from the reactor itself.
  AddHold();
  StartRead(&response_);
  StartCall();
  return absl::OkStatus();
}

bool TrajectoryWriter::WaitForPendingItems() {
  auto trigger = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return !write_queue_.empty() || closed_ || !stream_ok_;
  };
  mu_.Await(absl::Condition(&trigger));
  return !closed_ && stream_ok_;
}

internal::flat_hash_set<uint64_t> TrajectoryWriter::GetKeepKeys(
    const internal::flat_hash_set<uint64_t>& streamed_chunk_keys) const {
  internal::flat_hash_set<uint64_t> keys;
  for (const auto& it : chunkers_) {
    for (uint64_t key : it.second->GetKeepKeys()) {
      if (streamed_chunk_keys.contains(key)) {
        keys.insert(key);
      }
    }
  }

  for (auto it = write_queue_.begin(); it != write_queue_.end(); it++) {
    // Ignore chunks only referenced by the front item since keep keys is sent
    // together with this item and thus there is no need for the server to keep
    // these chunks around after the item has been written.
    if (it == write_queue_.begin()) continue;

    for (const std::shared_ptr<CellRef>& ref : (*it)->refs) {
      if (streamed_chunk_keys.contains(ref->chunk_key())) {
        keys.insert(ref->chunk_key());
      }
    }
  }

  return keys;
}

absl::Status TrajectoryWriter::RunLocalWorker() {
  while (true) {
    ItemAndRefs* item_and_refs = nullptr;
    {
      absl::MutexLock l(&mu_);
      while (write_queue_.empty() && !closed_ && stream_ok_) {
        data_cv_.Wait(&mu_);
      }
      if (!stream_ok_) {
        return stream_status_;
      }
      if (closed_ && write_queue_.empty()) {
        return absl::OkStatus();
      }
      item_and_refs = write_queue_.front().get();
    }

    // Wait until all chunks referenced by the item have been finalized.
    // This mirrors the ContainsAll/AllReady dance in `RunStreamWorker`: if a
    // referenced chunk is incomplete, flush its chunker and wait for
    // `data_cv_` to be signalled when the chunk becomes ready.
    if (!AllReady(item_and_refs->refs)) {
      for (const std::shared_ptr<CellRef>& ref : item_and_refs->refs) {
        if (!ref->IsReady()) {
          auto chunker_sp = ref->chunker().lock();
          if (chunker_sp) {
            absl::Status s = chunker_sp->Flush();
            if (!s.ok()) {
              absl::MutexLock l(&mu_);
              stream_ok_ = false;
              stream_status_ = s;
              unrecoverable_status_ = s;
              data_cv_.Signal();
              return s;
            }
          }
        }
      }
      absl::MutexLock l(&mu_);
      // Re-check now that we hold the lock; the chunk may already be ready.
      if (!AllReady(item_and_refs->refs)) {
        data_cv_.Wait(&mu_);
      }
      continue;
    }

    // Notify chunkers so they can adapt (mirrors the gRPC path). This MUST run
    // before the item is handed to the table / moved to `in_flight_items_`, as
    // the insert-completion callback (fired asynchronously by the table
    // worker) erases the item and would otherwise free it under us.
    internal::flat_hash_map<Chunker*, std::vector<std::shared_ptr<CellRef>>>
        refs_per_chunker;
    for (auto& ref : item_and_refs->refs) {
      auto chunker_sp = ref->chunker().lock();
      if (!chunker_sp) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Chunker::OnItemFinalized: Unable to lock the weak_ptr for the "
            "chunker associated with chunk_key: ",
            ref->chunk_key()));
      }
      refs_per_chunker[chunker_sp.get()].push_back(ref);
    }
    for (auto& [chunker, refs] : refs_per_chunker) {
      absl::Status status =
          chunker->OnItemFinalized(item_and_refs->item, refs);
      if (!status.ok()) {
        absl::MutexLock l(&mu_);
        stream_ok_ = false;
        stream_status_ = status;
        unrecoverable_status_ = status;
        data_cv_.Signal();
        return status;
      }
    }

    // Dispatch by item.table(): look up the target table in the writer's map.
    // An unknown table yields kNotFound (previously the single-table binding
    // silently ignored the table field).
    const std::string& table_name = item_and_refs->item.table();
    auto table_it = tables_.find(table_name);
    if (table_it == tables_.end()) {
      absl::MutexLock l(&mu_);
      stream_ok_ = false;
      stream_status_ = absl::NotFoundError(absl::StrCat(
          "TrajectoryWriter::RunLocalWorker: table '", table_name,
          "' not found."));
      unrecoverable_status_ = stream_status_;
      data_cv_.Signal();
      return stream_status_;
    }
    Table* target_table = table_it->second.get();

    // Assemble the list of referenced chunks, deduplicating by chunk key.
    std::vector<std::shared_ptr<ChunkStore::Chunk>> chunks;
    internal::flat_hash_set<uint64_t> sent_keys;
    for (const std::shared_ptr<CellRef>& ref : item_and_refs->refs) {
      uint64_t ck = ref->chunk_key();
      if (sent_keys.insert(ck).second) {
        auto chunk_container = ref->GetChunk();
        // `ChunkStore::Chunk` owns a copy of the `ChunkData` proto.
        chunks.push_back(
            std::make_shared<ChunkStore::Chunk>(*chunk_container->get()));
      }
    }

    TableItem table_item(item_and_refs->item, std::move(chunks));
    uint64_t key = item_and_refs->item.key();

    // Backpressure: `InsertOrAssignAsync` reports via `can_insert_more`
    // whether another insert can be queued right away. When it can't, we wait
    // for an outstanding insert to complete (signalled by the callback, which
    // also clears `local_can_insert_more_`).
    //
    // The callback is stored on the `ItemAndRefs` so the `weak_ptr` handed to
    // the table stays alive until the insert actually completes; otherwise the
    // table would drop a dead callback and never confirm the item.
    bool can_insert_more = false;
    item_and_refs->insert_callback =
        std::make_shared<Table::InsertCallback>(
            [this](uint64_t completed_key) {
              absl::MutexLock l(&mu_);
              in_flight_items_.erase(completed_key);
              local_can_insert_more_ = true;
              data_cv_.Signal();
            });

    // Move the item from `write_queue_` to `in_flight_items_` BEFORE calling
    // `InsertOrAssignAsync`. The table worker runs inserts on its own thread
    // holding only `worker_mu_`, not `mu_`, so the completion callback can
    // fire before `InsertOrAssignAsync` returns. If the item were registered
    // in `in_flight_items_` only after the call (as the original code did),
    // the callback's `erase` would miss and the item would leak, while
    // `local_can_insert_more_` set by the callback would then be clobbered
    // to `false` below, deadlocking the backpressure wait. Registering first
    // guarantees the callback's `erase` always hits.
    {
      absl::MutexLock l(&mu_);
      in_flight_items_[key] = std::move(write_queue_.front());
      write_queue_.pop_front();
    }

    absl::Status s = target_table->InsertOrAssignAsync(
        std::move(table_item), &can_insert_more, item_and_refs->insert_callback);
    if (!s.ok()) {
      absl::MutexLock l(&mu_);
      in_flight_items_.erase(key);
      stream_ok_ = false;
      stream_status_ = s;
      unrecoverable_status_ = s;
      data_cv_.Signal();
      return s;
    }

    // Apply backpressure: if the table's insert queue is full, wait for an
    // outstanding insert to complete before pulling another item. The
    // callback may already have fired (setting `local_can_insert_more_ =
    // true`) by the time we reacquire `mu_`; in that case skip the wait. We
    // preserve the callback's signal by only clearing the flag when we
    // actually intend to block.
    if (!can_insert_more) {
      absl::MutexLock l(&mu_);
      if (!local_can_insert_more_) {
        local_can_insert_more_ = false;
        while (!local_can_insert_more_ && !closed_ && stream_ok_) {
          data_cv_.Wait(&mu_);
        }
      }
    }
    {
      absl::MutexLock l(&mu_);
      if (in_flight_items_.size() + write_queue_.size() >=
          kPendingItemsWarningThreshold) {
        REVERB_LOG_EVERY_N(REVERB_WARNING, 10) << absl::StrFormat(
            "The number of pending items is alarmingly high, did you forget "
            "to call Flush? %d items are waiting to be inserted and %d items "
            "have been queued but haven't been confirmed yet.",
            write_queue_.size(), in_flight_items_.size());
      }
    }
  }
}

absl::Status TrajectoryWriter::RunShmWorker() {
  // ponytail: this is RunLocalWorker with the InsertOrAssignAsync step swapped
  // for an SHM round-trip (ALLOCATE per chunk → memcpy → INSERT → wait
  // INSERT_ACK → RELEASE offsets). All chunker/column/backpressure machinery
  // is identical; only the transport changes (appendix A4 / decision C2).
  //
  // ticket 01 (shm-batch-insert): the round-trip is BATCHED. The worker waits
  // for the first ready item as before, then opportunistically gathers up to
  // kMaxInsertBatchItems-1 further ALREADY-ready items behind it (no chunker
  // Flush, no waiting — an extra is taken only if AllReady when gathered) and
  // merges them into ONE ShmInsertRequest. Every unique chunk of the batch is
  // ALLOCATEd in a pipelined burst (N requests written back-to-back, N RESPs
  // read in FIFO order), then one INSERT is sent and ONE aggregate INSERT_ACK
  // confirms the whole batch: the worker erases all batched items from
  // in_flight_items_ and RELEASEs the ACK's offsets_to_release. A batch costs
  // ~2 ring round-trips regardless of size (was: 2 per item — the per-item
  // ACK wait pinned SHM insert throughput at ~1/RTT, client-benchmark §3).
  //
  // Decision D: the insert flow uses its OWN ring pair (insert_c2s /
  // insert_s2c), separate from the sampler's pair, so this worker thread and
  // ShmSampler's worker thread never contend as producers/consumers on one
  // SPSC ring. The SPSC invariant (single producer per ring) is restored.
  //
  // The ACK IS the completion signal: on ACK the worker erases the batch from
  // in_flight_items_, sets local_can_insert_more_, signals data_cv_, and
  // RELEASEs the chunk offsets. There is no async table callback in SHM mode
  // (the server's table callback writes the ACK; the client side is a
  // synchronous poll).
  //
  // The SHM proto types (Ring, MsgType, ShmInsertRequest, ...) live in
  // deepmind::reverb::shm; bring them into scope locally rather than
  // qualifying every use. ponytail: function-local using-directive keeps the
  // blast radius to this worker only.
  using namespace ::deepmind::reverb::shm;

  // ticket ⑩ 死锁修复（方向 C）：加有限超时，避免服务端 dispatch 被卡时
  // RunShmWorker 无限忙等 insert ACK。方向 A 修了 sample 路径根因，但 insert
  // ACK 仍可能因 ring 满 / outbox 堆积而延迟；硬上限作健壮性兜底。超时后
  // 返 DeadlineExceededError，Flush/EndEpisode 会把它作为 unrecoverable_status_
  // 表面给调用者。ponytail: 60s 是宽松上限——正常 ACK 应在 ms 级返回。
  constexpr absl::Duration kInsertAckTimeout = absl::Seconds(60);

  // ticket 01: batch bounds. kMaxInsertBatchItems caps the item count; the
  // pipelined ALLOCATE burst is bounded by the unique-chunk count, which
  // stays ≪ insert_s2c's 1024-slot ring, so RESP bursts never stash into the
  // outbox mid-burst (see EnqueueInsertS2C's FIFO guard). kMaxInsertBatchBytes
  // caps the summed PrioritizedItem proto bytes so the INSERT message always
  // fits the c2s ring (~245KB of slot bodies; tensor bytes ride the pool, not
  // the ring — a PrioritizedItem is table/key/priority + chunk-key refs).
  // A single item larger than the byte cap goes alone, matching the pre-batch
  // behavior for oversize items.
  constexpr size_t kMaxInsertBatchItems = 64;
  constexpr size_t kMaxInsertBatchBytes = 128 * 1024;

  auto read_blocking = [](Ring* ring, MsgType* type,
                          std::string* payload,
                          const ShmConnection* conn,
                          absl::Duration timeout) -> absl::Status {
    // poll non-blocking Read + sched_yield (spec R5: blocking policy is the
    // caller's job, not Ring's). Same helper as ShmSampler/ShmClient.
    // ticket ⑥ (spec §8.8): if the liveness fd (control_fd) shows EOF/HUP the
    // server is gone — return UnavailableError so in-flight inserts fail fast
    // instead of spinning forever on a dead server.
    // ticket「shm-close-while-in-flight」: also bail on the connection's
    // closed flag — after a racing Close() the fd is -1/recycled and the fd
    // probe is skipped, which used to hang this loop indefinitely.
    absl::Time deadline = absl::Now() + timeout;
    while (true) {
      absl::Status s = ring->Read(type, payload);
      if (s.ok()) return absl::OkStatus();
      if (!absl::IsNotFound(s)) return s;
      if (conn->closed.load(std::memory_order_acquire) ||
          (conn->control_fd >= 0 && IsPeerClosed(conn->control_fd))) {
        return absl::UnavailableError("SHM server closed connection");
      }
      if (absl::Now() >= deadline) {
        return absl::DeadlineExceededError(
            "RunShmWorker: insert response timed out");
      }
      sched_yield();
    }
  };

  while (true) {
    ItemAndRefs* item_and_refs = nullptr;
    {
      absl::MutexLock l(&mu_);
      while (write_queue_.empty() && !closed_ && stream_ok_) {
        data_cv_.Wait(&mu_);
      }
      if (!stream_ok_) {
        return stream_status_;
      }
      if (closed_ && write_queue_.empty()) {
        return absl::OkStatus();
      }
      item_and_refs = write_queue_.front().get();
    }

    // Wait until all referenced chunks are finalized (mirrors RunLocalWorker).
    if (!AllReady(item_and_refs->refs)) {
      for (const std::shared_ptr<CellRef>& ref : item_and_refs->refs) {
        if (!ref->IsReady()) {
          auto chunker_sp = ref->chunker().lock();
          if (chunker_sp) {
            absl::Status s = chunker_sp->Flush();
            if (!s.ok()) {
              absl::MutexLock l(&mu_);
              stream_ok_ = false;
              stream_status_ = s;
              unrecoverable_status_ = s;
              data_cv_.Signal();
              return s;
            }
          }
        }
      }
      absl::MutexLock l(&mu_);
      if (!AllReady(item_and_refs->refs)) {
        data_cv_.Wait(&mu_);
      }
      continue;
    }

    // ticket ⑩+01: hold insert_flow_mu across the WHOLE batch operation —
    // every ALLOCATE→ALLOCATE_RESP and the INSERT→INSERT_ACK round-trip — so
    // MutatePriorities/Reset (caller thread, same conn) never race this
    // worker as a second producer on insert_c2s. RunShmWorker is strictly
    // synchronous at batch granularity (one batch round-trip at a time), so
    // one lock per batch is the natural granularity; the mutex is uncontended
    // except when a control-plane call overlaps an in-flight insert. See
    // ShmConnection::insert_flow_mu. The sample flow is untouched (ShmSampler
    // writes sample_c2s).
    //
    // The lock is taken BEFORE the batch gather: the batch is composed at the
    // latest possible moment, so items queued while the previous round-trip
    // (or a control-plane call) was in flight join this batch. Lock order is
    // insert_flow_mu → mu_; nothing takes them in the reverse order
    // (control-plane ops never touch mu_).
    absl::MutexLock insert_flow_lock(&shm_conn_->insert_flow_mu);

    // ticket 01: gather the batch — the known-ready front item plus every
    // consecutive ALREADY-ready item behind it, bounded by the count and byte
    // caps. Extras are NOT chunker-Flushed and NOT waited on (凑批不引入等待);
    // the scan stops at the first not-ready item, preserving FIFO order.
    std::vector<ItemAndRefs*> batch;
    {
      absl::MutexLock l(&mu_);
      size_t batch_bytes = 0;
      for (const std::unique_ptr<ItemAndRefs>& queued : write_queue_) {
        if (!AllReady(queued->refs)) break;
        size_t item_bytes = queued->item.ByteSizeLong();
        if (!batch.empty() &&
            (batch.size() >= kMaxInsertBatchItems ||
             batch_bytes + item_bytes > kMaxInsertBatchBytes)) {
          break;
        }
        batch.push_back(queued.get());
        batch_bytes += item_bytes;
      }
    }

    // Notify chunkers for every batched item (mirrors RunLocalWorker / gRPC
    // path). This MUST run before the items are moved to `in_flight_items_`
    // (see RunLocalWorker for the callback-erases-item rationale).
    for (const ItemAndRefs* item : batch) {
      internal::flat_hash_map<Chunker*, std::vector<std::shared_ptr<CellRef>>>
          refs_per_chunker;
      for (auto& ref : item->refs) {
        auto chunker_sp = ref->chunker().lock();
        if (!chunker_sp) {
          return absl::FailedPreconditionError(absl::StrCat(
              "Chunker::OnItemFinalized: Unable to lock the weak_ptr for the "
              "chunker associated with chunk_key: ",
              ref->chunk_key()));
        }
        refs_per_chunker[chunker_sp.get()].push_back(ref);
      }
      for (auto& [chunker, refs] : refs_per_chunker) {
        absl::Status status = chunker->OnItemFinalized(item->item, refs);
        if (!status.ok()) {
          absl::MutexLock l(&mu_);
          stream_ok_ = false;
          stream_status_ = status;
          unrecoverable_status_ = status;
          data_cv_.Signal();
          return status;
        }
      }
    }

    // Shared fail tails. release_offsets: RELEASE pool offsets (C3: otherwise
    // they leak until the server reclaims them at disconnect). erase_batch:
    // drop the whole batch from in_flight_items_. fail_stream: kill the
    // stream and wake Flush/EndEpisode waiters. ANY failure fails the WHOLE
    // batch as a unit, exactly like the single-item path.
    auto release_offsets = [&](absl::Span<const uint64_t> offsets)
        -> absl::Status {
      if (offsets.empty()) return absl::OkStatus();
      ShmReleaseRequest rel;
      for (uint64_t off : offsets) rel.add_offsets(off);
      std::string rel_body;
      rel.SerializeToString(&rel_body);
      return WriteBlocking(&shm_conn_->insert_c2s, RELEASE,
                           absl::MakeSpan(rel_body), shm_conn_->control_fd);
    };
    auto erase_batch = [&] {
      absl::MutexLock l(&mu_);
      for (const ItemAndRefs* item : batch) {
        in_flight_items_.erase(item->item.key());
      }
    };
    auto fail_stream = [&](const absl::Status& s) {
      absl::MutexLock l(&mu_);
      stream_ok_ = false;
      stream_status_ = s;
      unrecoverable_status_ = s;
      data_cv_.Signal();
    };

    // Assemble the unique referenced chunks across the WHOLE batch,
    // deduplicating by chunk key: a chunk shared by several batched items is
    // ALLOCATEd, serialized and referenced exactly once. The container keeps
    // the ChunkData alive until it is serialized into the pool.
    struct BatchChunk {
      uint64_t key;
      std::shared_ptr<ChunkDataContainer> container;
      size_t num_bytes;
      uint64_t offset;  // granted by ALLOCATE_RESP
    };
    std::vector<BatchChunk> batch_chunks;
    std::vector<uint64_t> chunk_offsets;  // granted, for RELEASE (C2/C3)
    {
      internal::flat_hash_set<uint64_t> sent_keys;
      for (const ItemAndRefs* item : batch) {
        for (const std::shared_ptr<CellRef>& ref : item->refs) {
          uint64_t ck = ref->chunk_key();
          if (!sent_keys.insert(ck).second) continue;
          auto container = ref->GetChunk();
          // ponytail: 先算尺寸,ALLOCATE 后 SerializeToArray 直写 pool,省掉
          // SerializeToString 中间 string + 一次 memcpy。
          size_t num_bytes = container->get()->ByteSizeLong();
          batch_chunks.push_back({ck, std::move(container), num_bytes, 0});
        }
      }
    }

    // Phase 1 (C4): ask the server (sole allocator) for one pool offset per
    // unique chunk — PIPELINED (ticket 01): all ALLOCATEs are written
    // back-to-back, then one RESP is read per sent ALLOCATE. The server
    // drains insert_c2s in FIFO order on its single dispatch thread and its
    // responses stay in order (EnqueueInsertS2C), so RESP i answers ALLOCATE
    // i. This is what makes a batch cost ~2 ring round-trips instead of
    // 2 × items.
    absl::Status batch_error;
    size_t allocs_sent = 0;
    for (const BatchChunk& bc : batch_chunks) {
      ShmAllocateRequest areq;
      areq.set_num_bytes(bc.num_bytes);
      std::string areq_body;
      areq.SerializeToString(&areq_body);
      absl::Status ws = WriteBlocking(&shm_conn_->insert_c2s, ALLOCATE,
                                      absl::MakeSpan(areq_body),
                                      shm_conn_->control_fd);
      if (!ws.ok()) {
        batch_error = ws;
        break;
      }
      allocs_sent++;
    }
    // Drain the RESPs. A transport-level read failure STOPS the drain (later
    // RESPs may never arrive; offsets granted in them are unknown and get
    // reclaimed by the server at disconnect — same as the single-item path).
    // A per-request ERROR (e.g. pool exhausted) does NOT stop the drain:
    // every granted offset must be collected so it can be released below.
    for (size_t i = 0; i < allocs_sent; ++i) {
      MsgType atype;
      std::string aresp_body;
      absl::Status rs = read_blocking(&shm_conn_->insert_s2c, &atype,
                                      &aresp_body, shm_conn_,
                                      kInsertAckTimeout);
      if (!rs.ok()) {
        if (batch_error.ok()) batch_error = rs;
        break;
      }
      if (atype == ERROR) {
        // The server rejected the allocation (e.g. pool exhausted). Surface
        // the ShmError's real status instead of a generic type mismatch.
        if (batch_error.ok()) {
          ShmError err;
          if (err.ParseFromString(aresp_body) &&
              err.code() == ShmError::RESOURCE_EXHAUSTED) {
            batch_error = absl::ResourceExhaustedError(err.message());
          } else if (err.ParseFromString(aresp_body) &&
                     err.code() == ShmError::INVALID_ARGUMENT) {
            batch_error = absl::InvalidArgumentError(err.message());
          } else {
            batch_error = absl::InternalError(absl::StrCat(
                "RunShmWorker: ALLOCATE rejected: ", aresp_body));
          }
        }
        continue;
      }
      if (atype != ALLOCATE_RESP) {
        if (batch_error.ok()) {
          batch_error = absl::InternalError(absl::StrCat(
              "RunShmWorker: expected ALLOCATE_RESP, got ", atype));
        }
        continue;
      }
      ShmAllocateResponse aresp;
      if (!aresp.ParseFromString(aresp_body)) {
        if (batch_error.ok()) {
          batch_error = absl::InternalError(
              "RunShmWorker: malformed ShmAllocateResponse");
        }
        continue;
      }
      uint64_t offset = aresp.shm_offset();
      // Bounds-check the granted region before memcpy: BytePool::At is raw
      // pointer arithmetic, so a buggy/corrupt ALLOCATE_RESP would otherwise
      // make us write outside our OWN RW mapping and corrupt this process.
      if (offset > shm_conn_->pool.size() ||
          batch_chunks[i].num_bytes > shm_conn_->pool.size() - offset) {
        if (batch_error.ok()) {
          batch_error = absl::InternalError(absl::StrCat(
              "RunShmWorker: ALLOCATE_RESP granted offset ", offset, " (len ",
              batch_chunks[i].num_bytes, ") outside mapped pool of ",
              shm_conn_->pool.size(), " bytes"));
        }
        continue;
      }
      batch_chunks[i].offset = offset;
      chunk_offsets.push_back(offset);
    }
    if (!batch_error.ok()) {
      // Release any offsets already granted so the pool doesn't leak (C3).
      (void)release_offsets(chunk_offsets);
      fail_stream(batch_error);
      return batch_error;
    }

    // Phase 2: serialize each ChunkData straight into its granted region
    // (client RW mmap, C4). The region must stay valid until INSERT_ACK (C2).
    // ByteSizeLong and SerializeToArray walk the proto twice but share no
    // intermediate buffer; failure (same source for the size, so only if the
    // proto is malformed) releases every granted offset.
    for (const BatchChunk& bc : batch_chunks) {
      if (!bc.container->get()->SerializeToArray(
              shm_conn_->pool.At(bc.offset), static_cast<int>(bc.num_bytes))) {
        (void)release_offsets(chunk_offsets);
        absl::Status s = absl::InternalError(absl::StrCat(
            "RunShmWorker: failed to serialize ChunkData ", bc.key));
        fail_stream(s);
        return s;
      }
    }

    // Phase 3: ONE ShmInsertRequest carrying the whole batch.
    //
    // ponytail: ShmChunkRef.specs/sequence_range/delta_encoded are redundant —
    // ChunkData is self-describing and the server deserializes it whole. We
    // populate only chunk_key/shm_offset/total_length. Upgrade: populate the
    // metadata if the server ever skips deserialization for the fast path.
    ShmInsertRequest req;
    for (const BatchChunk& bc : batch_chunks) {
      ShmChunkRef* cref = req.add_chunks();
      cref->set_chunk_key(bc.key);
      cref->set_shm_offset(bc.offset);
      cref->set_total_length(bc.num_bytes);
    }
    for (const ItemAndRefs* item : batch) {
      // The server routes each item to its table by item.table() (ticket ⑨).
      *req.add_items() = item->item;
    }

    // Move the WHOLE batch to in_flight BEFORE sending INSERT. The server's
    // aggregate ACK is the completion signal for every item in the batch; on
    // ACK they are all erased and the offsets released. Registering first
    // keeps Flush's (write_queue_ + in_flight_items_) accounting exact: a
    // batched item counts as in-flight until the batch ACK lands, so
    // Flush/EndEpisode can never return before their batch is confirmed. The
    // batch items are the front of write_queue_; the raw `batch` pointers
    // stay valid (the unique_ptr moves only transfer ownership).
    {
      absl::MutexLock l(&mu_);
      for (const ItemAndRefs* item : batch) {
        in_flight_items_[item->item.key()] =
            std::move(write_queue_.front());
        write_queue_.pop_front();
      }
    }

    // Send INSERT (blocks if the C→S ring is full — natural backpressure).
    std::string req_body;
    req.SerializeToString(&req_body);
    absl::Status is = WriteBlocking(&shm_conn_->insert_c2s, INSERT, absl::MakeSpan(req_body), shm_conn_->control_fd);
    if (!is.ok()) {
      erase_batch();
      // Release the offsets we allocated; the INSERT never went.
      (void)release_offsets(chunk_offsets);
      fail_stream(is);
      return is;
    }

    // Wait for the aggregate INSERT_ACK (C2: client must not reuse offsets
    // until ACK's offsets_to_release arrives). The ACK is the completion
    // signal for the WHOLE batch; the 60s cap covers one batch round-trip —
    // on timeout the batch fails as a unit into unrecoverable_status_.
    MsgType ack_type;
    std::string ack_body;
    absl::Status as = read_blocking(&shm_conn_->insert_s2c, &ack_type, &ack_body,
                                     shm_conn_, kInsertAckTimeout);
    if (!as.ok()) {
      erase_batch();
      fail_stream(as);
      return as;
    }
    if (ack_type == ERROR) {
      // The server rejected the WHOLE request (e.g. an unknown table on any
      // batched item, or a chunk it couldn't place). Its chunk bytes were
      // already copied out of the pool (or never read), so release our
      // offsets to avoid a pool leak and fail the batch as a unit.
      (void)release_offsets(chunk_offsets);
      erase_batch();
      // ticket ⑨: map ShmError codes to the same absl statuses as FetchOne so
      // an unknown-table insert surfaces as NotFoundError (not a generic
      // InternalError). Falls back to InternalError for unknown codes.
      absl::Status mapped;
      ShmError err;
      if (err.ParseFromString(ack_body)) {
        switch (err.code()) {
          case ShmError::NOT_FOUND:
            mapped = absl::NotFoundError(err.message());
            break;
          case ShmError::INVALID_ARGUMENT:
            mapped = absl::InvalidArgumentError(err.message());
            break;
          default:
            mapped = absl::InternalError(absl::StrCat(
                "RunShmWorker: server error: ", err.message()));
            break;
        }
      } else {
        mapped = absl::InternalError(
            "RunShmWorker: server returned ERROR for INSERT");
      }
      fail_stream(mapped);
      return mapped;
    }
    if (ack_type != INSERT_ACK) {
      erase_batch();
      absl::Status s = absl::InternalError(absl::StrCat(
          "RunShmWorker: expected INSERT_ACK, got ", ack_type));
      fail_stream(s);
      return s;
    }
    InsertAck ack;
    if (!ack.ParseFromString(ack_body)) {
      erase_batch();
      absl::Status s =
          absl::InternalError("RunShmWorker: malformed InsertAck");
      fail_stream(s);
      return s;
    }

    // Verify the server confirmed EVERY batched item's key. The aggregate ACK
    // carries one key per completed item; a missing key means the server
    // rejected that item (e.g. unknown table) — fail the batch as a unit.
    bool all_confirmed = true;
    {
      internal::flat_hash_set<uint64_t> acked(ack.keys().begin(),
                                              ack.keys().end());
      for (const ItemAndRefs* item : batch) {
        if (!acked.contains(item->item.key())) {
          all_confirmed = false;
          break;
        }
      }
    }
    if (!all_confirmed) {
      // Release offsets (C2) and surface.
      (void)release_offsets(chunk_offsets);
      erase_batch();
      absl::Status s = absl::NotFoundError(
          "RunShmWorker: server did not confirm all batched items "
          "(unknown table or rejected)");
      fail_stream(s);
      return s;
    }

    // Completion: erase the whole batch from in_flight, allow more inserts,
    // signal waiters. ticket 01: in_flight_items_ now peaks at the batch size
    // (≤ kMaxInsertBatchItems) instead of 1. local_can_insert_more_ is set
    // here but NEVER read/awaited by RunShmWorker (unlike RunLocalWorker,
    // which waits on it while false). It is vestigial from RunLocalWorker and
    // NOT a backpressure gate. Upgrade: async pipelined batches reusing this
    // as the in_flight>1 gate.
    {
      absl::MutexLock l(&mu_);
      for (const ItemAndRefs* item : batch) {
        in_flight_items_.erase(item->item.key());
      }
      local_can_insert_more_ = true;
      data_cv_.Signal();
    }

    // C2: the server has copied the batch's bytes into its ChunkStore; the
    // ACK's offsets_to_release covers every chunk of the batch — release them
    // all in one message.
    std::vector<uint64_t> ack_offsets(ack.offsets_to_release().begin(),
                                      ack.offsets_to_release().end());
    absl::Status rs = release_offsets(ack_offsets);
    if (!rs.ok()) {
      // Non-fatal for data integrity (the server tracks outstanding offsets
      // and reclaims on disconnect, ⑥); but log via unrecoverable_status_ so
      // it surfaces. ponytail: a stuck RELEASE would eventually exhaust the
      // pool tier; surface it.
      fail_stream(rs);
      return rs;
    }

    {
      absl::MutexLock l(&mu_);
      if (in_flight_items_.size() + write_queue_.size() >=
          kPendingItemsWarningThreshold) {
        REVERB_LOG_EVERY_N(REVERB_WARNING, 10) << absl::StrFormat(
            "The number of pending items is alarmingly high, did you forget "
            "to call Flush? %d items are waiting to be inserted and %d items "
            "have been queued but haven't been confirmed yet.",
            write_queue_.size(), in_flight_items_.size());
      }
    }
  }
}

absl::Status TrajectoryWriter::RunStreamWorker() {
  REVERB_RETURN_IF_ERROR(SetContextAndCreateStream());
  internal::flat_hash_set<uint64_t> streamed_chunk_keys;
  ArenaOwnedRequest request;

  // How many more items to add to the current request. When a new request is
  // started this value is set to the number of currently pending items, so that
  // all of them are written in one go, but items enqueued in the meantime are
  // not.
  int add_items_to_batch = 0;
  while (true) {
    ItemAndRefs* item_and_refs;
    {
      absl::WriterMutexLock lock(mu_);
      if (!WaitForPendingItems()) {
        break;
      }
      if (add_items_to_batch == 0) {
        add_items_to_batch = write_queue_.size();
      }
      item_and_refs = write_queue_.front().get();
    }

    // Send referenced chunks which haven't already been sent. This call also
    // inserts the new chunk keys into `streamed_chunk_keys`.
    if (!SendNotAlreadySentChunks(&streamed_chunk_keys, item_and_refs->refs,
                                  &request)) {
      return Finish();
    }

    // Check whether all chunks referenced by the item have been written to
    // the stream. If not, then at least one chunk is incomplete and the
    // worker will wait for the chunk state to change and then retry.
    if (!ContainsAll(streamed_chunk_keys, item_and_refs->refs)) {
      // Before going to sleep send ready items for better pipelining.
      if (!WriteIfNotEmpty(streamed_chunk_keys, &request)) {
        return Finish();
      }
      absl::WriterMutexLock lock(mu_);
      // Do a final check that the chunks didn't change since the lock was
      // last held. If the item still references incomplete chunks then we
      // sleep until the chunks changed. If all the chunks are now completed
      // then we move straight to the top of the loop.
      if (!AllReady(item_and_refs->refs)) {
        data_cv_.Wait(&mu_);
      }
      continue;
    }
    {
      absl::WriterMutexLock lock(mu_);
      // Item is about to be written - move from write_queue_ to
      // in_flight_items_.
      in_flight_items_[item_and_refs->item.key()] =
          std::move(write_queue_.front());
      write_queue_.pop_front();

      // Check if the number if the total number of pending items is very large
      // and if so log a warning message.
      if (in_flight_items_.size() + write_queue_.size() >=
          kPendingItemsWarningThreshold) {
        REVERB_LOG_EVERY_N(REVERB_WARNING, 10) << absl::StrFormat(
            "The number of pending items is alarmingly high, did you forget "
            "to call Flush? %d items are waiting to be sent and %d items "
            "have been sent to the server but haven't been confirmed yet. It "
            "is important to call Flush regularly as large numbers of pending "
            "items can result in OOM crashes on both client and server.",
            write_queue_.size(), in_flight_items_.size());
      }

      // Remove keys of expired chunks from streamed_chunk_keys to avoid OOM
      // issues caused by the otherwise indefinitely growing hash set.
      streamed_chunk_keys = GetKeepKeys(streamed_chunk_keys);
    }

    // Group the item references by chunker and pass it to their respective
    // chunker to allow it to adapt to the data.
    internal::flat_hash_map<Chunker*, std::vector<std::shared_ptr<CellRef>>>
        refs_per_chunker;
    for (auto& ref : item_and_refs->refs) {
      auto chunker_sp = ref->chunker().lock();
      if (!chunker_sp) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Chunker::OnItemFinalized: Unable to lock the weak_ptr for the "
            "chunker associated with chunk_key: ",
            ref->chunk_key()));
      }
      refs_per_chunker[chunker_sp.get()].push_back(ref);
    }

    for (auto& [chunker, refs] : refs_per_chunker) {
      REVERB_RETURN_IF_ERROR(
          chunker->OnItemFinalized(item_and_refs->item, refs));
    }

    // All chunks have been written to the stream so the item can now be
    // added to the request.
    request.AddItem(item_and_refs->item);

    if (--add_items_to_batch == 0) {
      if (!WriteIfNotEmpty(streamed_chunk_keys, &request)) {
        return Finish();
      }
    }
  }
  return Finish();
}

absl::Status TrajectoryWriter::Flush(int ignore_last_num_items,
                                     absl::Duration timeout) {
  absl::MutexLock lock(mu_);
  return FlushLocked(ignore_last_num_items, timeout);
}

absl::Status TrajectoryWriter::FlushLocked(int ignore_last_num_items,
                                           absl::Duration timeout) {
  // If items are referencing any data which has not yet been finalized into a
  // `ChunkData` then force the chunk to be created prematurely. This will allow
  // the worker to write all items to the stream. Note that we don't need to
  // force the finalization of the `ignore_last_num_items` last items in the
  // queue.
  int num_items_to_force_flush = write_queue_.size() - ignore_last_num_items;
  for (const auto& item : write_queue_) {
    if (num_items_to_force_flush-- <= 0) break;

    for (const std::shared_ptr<CellRef>& ref : item->refs) {
      if (!ref->IsReady()) {
        REVERB_RETURN_IF_ERROR(ref->chunker().lock()->Flush());
      }
    }
  }

  // Since all the (referenced) data have been finalized into chunks the worker
  // can be woken up.
  data_cv_.Signal();

  // The write worker is now able to send  (at least) all but the last
  // `ignore_last_num_items` items to the server. We release the mutex and wait
  // for the items to be confirmed or the TrajectoryWriter to be closed.
  //
  // We break out only when the worker reaches a terminal verdict
  // (unrecoverable_status_ set) or all items are confirmed. We deliberately
  // do NOT break on `!stream_ok_` alone: a transient gRPC stream failure
  // (OnWriteDone/OnReadDone with ok=false, or OnDone) sets stream_ok_=false
  // while the worker is still alive and about to retry on a new stream
  // (SetContextAndCreateStream resets stream_ok_=true). Breaking here would
  // surface the premature per-event stream_status_ (e.g. INTERNAL "stream
  // write failed") before the retry runs, making Flush return a
  // non-retryable-looking error for a transient failure. Every worker death
  // path (gRPC worker loop at the thread exit, RunLocalWorker, RunShmWorker)
  // sets unrecoverable_status_ before exiting, so waiting on it is sufficient
  // to avoid the worker-dead deadlock; the AwaitWithTimeout bounds the wait.
  auto cond = [ignore_last_num_items, this]()
                  ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) -> bool {
    if (!unrecoverable_status_.ok()) {
      return true;
    }
    return write_queue_.size() + in_flight_items_.size() <=
           ignore_last_num_items;
  };

  if (!mu_.AwaitWithTimeout(absl::Condition(&cond), timeout)) {
    return absl::DeadlineExceededError(
        absl::StrCat("Timeout exceeded with ", write_queue_.size(),
                     " items waiting to be written and ",
                     in_flight_items_.size(), " items awaiting confirmation."));
  }

  // If the worker died without setting unrecoverable_status_, surface the
  // stream_status_ (the per-error reason) so callers see a real error instead
  // of a silent OK.
  if (!stream_ok_) {
    return stream_status_;
  }
  return unrecoverable_status_;
}

absl::Status TrajectoryWriter::EndEpisode(bool clear_buffers,
                                          absl::Duration timeout) {
  absl::MutexLock lock(mu_);
  REVERB_RETURN_IF_ERROR(unrecoverable_status_);

  REVERB_RETURN_IF_ERROR(FlushLocked(0, timeout));

  for (auto& [_, chunker] : chunkers_) {
    if (clear_buffers) {
      chunker->Reset();
    } else {
      // This call should NEVER fail but if it does then we will not be able to
      // recover from it.
      unrecoverable_status_ = chunker->Flush();
      REVERB_RETURN_IF_ERROR(unrecoverable_status_);
    }
  }

  episode_id_ = key_generator_.Generate();
  episode_step_ = 0;
  return absl::OkStatus();
}

int TrajectoryWriter::episode_steps() const {
  absl::MutexLock lock(mu_);
  return episode_step_;
}

absl::Status TrajectoryWriter::ConfigureChunker(
    int column, const std::shared_ptr<ChunkerOptions>& options) {
  REVERB_RETURN_IF_ERROR(ValidateChunkerOptions(options.get()));

  if (auto it = chunkers_.find(column); it != chunkers_.end()) {
    return it->second->ApplyConfig(options->Clone());
  }

  options_override_[column] = options->Clone();
  return absl::OkStatus();
}

int TrajectoryWriter::max_num_keep_alive_refs() const {
  int max_value = options_.chunker_options->GetNumKeepAliveRefs();
  for (const auto& [_, options] : options_override_) {
    max_value = std::max(max_value, options->GetNumKeepAliveRefs());
  }
  return max_value;
}

TrajectoryColumn::TrajectoryColumn(std::vector<std::weak_ptr<CellRef>> refs,
                                   bool squeeze)
    : refs_(std::move(refs)), squeeze_(squeeze) {}

void TrajectoryColumn::ToProto(FlatTrajectory::Column* proto) const {
  // Note that MergeAdjacent can safely assume that all weak_ptrs are alive
  // since the corresponding shared_ptrs exists in item_and_refs.
  for (FlatTrajectory::ChunkSlice& slice : MergeAdjacent(refs_)) {
    *proto->add_chunk_slices() = std::move(slice);
  }
  proto->set_squeeze(squeeze_);
}

absl::Status TrajectoryColumn::Validate() const {
  std::vector<std::shared_ptr<CellRef>> locked_refs;
  if (!LockReferences(&locked_refs)) {
    return absl::InvalidArgumentError("Column contains expired CellRef.");
  }

  if (squeeze_ && locked_refs.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("TrajectoryColumn must contain exactly one row when "
                     "squeeze is set but got ",
                     locked_refs.size(), "."));
  }

  // Check that the column only contains compatible data references.
  const internal::TensorSpec& col_spec =
      locked_refs[0]->chunker().lock()->spec();
  for (int i = 1; i < locked_refs.size(); ++i) {
    const internal::TensorSpec& spec = locked_refs[i]->chunker().lock()->spec();
    if (spec.dtype != col_spec.dtype) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Column references tensors with different dtypes: ",
          DataTypeName(col_spec.dtype), " (index 0) != ",
          DataTypeName(spec.dtype), " (index ", i, ")."));
    }
    if (!spec.IsCompatibleWith(col_spec)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Column references tensors with incompatible shapes: ",
          ShapeString(col_spec.shape), " (index 0) not compatible with ",
          ShapeString(spec.shape), " (index ", i, ")."));
    }
  }

  return absl::OkStatus();
}

bool TrajectoryColumn::LockReferences(
    std::vector<std::shared_ptr<CellRef>>* locked_refs) const {
  for (const std::weak_ptr<CellRef>& ref : refs_) {
    locked_refs->push_back(ref.lock());
    if (!locked_refs->back()) return false;
  }
  return true;
}

}  // namespace reverb
}  // namespace deepmind
