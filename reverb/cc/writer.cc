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
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "grpcpp/client_context.h"
#include "grpcpp/impl/call_op_set.h"
#include "absl/base/thread_annotations.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/random/distributions.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "reverb/cc/platform/default/hash_set.h"
#include "reverb/cc/platform/default/logging.h"
#include "reverb/cc/platform/default/status_macros.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/platform/default/hash_map.h"
#include "reverb/cc/support/grpc_util.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/support/trajectory_util.h"
#include "reverb/cc/table.h"
#include "reverb/cc/tensor_compression.h"

namespace deepmind {
namespace reverb {
namespace {

int PositiveModulo(int value, int divisor) {
  if (divisor == 0) return value;

  if ((value > 0) == (divisor > 0)) {
    // value and divisor have the same sign.
    return value % divisor;
  } else {
    // value and divisor have different signs.
    return (value % divisor + divisor) % divisor;
  }
}

}  // namespace

Writer::Writer(std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub,
               int chunk_length, int max_timesteps, bool delta_encoded,
               std::shared_ptr<internal::FlatSignatureMap> signatures,
               int max_in_flight_items)
    : stub_(std::move(stub)),
      chunk_length_(chunk_length),
      max_timesteps_(max_timesteps),
      delta_encoded_(delta_encoded),
      max_in_flight_items_(max_in_flight_items),
      num_items_in_flight_(0),
      signatures_(std::move(signatures)),
      next_chunk_key_(NewID()),
      episode_id_(NewID()),
      index_within_episode_(0),
      closed_(false),
      inserted_dtypes_and_shapes_(max_timesteps) {
  CHECK_GT(max_in_flight_items_, 0);
}

Writer::Writer(
    internal::flat_hash_map<std::string, std::shared_ptr<Table>> tables,
    int chunk_length, int max_timesteps, bool delta_encoded,
    std::shared_ptr<internal::FlatSignatureMap> signatures,
    int max_in_flight_items)
    : is_local_(true),
      tables_(std::move(tables)),
      chunk_length_(chunk_length),
      max_timesteps_(max_timesteps),
      delta_encoded_(delta_encoded),
      max_in_flight_items_(max_in_flight_items),
      num_items_in_flight_(0),
      signatures_(std::move(signatures)),
      next_chunk_key_(NewID()),
      episode_id_(NewID()),
      index_within_episode_(0),
      closed_(false),
      inserted_dtypes_and_shapes_(max_timesteps) {
  CHECK_GT(max_in_flight_items_, 0);
  REVERB_CHECK(!tables_.empty());
}

Writer::~Writer() {
  if (!closed_) Close().IgnoreError();
}

absl::Status Writer::Append(std::vector<TensorBuffer> data) {
  if (closed_) {
    return absl::FailedPreconditionError(
        "Calling method Append after Close has been called");
  }
  if (!buffer_.empty() && buffer_.front().size() != data.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Number of tensors per timestep was "
        "inconsistent. Previously it was ",
        buffer_.front().size(), ", but is now ", data.size(), "."));
  }

  // Store flattened signature into inserted_dtypes_and_shapes_
  internal::DtypesAndShapes dtypes_and_shapes_t(0);
  dtypes_and_shapes_t->reserve(data.size());
  for (const auto& t : data) {
    dtypes_and_shapes_t->push_back(
        {/*name=*/"", t.dtype(), t.shape()});
  }
  std::swap(dtypes_and_shapes_t,
            inserted_dtypes_and_shapes_[insert_dtypes_and_shapes_location_]);
  insert_dtypes_and_shapes_location_ =
      (insert_dtypes_and_shapes_location_ + 1) % max_timesteps_;

  buffer_.push_back(std::move(data));
  if (buffer_.size() < chunk_length_) return absl::OkStatus();

  auto status = Finish(/*retry_on_unavailable=*/true);
  if (!status.ok()) {
    // Undo adding stuff to the buffer and undo the dtypes_and_shapes_ changes.
    buffer_.pop_back();
    insert_dtypes_and_shapes_location_ =
        PositiveModulo(insert_dtypes_and_shapes_location_ - 1, max_timesteps_);
    std::swap(dtypes_and_shapes_t,
              inserted_dtypes_and_shapes_[insert_dtypes_and_shapes_location_]);
  }
  return status;
}

absl::Status Writer::AppendSequence(std::vector<TensorBuffer> sequence) {
  if (sequence.empty()) {
    return absl::InvalidArgumentError("AppendSequence called with empty data.");
  }
  for (int i = 0; i < sequence.size(); i++) {
    if (sequence[i].shape().empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "AppendSequence called with scalar tensor at index ", i, "."));
    }
    if (sequence[i].shape()[0] != sequence[0].shape()[0]) {
      std::vector<internal::TensorSpec> specs;
      specs.reserve(sequence.size());
      for (const auto& t : sequence) {
        specs.push_back({/*name=*/"", t.dtype(), t.shape()});
      }
      return absl::InvalidArgumentError(
          absl::StrCat("AppendSequence called with tensors of non equal batch "
                       "dimension: ",
                       internal::DtypesShapesString(specs), "."));
    }
  }

  for (int i = 0; i < sequence[0].shape()[0]; i++) {
    std::vector<TensorBuffer> step;
    step.reserve(sequence.size());
    for (const auto& column : sequence) {
      step.push_back(column.SubSlice(i));
    }
    REVERB_RETURN_IF_ERROR(Append(std::move(step)));
  }

  return absl::OkStatus();
}

absl::Status Writer::CreateItem(const std::string& table, int num_timesteps,
                                double priority) {
  if (closed_) {
    return absl::FailedPreconditionError(
        "Calling method CreateItem after Close has been called");
  }
  if (num_timesteps > chunks_.size() * chunk_length_ + buffer_.size()) {
    return absl::InvalidArgumentError(
        "Argument `num_timesteps` is larger than number of buffered "
        "timesteps.");
  }
  if (num_timesteps > max_timesteps_) {
    return absl::InvalidArgumentError(
        "`num_timesteps` must be <= `max_timesteps`");
  }
  if (std::isnan(priority)) {
    return absl::InvalidArgumentError("`priority` must not be nan.");
  }

  const internal::DtypesAndShapes* dtypes_and_shapes = nullptr;
  REVERB_RETURN_IF_ERROR(GetFlatSignature(table, &dtypes_and_shapes));
  CHECK(dtypes_and_shapes != nullptr);
  if (dtypes_and_shapes->has_value()) {
    for (int t = 0; t < num_timesteps; ++t) {
      // Subtract 1 from the location since it is currently pointing to the next
      // write.
      const int check_offset = PositiveModulo(
          insert_dtypes_and_shapes_location_ - 1 - t, max_timesteps_);
      const auto& dtypes_and_shapes_t =
          inserted_dtypes_and_shapes_[check_offset];
      if (!dtypes_and_shapes_t.has_value()) {
        return absl::InternalError(
            absl::StrCat("Unexpected missing dtypes and shapes while calling "
                         "CreateItem: "
                         "expected a value at index ",
                         check_offset, " (timestep offset ", t, ")"));
      }

      if (dtypes_and_shapes_t->size() != (*dtypes_and_shapes)->size()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Unable to CreateItem in table '", table,
            "' because Append was called with a tensor signature "
            "inconsistent with table signature.  Append for timestep "
            "offset ",
            t, " was called with ", dtypes_and_shapes_t->size(),
            " tensors, but table requires ", (*dtypes_and_shapes)->size(),
            " tensors per entry.  Table signature: ",
            internal::DtypesShapesString(**dtypes_and_shapes),
            ", data signature: ",
            internal::DtypesShapesString(*dtypes_and_shapes_t), "."));
      }

      for (int c = 0; c < dtypes_and_shapes_t->size(); ++c) {
        const auto& signature_dtype_and_shape = (**dtypes_and_shapes)[c];
        const auto& seen_dtype_and_shape = (*dtypes_and_shapes_t)[c];

        const bool dtypes_equal =
            seen_dtype_and_shape.dtype == signature_dtype_and_shape.dtype;

        // In order to make the writer compatible with tables with signatures
        // defined for the entire trajectory (rather than for a single step)
        // we consider the shape valid if either a single step matches the
        // signature or if the entire trajectory matches the signature.
        const bool timestep_shape_compatible =
            signature_dtype_and_shape.IsCompatibleWith(seen_dtype_and_shape);
        // Prepend the trajectory length to the per-step shape and check
        // compatibility against the (possibly wildcarded) table signature.
        std::vector<int64_t> trajectory_shape;
        trajectory_shape.reserve(seen_dtype_and_shape.shape.size() + 1);
        trajectory_shape.push_back(num_timesteps);
        for (int64_t d : seen_dtype_and_shape.shape) {
          trajectory_shape.push_back(d);
        }
        const internal::TensorSpec trajectory_spec{
            seen_dtype_and_shape.name, seen_dtype_and_shape.dtype,
            std::move(trajectory_shape)};
        const bool trajectory_shape_compatible =
            signature_dtype_and_shape.IsCompatibleWith(trajectory_spec);

        if (!dtypes_equal ||
            !(timestep_shape_compatible || trajectory_shape_compatible)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Unable to CreateItem in table '", table,
              "' because Append was called with a tensor signature "
              "inconsistent with the table signature. At timestep offset ",
              t, ", flattened index ", c, ", saw a tensor of dtype ",
              DataTypeName(seen_dtype_and_shape.dtype), ", shape ",
              seen_dtype_and_shape.DebugString(),
              ", but expected tensor '", signature_dtype_and_shape.name,
              "' of dtype ", DataTypeName(signature_dtype_and_shape.dtype),
              " and shape compatible with ",
              signature_dtype_and_shape.DebugString(),
              ".  (Flattened) table signature: ",
              internal::DtypesShapesString(**dtypes_and_shapes),
              ", data signature: ",
              internal::DtypesShapesString(*dtypes_and_shapes_t), "."));
        }
      }
    }
  }

  PrioritizedItem item;
  item.set_key(NewID());
  item.set_table(table.data(), table.size());
  item.set_priority(priority);

  // Build the trajectory.
  int remaining = num_timesteps;
  std::vector<int> chunk_lengths;
  std::vector<uint64_t> chunk_keys;

  // Include the current chunk unless empty.
  if (!buffer_.empty()) {
    chunk_lengths.push_back(buffer_.size());
    chunk_keys.push_back(next_chunk_key_);
    remaining -= buffer_.size();
  }

  // Traverse historic chunks backwards until trajectory complete.
  for (auto rit = chunks_.rbegin(); remaining > 0 && rit != chunks_.rend();
       ++rit) {
    const auto& range = rit->sequence_range();
    chunk_lengths.push_back(range.end() - range.start() + 1);
    chunk_keys.push_back(rit->chunk_key());
    remaining -= chunk_lengths.back();
  }

  // Reverse the chunk order.
  std::reverse(chunk_lengths.begin(), chunk_lengths.end());
  std::reverse(chunk_keys.begin(), chunk_keys.end());

  *item.mutable_flat_trajectory() = internal::FlatTimestepTrajectory(
      chunk_keys, chunk_lengths,
      /*num_columns=*/buffer_.empty() ? chunks_.front().data().tensors_size()
                                      : buffer_[0].size(),
      /*offset=*/-remaining,
      /*length=*/num_timesteps);

  pending_items_.push_back(item);

  if (buffer_.empty()) {
    auto status = WriteWithRetries(/*retry_on_unavailable=*/true);
    if (!status.ok()) pending_items_.pop_back();
    return status;
  }

  return absl::OkStatus();
}

absl::Status Writer::Flush() {
  if (closed_) {
    return absl::FailedPreconditionError(
        "Calling method Flush after Close has been called");
  }

  if (!pending_items_.empty()) {
    return Finish(/*retry_on_unavailable=*/true);
  }
  if (!ConfirmItems(0)) {
    return absl::InternalError(
        "Error when confirming that all items written to table.");
  }
  return absl::OkStatus();
}

std::string Writer::DebugString() const {
  std::string str = absl::StrCat(
      "Writer(chunk_length=", chunk_length_, ", max_timesteps=", max_timesteps_,
      ", delta_encoded=", delta_encoded_, ", max_in_flight_items=",
      max_in_flight_items_, ", episode_id=", episode_id_,
                  ", index_within_episode=", index_within_episode_,
                  ", closed=", closed_, ")");
  return str;
}

absl::Status Writer::StopItemConfirmationWorker() {
  absl::MutexLock lock(mu_);
  item_confirmation_worker_stop_requested_ = true;

  mu_.Await(absl::Condition(
      +[](bool* running) { return !(*running); },
      &item_confirmation_worker_running_));

  item_confirmation_worker_stop_requested_ = false;
  item_confirmation_worker_thread_ = nullptr;

  if (num_items_in_flight_ > 0) {
    return absl::DataLossError(absl::StrCat(
        "Item confirmation worker were stopped when ", num_items_in_flight_,
        " unconfirmed items (sent to server but validation "
        "response not yet "
        "received)."));
  }
  num_items_in_flight_ = 0;
  return absl::OkStatus();
}

void Writer::StartItemConfirmationWorker() {
  absl::MutexLock lock(mu_);

  // Sanity check the state of the writer. None of these should ever trigger in
  // the absence of fatal bugs.
  REVERB_CHECK(stream_ != nullptr);
  REVERB_CHECK(item_confirmation_worker_thread_ == nullptr);
  REVERB_CHECK_EQ(num_items_in_flight_, 0);
  REVERB_CHECK(!item_confirmation_worker_running_);
  REVERB_CHECK(!item_confirmation_worker_stop_requested_);

  item_confirmation_worker_thread_ = internal::StartThread(
      "WriterItemConfirmer",
      absl::bind_front(&Writer::ItemConfirmationWorker, this));
  mu_.Await(absl::Condition(
      +[](bool* started) { return *started; },
      &item_confirmation_worker_running_));
}

absl::Status Writer::Close(bool retry_on_unavailable) {
  if (closed_) {
    return absl::FailedPreconditionError(
        "Calling method Close after Close has been called");
  }
  bool server_unavailable = false;
  if (!pending_items_.empty()) {
    auto status = Finish(retry_on_unavailable);
    if (!status.ok()) {
      if (!absl::IsUnavailable(status) || retry_on_unavailable) {
        return status;
      }
      server_unavailable = true;
      // if retries are disabled and the server is Unavailable, we continue and
      // set the Writer as closed.
      REVERB_LOG(REVERB_INFO)
          << "The Writer will be closed although the server was Unavailable";
    }
  }
  if (stream_) {
    if (!server_unavailable){
      // If the stream was closed on the server side, we cannot write on the
      // stream, but there is no good way of checking if the stream is closed
      // from the client without calling Read or Finish.
      stream_->WritesDone();
      REVERB_LOG_IF(REVERB_ERROR, !ConfirmItems(0))
          << "Unable to confirm that items were written.";
    }
    auto confirmation_status = StopItemConfirmationWorker();
    REVERB_LOG_IF(REVERB_ERROR, !confirmation_status.ok())
        << "Error when stopping the confirmation worker: "
        << confirmation_status;
    auto status = stream_->Finish();
    if (!status.ok()) {
      REVERB_LOG(REVERB_INFO) << "Received error when closing the stream: "
                              << FormatGrpcStatus(status);
    }
    stream_ = nullptr;
  } else if (is_local_) {
    // Local path: no stream/confirmation worker. Drain all in-flight inserts via
    // the callback-driven `ConfirmItems(0)`, then release local chunk/callback
    // state.
    REVERB_LOG_IF(REVERB_ERROR, !ConfirmItems(0))
        << "Unable to confirm that all items were written.";
    local_chunks_.clear();
    local_pending_callbacks_.clear();
  }
  chunks_.clear();
  closed_ = true;
  return absl::OkStatus();
}

absl::Status Writer::Finish(bool retry_on_unavailable) {
  std::vector<TensorBuffer> batched_tensors;
  for (int i = 0; i < buffer_[0].size(); ++i) {
    std::vector<TensorBuffer> tensors;
    tensors.reserve(buffer_.size());
    for (int j = 0; j < buffer_.size(); ++j) {
      const TensorBuffer& item = buffer_[j][i];
      if (j > 0 && item.shape() != buffer_[0][i].shape()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Unable to concatenate tensors at index ", i,
                         " due to mismatched shapes.  Tensor 0 has shape: ",
                         internal::TensorSpec{"", item.dtype(),
                                              buffer_[0][i].shape()}
                             .DebugString(),
                         ", but tensor ", j, " has shape: ",
                         internal::TensorSpec{"", item.dtype(), item.shape()}
                             .DebugString()));
      }
      // Insert a leading batch dimension of 1 so Concat stacks the timesteps.
      tensors.push_back(item.InsertBatchDim());
    }
    auto concat_result = TensorBuffer::Concat(tensors);
    REVERB_RETURN_IF_ERROR(concat_result.status());
    batched_tensors.emplace_back(std::move(*concat_result));
  }

  ChunkData chunk_data;
  chunk_data.set_chunk_key(next_chunk_key_);
  chunk_data.mutable_sequence_range()->set_episode_id(episode_id_);
  chunk_data.mutable_sequence_range()->set_start(index_within_episode_);
  chunk_data.mutable_sequence_range()->set_end(index_within_episode_ +
                                               buffer_.size() - 1);

  // Calculate the size of the data before compression.
  int64_t total_uncompressed_size = 0;
  for (const auto& tensor : batched_tensors) {
    total_uncompressed_size += tensor.TotalBytes();
  }
  chunk_data.set_data_uncompressed_size(total_uncompressed_size);

  if (delta_encoded_) {
    batched_tensors = DeltaEncodeList(batched_tensors, true);
    chunk_data.set_delta_encoded(true);
  }

  chunk_data.set_data_tensors_len(batched_tensors.size());
  for (const auto& tensor : batched_tensors) {
    REVERB_RETURN_IF_ERROR(CompressTensorAsProto(
        tensor, chunk_data.mutable_data()->add_tensors()));
  }

  chunks_.emplace_back(std::move(chunk_data));

  auto status = WriteWithRetries(retry_on_unavailable);
  if (status.ok()) {
    index_within_episode_ += buffer_.size();
    buffer_.clear();
    next_chunk_key_ = NewID();
    while ((chunks_.size() - 1) * chunk_length_ >= max_timesteps_) {
      streamed_chunk_keys_.erase(chunks_.front().chunk_key());
      chunks_.pop_front();
    }
  } else {
    chunks_.pop_back();
  }
  return status;
}

absl::Status Writer::WriteWithRetries(bool retry_on_unavailable) {
  // Local mode: no stream, no retries. `WritePendingDataLocal` dispatches
  // directly into the tables and its insert-completion callbacks drive
  // `num_items_in_flight_`.
  if (is_local_) {
    return WritePendingDataLocal() ? absl::OkStatus()
                                   : absl::InternalError(
                                         "Local Writer failed to write data.");
  }
  while (true) {
    if (WritePendingData()) return absl::OkStatus();
    stream_->WritesDone();
    REVERB_RETURN_IF_ERROR(StopItemConfirmationWorker());
    auto status = FromGrpcStatus(stream_->Finish());
    stream_ = nullptr;
    if (!absl::IsUnavailable(status) || !retry_on_unavailable) {
      return status;
    }
  }
}

bool Writer::WritePendingData() {
  class ArenaOwnedRequest {
   public:
    ~ArenaOwnedRequest() {
      Clear();
    }

    void Clear() {
      while (!r.chunks().empty()) {
        r.mutable_chunks()->UnsafeArenaReleaseLast();
      }
    }
    InsertStreamRequest r;
  };
  if (!stream_) {
    streamed_chunk_keys_.clear();
    context_ = std::make_unique<grpc::ClientContext>();
    stream_ = stub_->InsertStream(context_.get());
    StartItemConfirmationWorker();
  }

  // Stream all chunks which are referenced by the pending items and haven't
  // already been sent. After the items has been inserted we want the server
  // to keep references only to the ones which the client still keeps
  // around.
  internal::flat_hash_set<uint64_t> item_chunk_keys;
  for (const auto& item : pending_items_) {
    for (auto key : internal::GetChunkKeys(item.flat_trajectory())) {
      item_chunk_keys.insert(key);
    }
  }

  ArenaOwnedRequest request;
  grpc::WriteOptions options;
  options.set_no_compression();
  std::vector<uint64_t> keep_chunk_keys;
  for (auto& chunk : chunks_) {
    if (item_chunk_keys.contains(chunk.chunk_key()) &&
        !streamed_chunk_keys_.contains(chunk.chunk_key())) {
      request.r.mutable_chunks()->UnsafeArenaAddAllocated(&chunk);
      keep_chunk_keys.push_back(chunk.chunk_key());

      // If the message has grown beyond the cutoff point then we send it.
      if (request.r.ByteSizeLong() >= Writer::kMaxRequestSizeBytes) {
        if (!stream_->Write(request.r, options)) {
          return false;
        }
        for (const auto& chunk : request.r.chunks()) {
          streamed_chunk_keys_.insert(chunk.chunk_key());
        }
        request.Clear();
      }
    } else if (streamed_chunk_keys_.contains(chunk.chunk_key())) {
      keep_chunk_keys.push_back(chunk.chunk_key());
    }
  }
  while (!pending_items_.empty()) {
    if (!ConfirmItems(max_in_flight_items_ - 1)) {
      return false;
    }
    *request.r.add_items() = pending_items_.front();
    *request.r.mutable_keep_chunk_keys() = {
        keep_chunk_keys.begin(), keep_chunk_keys.end()};
    bool ok = stream_->Write(request.r, options);
    request.r.clear_items();
    if (!ok) return false;
    for (const auto& chunk : request.r.chunks()) {
      streamed_chunk_keys_.insert(chunk.chunk_key());
    }
    pending_items_.pop_front();
    absl::MutexLock lock(mu_);
    ++num_items_in_flight_;
  }

  return true;
}

bool Writer::WritePendingDataLocal() {
  // Materialize referenced chunks into `shared_ptr<ChunkStore::Chunk>` and
  // dedup via `local_chunks_` (keyed by chunk key), mirroring the reactor's
  // `chunks_` map. Only chunks not already materialized are created. This
  // keeps a single shared_ptr per chunk so multi-table inserts referencing
  // the same chunk share storage (gRPC-path semantics).
  for (auto& chunk : chunks_) {
    uint64_t key = chunk.chunk_key();
    if (!streamed_chunk_keys_.contains(key)) {
      // `ChunkStore::Chunk` owns a copy of the `ChunkData` proto.
      local_chunks_[key] = std::make_shared<ChunkStore::Chunk>(chunk);
      streamed_chunk_keys_.insert(key);
    }
  }

  while (!pending_items_.empty()) {
    // Backpressure: block until in-flight count drops below the limit. The
    // insert-completion callback (fired on the table worker thread) decrements
    // `num_items_in_flight_` and wakes this await.
    if (!ConfirmItems(max_in_flight_items_ - 1)) {
      return false;
    }

    PrioritizedItem item = pending_items_.front();
    const std::string& table_name = item.table();
    auto table_it = tables_.find(table_name);
    if (table_it == tables_.end()) {
      return false;
    }

    // Gather the chunks referenced by this item's trajectory.
    std::vector<uint64_t> item_chunk_keys =
        internal::GetChunkKeys(item.flat_trajectory());
    std::vector<std::shared_ptr<ChunkStore::Chunk>> item_chunks;
    item_chunks.reserve(item_chunk_keys.size());
    for (uint64_t ck : item_chunk_keys) {
      auto it = local_chunks_.find(ck);
      if (it == local_chunks_.end()) {
        // Should not happen: the chunk was just materialized above.
        return false;
      }
      item_chunks.push_back(it->second);
    }

    uint64_t item_key = item.key();
    TableItem table_item(std::move(item), std::move(item_chunks));

    // Increment in-flight BEFORE dispatching: the table worker runs inserts on
    // its own thread and the completion callback can fire before
    // `InsertOrAssignAsync` returns. Registering first keeps the count correct
    // and lets the callback's decrement land.
    {
      absl::MutexLock lock(mu_);
      ++num_items_in_flight_;
    }

    // ponytail: per-item shared_ptr callback so the weak_ptr handed to the
    // table stays alive until the insert completes. Kept in
    // `local_pending_callbacks_` (cleared on Close drain) so the table worker's
    // `weak_ptr::lock` succeeds. The callback decrements
    // `num_items_in_flight_`, mirroring gRPC's ItemConfirmationWorker.
    auto cb = std::make_shared<Table::InsertCallback>(
        [this](uint64_t /*completed_key*/) {
          absl::MutexLock lock(mu_);
          --num_items_in_flight_;
        });
    local_pending_callbacks_.push_back(cb);

    bool can_insert_more = false;
    absl::Status s = table_it->second->InsertOrAssignAsync(
        std::move(table_item), &can_insert_more, cb);
    if (!s.ok()) {
      // Undo the in-flight bump; surface failure.
      absl::MutexLock lock(mu_);
      --num_items_in_flight_;
      return false;
    }

    pending_items_.pop_front();

    // If the table's insert queue is full, wait for an outstanding insert to
    // complete before pulling another item (writer-level backpressure, aligned
    // with the gRPC path's ConfirmItems gate above).
    if (!can_insert_more) {
      ConfirmItems(max_in_flight_items_ - 1);
    }
  }
  return true;
}

uint64_t Writer::NewID() {
  return absl::Uniform<uint64_t>(bit_gen_, 0, UINT64_MAX);
}

bool Writer::ConfirmItems(int limit) {
  absl::ReaderMutexLock lock(mu_);
  // gRPC: unblock when the worker thread stops (stream error) so awaiters
  // don't deadlock. Local: there is no confirmation worker thread; the
  // insert-completion callback decrements `num_items_in_flight_`, so we wait
  // on `num_items_in_flight_ <= limit` with a `closed_` escape (mirrors
  // TrajectoryWriter's `!closed_ && !stream_ok_` await pattern) to avoid
  // permanent blocking if a table worker errors and a callback never fires.
  auto done = [limit, this]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    if (is_local_) {
      return num_items_in_flight_ <= limit || closed_;
    }
    return num_items_in_flight_ <= limit ||
           !item_confirmation_worker_running_;
  };
  mu_.Await(absl::Condition(&done));
  return num_items_in_flight_ <= limit;
}

absl::Status Writer::GetFlatSignature(
    absl::string_view table,
    const internal::DtypesAndShapes** dtypes_and_shapes) const {
  static const auto* empty_dtypes_and_shapes =
      new internal::DtypesAndShapes(std::nullopt);
  if (!signatures_) {
    // No signatures available, return an unknown set.
    *dtypes_and_shapes = empty_dtypes_and_shapes;
    return absl::OkStatus();
  }
  auto iter = signatures_->find(table);
  if (iter == signatures_->end()) {
    std::vector<std::string> table_names;
    for (const auto& table : *signatures_) {
      table_names.push_back(absl::StrCat("'", table.first, "'"));
    }
    return absl::InvalidArgumentError(
        absl::StrCat("Unable to find signatures for table '", table,
                     "' in signature cache.  Available tables: [",
                     absl::StrJoin(table_names, ", "), "]."));
  }
  *dtypes_and_shapes = &(iter->second);
  return absl::OkStatus();
}

void Writer::ItemConfirmationWorker() {
  InsertStreamResponse response;
  while (true) {
    {
      absl::MutexLock lock(mu_);

      // Setting this will unblock `StartItemConfirmationWorker` when Await is
      // called.
      item_confirmation_worker_running_ = true;

      // Wait until an item has been sent we expect the server to respond or
      // until the reader has been explicitly requested to stop prematurely.
      mu_.Await(absl::Condition(
          +[](Writer* w) ABSL_SHARED_LOCKS_REQUIRED(mu_) {
            return w->num_items_in_flight_ > 0 ||
                   w->item_confirmation_worker_stop_requested_;
          },
          this));
      if (item_confirmation_worker_stop_requested_) break;
    }
    if (!stream_->Read(&response)) break;
    absl::WriterMutexLock lock(mu_);
    num_items_in_flight_ -= response.keys_size();
  }
  absl::WriterMutexLock lock(mu_);
  item_confirmation_worker_running_ = false;
}

}  // namespace reverb
}  // namespace deepmind
