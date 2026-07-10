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

#ifndef REVERB_CC_SHM_RING_H_
#define REVERB_CC_SHM_RING_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "reverb/cc/shm/shm_protocol.pb.h"

namespace deepmind {
namespace reverb {
namespace shm {

// Wire constants. A ring segment is laid out as:
//   [RingHeader (128 bytes)] [capacity * slot_size bytes of slots]
// Each slot is [SlotHeader (16 bytes)][body bytes].
inline constexpr uint64_t kRingMagic = 0x524556524253484DULL;  // "REVRBSHM"
inline constexpr uint32_t kRingVersion = 1;
// ponytail: fixed slot/capacity for v1; make configurable if throughput needs it.
inline constexpr size_t kDefaultSlotSize = 256;    // bytes (incl. SlotHeader)
inline constexpr size_t kDefaultCapacity = 1024;   // slots (256KB/ring)

// Segment header, exactly 128 bytes = 2 cache lines. `head` (producer) lives
// on the first line, `tail` (consumer) on the second, so the SPSC producer
// and consumer never share a cache line (no false sharing). Both start at 1;
// seq 0 means a slot was never written. We lay the fields out explicitly
// rather than relying on alignas(64) members (which would inflate the struct).
struct RingHeader {
  // Line 0 (offset 0..63).
  uint64_t magic = kRingMagic;
  uint32_t version = kRingVersion;
  uint32_t capacity = kDefaultCapacity;   // slot count (power of two)
  uint32_t slot_size = kDefaultSlotSize;  // bytes per slot (incl. SlotHeader)
  uint32_t reserved = 0;
  uint64_t capacity_mask = kDefaultCapacity - 1;
  std::atomic<uint64_t> head{1};  // producer: next slot seq to write
  uint64_t pad0[3];               // pad line 0 to 64 bytes
  // Line 1 (offset 64..127).
  std::atomic<uint64_t> tail{1};  // consumer: next slot seq to read
  uint64_t pad1[7];               // pad line 1 to 64 bytes
};
static_assert(sizeof(RingHeader) == 128,
              "RingHeader must be exactly 128 bytes (2 cache lines)");

// Per-slot header, exactly 16 bytes. `seq` is the producer's head value when
// this slot was written; the consumer treats the slot as valid when seq == its
// expected tail. seq 0 = never written. `msg_type` is stored as uint16_t (the
// proto MsgType values fit); cast at the API boundary.
struct SlotHeader {
  uint64_t seq = 0;  // published via std::atomic ops in Ring::Write/Read
  uint16_t msg_type = 0;   // MsgType enum value (raw uint16 on the wire)
  uint16_t flags = 0;     // bit0: HAS_CONTINUATION; bit1: IS_CONTINUATION
  uint32_t body_len = 0;  // body bytes in this slot (<= slot_size - 16)
};
static_assert(sizeof(SlotHeader) == 16, "SlotHeader must be exactly 16 bytes");

inline constexpr uint16_t kFlagHasContinuation = 0x0001;
inline constexpr uint16_t kFlagIsContinuation = 0x0002;

// A single-producer single-consumer ring over a POSIX shared-memory segment.
// One side calls Create (server/owner), the other Open (client). Write blocks
// (busy-wait sched_yield) when full; Read is NON-BLOCKING and returns
// NotFoundError("NOT_READY") when no message is ready (spec §3.1). The busy-wait
// polling policy for callers that need blocking semantics lives in
// ShmConnection (spec R5), not here.
class Ring {
 public:
  Ring();
  ~Ring();

  Ring(const Ring&) = delete;
  Ring& operator=(const Ring&) = delete;
  Ring(Ring&& other) noexcept;
  Ring& operator=(Ring&& other) noexcept;

  // Owner side: create + truncate a new SHM segment, init the header.
  static absl::StatusOr<Ring> Create(std::string shm_name,
                                     uint32_t capacity = kDefaultCapacity,
                                     uint32_t slot_size = kDefaultSlotSize);

  // Client side: open an existing SHM segment read/write.
  static absl::StatusOr<Ring> Open(std::string shm_name);

  // Write one message (may span multiple slots). Blocks (sched_yield) until
  // enough contiguous slots are free. Returns RESOURCE_EXHAUSTED if the message
  // is larger than the whole ring.
  absl::Status Write(MsgType msg_type, absl::Span<const char> payload);

  // Non-blocking write (ticket ⑥ / spec §8.7). Identical slot math to Write,
  // but when `num_slots` free slots are not available, returns
  // ResourceExhaustedError("RING_FULL") IMMEDIATELY without touching `head`
  // — so the caller (ShmServer::EnqueueS2C) can stash the message in an
  // outbox and retry next pass instead of blocking the dispatch thread.
  absl::Status TryWrite(MsgType msg_type, absl::Span<const char> payload);

  // Read one message, reassembling cross-slot fragments. NON-BLOCKING: if no
  // message is ready (next slot's seq != consumer_seq), returns
  // absl::NotFoundError("NOT_READY") immediately (spec §3.1). A missing
  // continuation slot after the first is a corruption signal and returns
  // InternalError. Callers needing blocking reads must poll at the call site
  // (per spec R5, that policy belongs to ShmConnection, not Ring).
  absl::Status Read(MsgType* msg_type, std::string* payload);

  uint32_t capacity() const { return header_->capacity; }
  uint32_t slot_size() const { return header_->slot_size; }
  const std::string& shm_name() const { return shm_name_; }

  // Total segment bytes for the given geometry (header + slots).
  static size_t TotalBytes(uint32_t capacity, uint32_t slot_size);

 private:
  // Pointers into the mapped segment (header_ owns the mapping lifetime).
  RingHeader* header_ = nullptr;
  SlotHeader* slots_ = nullptr;
  void* mapped_ = nullptr;      // mmap base, for munmap
  size_t mapped_len_ = 0;
  std::string shm_name_;
  bool owner_ = false;          // true => shm_unlink on destruction

  SlotHeader* Slot(uint64_t seq) const {
    return reinterpret_cast<SlotHeader*>(
        reinterpret_cast<char*>(slots_) +
        (seq & header_->capacity_mask) * header_->slot_size);
  }

  char* SlotBody(SlotHeader* s) const {
    return reinterpret_cast<char*>(s) + sizeof(SlotHeader);
  }

  size_t SlotBodyCap() const { return header_->slot_size - sizeof(SlotHeader); }

  // Shared body of Write/TryWrite: fill `num_slots` slots starting at
  // `first_seq` and publish head. Caller has already verified free space.
  void WriteSlots(MsgType msg_type, absl::Span<const char> payload,
                  uint64_t first_seq, size_t num_slots, size_t body_cap);

  void Release();
};

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SHM_RING_H_
