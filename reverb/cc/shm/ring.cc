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

#include "reverb/cc/shm/ring.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <new>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace deepmind {
namespace reverb {
namespace shm {

namespace {

// Wrap errno into an absl status with the symbolic name + message.
absl::Status ErrnoStatus(std::string_view op, std::string_view detail) {
  return absl::InternalError(
      absl::StrCat(op, " failed: ", detail, " (errno ", errno, ": ",
                   std::strerror(errno), ")"));
}

}  // namespace

Ring::Ring() = default;

Ring::~Ring() { Release(); }

Ring::Ring(Ring&& other) noexcept { *this = std::move(other); }

Ring& Ring::operator=(Ring&& other) noexcept {
  if (this != &other) {
    Release();
    header_ = other.header_;
    slots_ = other.slots_;
    mapped_ = other.mapped_;
    mapped_len_ = other.mapped_len_;
    shm_name_ = std::move(other.shm_name_);
    owner_ = other.owner_;
    other.header_ = nullptr;
    other.slots_ = nullptr;
    other.mapped_ = nullptr;
    other.mapped_len_ = 0;
    other.owner_ = false;
  }
  return *this;
}

void Ring::Release() {
  if (mapped_) {
    munmap(mapped_, mapped_len_);
  }
  if (owner_ && !shm_name_.empty()) {
    shm_unlink(shm_name_.c_str());
  }
  mapped_ = nullptr;
  header_ = nullptr;
  slots_ = nullptr;
  mapped_len_ = 0;
  owner_ = false;
}

// static
size_t Ring::TotalBytes(uint32_t capacity, uint32_t slot_size) {
  return sizeof(RingHeader) + static_cast<size_t>(capacity) * slot_size;
}

absl::StatusOr<Ring> Ring::Create(std::string shm_name, uint32_t capacity,
                                  uint32_t slot_size) {
  // ponytail: require power-of-two capacity so the slot index is a bitmask.
  // Non-pow2 would need a modulo; not worth it for a fixed-size ring.
  if (capacity == 0 || (capacity & (capacity - 1)) != 0) {
    return absl::InvalidArgumentError("capacity must be a power of two");
  }
  if (slot_size <= sizeof(SlotHeader)) {
    return absl::InvalidArgumentError("slot_size must exceed SlotHeader");
  }

  int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0600);
  if (fd < 0) {
    // R6: stale segment under the same name. Unlink and retry once.
    if (errno == EEXIST) {
      shm_unlink(shm_name.c_str());
      fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0600);
    }
    if (fd < 0) {
      return ErrnoStatus("shm_open", shm_name);
    }
  }

  size_t total = TotalBytes(capacity, slot_size);
  if (ftruncate(fd, total) != 0) {
    auto s = ErrnoStatus("ftruncate", shm_name);
    close(fd);
    shm_unlink(shm_name.c_str());
    return s;
  }

  void* base =
      mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (base == MAP_FAILED) {
    auto s = ErrnoStatus("mmap", shm_name);
    shm_unlink(shm_name.c_str());
    return s;
  }

  Ring ring;
  ring.mapped_ = base;
  ring.mapped_len_ = total;
  ring.shm_name_ = std::move(shm_name);
  ring.owner_ = true;
  ring.header_ = static_cast<RingHeader*>(base);
  ring.slots_ = reinterpret_cast<SlotHeader*>(static_cast<char*>(base) +
                                              sizeof(RingHeader));

  // Initialize the header (server is the only writer here).
  ring.header_->magic = kRingMagic;
  ring.header_->version = kRingVersion;
  ring.header_->capacity = capacity;
  ring.header_->slot_size = slot_size;
  ring.header_->reserved = 0;
  ring.header_->capacity_mask = capacity - 1;
  ring.header_->head.store(1, std::memory_order_relaxed);
  ring.header_->tail.store(1, std::memory_order_relaxed);
  // All slots start with seq 0 (unwritten); consumer never reads them because
  // tail starts at 1 and only advances past slots whose seq == expected value.
  return ring;
}

absl::StatusOr<Ring> Ring::Open(std::string shm_name) {
  int fd = shm_open(shm_name.c_str(), O_RDWR, 0600);
  if (fd < 0) {
    return ErrnoStatus("shm_open", shm_name);
  }

  // Read the header to learn the segment geometry before mapping the full size.
  RingHeader hdr;
  if (read(fd, &hdr, sizeof(hdr)) != sizeof(hdr)) {
    auto s = ErrnoStatus("read header", shm_name);
    close(fd);
    return s;
  }
  if (hdr.magic != kRingMagic) {
    close(fd);
    return absl::InternalError(
        absl::StrCat("bad ring magic in ", shm_name));
  }

  size_t total = TotalBytes(hdr.capacity, hdr.slot_size);
  void* base =
      mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (base == MAP_FAILED) {
    return ErrnoStatus("mmap", shm_name);
  }

  Ring ring;
  ring.mapped_ = base;
  ring.mapped_len_ = total;
  ring.shm_name_ = std::move(shm_name);
  ring.owner_ = false;
  ring.header_ = static_cast<RingHeader*>(base);
  ring.slots_ = reinterpret_cast<SlotHeader*>(static_cast<char*>(base) +
                                              sizeof(RingHeader));
  return ring;
}

absl::Status Ring::Write(MsgType msg_type, absl::Span<const char> payload) {
  size_t body_cap = SlotBodyCap();
  size_t len = payload.size();
  size_t num_slots = (len + body_cap - 1) / body_cap;
  if (num_slots == 0) num_slots = 1;
  if (num_slots > header_->capacity) {
    // PERMANENT failure (retry can never succeed) — distinct from the
    // transient RING_FULL ResourceExhausted so callers never stash-and-retry.
    return absl::InvalidArgumentError("message larger than ring capacity");
  }

  // Wait for `num_slots` contiguous free slots. SPSC: producer reads tail
  // (acquire); head - tail == used, so free = capacity - used.
  uint64_t first_seq = header_->head.load(std::memory_order_relaxed);
  while (first_seq - header_->tail.load(std::memory_order_acquire) >
         header_->capacity - num_slots) {
    sched_yield();  // ponytail: busy-yield; no semaphore to avoid crash leaks
  }

  WriteSlots(msg_type, payload, first_seq, num_slots, body_cap);
  return absl::OkStatus();
}

absl::Status Ring::TryWrite(MsgType msg_type,
                            absl::Span<const char> payload) {
  size_t body_cap = SlotBodyCap();
  size_t len = payload.size();
  size_t num_slots = (len + body_cap - 1) / body_cap;
  if (num_slots == 0) num_slots = 1;
  if (num_slots > header_->capacity) {
    // PERMANENT failure — see Write().
    return absl::InvalidArgumentError("message larger than ring capacity");
  }

  // Non-blocking (spec §8.7): if not enough free slots, return immediately
  // WITHOUT touching head — the caller stashes the message and retries.
  uint64_t first_seq = header_->head.load(std::memory_order_relaxed);
  if (first_seq - header_->tail.load(std::memory_order_acquire) >
      header_->capacity - num_slots) {
    return absl::ResourceExhaustedError("RING_FULL");
  }

  WriteSlots(msg_type, payload, first_seq, num_slots, body_cap);
  return absl::OkStatus();
}

void Ring::WriteSlots(MsgType msg_type, absl::Span<const char> payload,
                      uint64_t first_seq, size_t num_slots,
                      size_t body_cap) {
  const char* src = payload.data();
  size_t len = payload.size();
  for (size_t i = 0; i < num_slots; i++) {
    uint64_t seq = first_seq + i;
    SlotHeader* s = Slot(seq);
    size_t chunk_len = std::min(body_cap, len - i * body_cap);
    s->msg_type = static_cast<uint16_t>(i == 0 ? msg_type : MSG_TYPE_UNSPECIFIED);
    s->flags = 0;
    if (i + 1 < num_slots) s->flags |= kFlagHasContinuation;
    if (i > 0) s->flags |= kFlagIsContinuation;
    s->body_len = static_cast<uint32_t>(chunk_len);
    std::memcpy(SlotBody(s), src + i * body_cap, chunk_len);
    // Publish continuation slots immediately; slot 0's seq is published LAST
    // (below). Publishing in slot order let a consumer pass slot 0 while slot
    // k+1 was still in flight -> spurious "continuation slot missing" on a
    // normal race.
    if (i > 0) {
      std::atomic_store_explicit(
          reinterpret_cast<std::atomic<uint64_t>*>(&s->seq), seq,
          std::memory_order_release);
    }
  }
  // Batch publish: release the FIRST slot's seq only after every slot of the
  // message is fully written. The consumer's acquire on slot 0
  // synchronizes-with this store, making all writes above (headers, bodies,
  // continuation seqs) visible as one atomic message — its continuation-slot
  // checks then always pass, and "continuation slot missing" once again
  // signals genuine corruption rather than a publish race.
  SlotHeader* first = Slot(first_seq);
  std::atomic_store_explicit(
      reinterpret_cast<std::atomic<uint64_t>*>(&first->seq), first_seq,
      std::memory_order_release);
  header_->head.store(first_seq + num_slots, std::memory_order_release);
}

absl::Status Ring::Read(MsgType* msg_type, std::string* payload) {
  uint64_t seq = header_->tail.load(std::memory_order_relaxed);
  SlotHeader* s = Slot(seq);
  // Non-blocking (spec §3.1): if the next slot hasn't been written yet, return
  // immediately. Callers needing to block must poll at the call site (R5).
  if (std::atomic_load_explicit(
          reinterpret_cast<const std::atomic<uint64_t>*>(&s->seq),
          std::memory_order_acquire) != seq) {
    return absl::NotFoundError("NOT_READY");
  }

  *msg_type = static_cast<MsgType>(s->msg_type);
  payload->clear();
  size_t i = 0;
  while (true) {
    payload->append(SlotBody(s), s->body_len);
    if (!(s->flags & kFlagHasContinuation)) break;
    i++;
    s = Slot(seq + i);
    if (std::atomic_load_explicit(
            reinterpret_cast<const std::atomic<uint64_t>*>(&s->seq),
            std::memory_order_acquire) != seq + i) {
      return absl::InternalError("ring continuation slot missing");
    }
  }
  // Advance tail (release) so the producer sees freed space.
  header_->tail.store(seq + i + 1, std::memory_order_release);
  return absl::OkStatus();
}

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
