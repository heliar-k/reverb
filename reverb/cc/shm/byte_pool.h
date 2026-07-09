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

#ifndef REVERB_CC_SHM_BYTE_POOL_H_
#define REVERB_CC_SHM_BYTE_POOL_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"  // Mutex + CondVar
#include "absl/types/span.h"

namespace deepmind {
namespace reverb {
namespace shm {

// Default slab tiers (bytes). 64B..4MB covers single tensor columns up to
// ~1M float32s. Spec §3.2 / R8.
inline constexpr size_t kDefaultSlabSizes[] = {
    64,      256,     1024,    4096,    16384,
    65536,   262144,  1048576, 4194304,
};
inline constexpr size_t kDefaultBlocksPerSlab = 256;

// Magic + version stored at the head of the pool segment so Open can sanity
// check it mapped the right thing.
inline constexpr uint64_t kPoolMagic = 0x52455652424F4F4CULL;  // "REVRBOOL"
inline constexpr uint32_t kPoolVersion = 1;

// Pool segment layout (single contiguous mmap):
//
//   [PoolHeader]                     // fixed, at offset 0
//   [SlabMeta * slab_count]          // one per tier, lives in SHM
//   [slab 0 data blocks ...]         // block_count * block_size bytes
//   [slab 1 data blocks ...]
//   ...
//
// `SlabMeta` lives in SHM (the free-list head is mutated by the server; the
// client never touches it). Each free block stores the offset of the next free
// block in its first 8 bytes (singly-linked free list). A free_head_offset of
// kNullOffset marks an empty free list.
//
// `Allocate` returns an offset relative to the segment base, so `At(offset) =
// base_ + offset` works identically in the server and client mappings.
//
// refcount is SERVER-PROCESS-ONLY (decision C3/C4: the server is the sole
// allocator; refcount never crosses the boundary). It is a plain in-process
// map, not stored in SHM.

struct PoolHeader {
  uint64_t magic = kPoolMagic;
  uint32_t version = kPoolVersion;
  uint32_t slab_count = 0;
  uint64_t total_size = 0;  // segment bytes
  uint64_t reserved = 0;
};
static_assert(sizeof(PoolHeader) == 32, "PoolHeader must be 32 bytes");

// Per-tier metadata, stored in the SHM segment right after PoolHeader. Only
// the server mutates these (single-threaded: the dispatch thread); the client
// never reads them (it only uses At(offset)).
struct SlabMeta {
  uint64_t block_size = 0;        // tier size in bytes
  uint64_t block_count = 0;       // blocks in this tier
  uint64_t region_offset = 0;     // data region start, relative to base_
  uint64_t free_head_offset = 0;  // offset of first free block (kNullOffset = none)
};
static_assert(sizeof(SlabMeta) == 32, "SlabMeta must be 32 bytes");

inline constexpr uint64_t kNullOffset = UINT64_MAX;  // free-list sentinel

// A slab-allocated POSIX shared-memory byte pool. The server Creates the
// segment (O_CREAT|O_EXCL) and is the sole allocator; clients Open it RW
// (decision C4) and read/write bytes at server-granted offsets via At().
//
// Allocate blocks when the requested tier is exhausted (condvar); a
// Deallocate frees a block and wakes the waiter. Allocate/Deallocate/Ref/
// Unref/ReleaseAll are SERVER-ONLY (single-threaded caller).
class ShmBytePool {
 public:
  ShmBytePool();
  ~ShmBytePool();

  ShmBytePool(const ShmBytePool&) = delete;
  ShmBytePool& operator=(const ShmBytePool&) = delete;
  ShmBytePool(ShmBytePool&&) noexcept;
  ShmBytePool& operator=(ShmBytePool&&) noexcept;

  // Server side: create + truncate a new SHM segment, init slab metadata +
  // free lists. `slab_sizes` defaults to kDefaultSlabSizes when empty.
  static absl::StatusOr<ShmBytePool> Create(
      const std::string& shm_name,
      absl::Span<const size_t> slab_sizes = {},
      size_t blocks_per_slab = kDefaultBlocksPerSlab);

  // Client side: open an existing SHM segment RW (decision C4). Validates
  // magic/version. The client only uses At(offset); it must NOT allocate.
  static absl::StatusOr<ShmBytePool> Open(const std::string& shm_name);

  // Allocate `bytes` from the smallest tier whose block_size >= bytes. Blocks
  // (condvar) if that tier's free list is empty, until a Deallocate frees a
  // block. SERVER-ONLY. Returns the offset relative to the segment base.
  absl::StatusOr<uint64_t> Allocate(size_t bytes);

  // Return `offset` to its tier's free list. SERVER-ONLY. Wakes a blocked
  // Allocate if the tier was full.
  void Deallocate(uint64_t offset);

  // Pointer to the block at `offset` (relative to segment base).
  void* At(uint64_t offset) {
    return static_cast<char*>(base_) + offset;
  }
  const void* At(uint64_t offset) const {
    return static_cast<const char*>(base_) + offset;
  }

  // The block size of the tier that owns `offset` (for tests/diagnostics).
  size_t block_size_at(uint64_t offset) const;

  const std::string& name() const { return shm_name_; }
  size_t size() const { return pool_size_; }

  // Refcount (server-process-only). Ref bumps; Unref returns true when the
  // count hits 0 (caller then Deallocates). Decision C3: server sets refcount=1
  // when it memcpys result bytes into the pool for a sample; the client sends
  // RELEASE after reading; server Unrefs, ->0 deallocates.
  void Ref(uint64_t offset);
  bool Unref(uint64_t offset);

  // Bulk release for crash recovery (ticket ⑥ will use this): decrement each
  // offset's refcount, deallocate those that hit 0.
  void ReleaseAll(const std::vector<uint64_t>& offsets);

 private:
  void Release();

  // Pick the smallest tier index whose block_size >= bytes. Returns slab_count
  // (one past end) if no tier fits.
  size_t PickSlab(size_t bytes) const;

  // Pop the head of a tier's free list; returns kNullOffset if empty.
  uint64_t PopFree(size_t slab_idx);
  void PushFree(size_t slab_idx, uint64_t offset);

  std::string shm_name_;
  void* base_ = nullptr;       // mmap base
  size_t pool_size_ = 0;
  PoolHeader* header_ = nullptr;
  SlabMeta* slabs_ = nullptr;  // slab_count entries, in-segment
  size_t slab_count_ = 0;
  bool owner_ = false;         // true => shm_unlink on destruction

  // ponytail: refcount as a process-local flat_hash_map (NOT in SHM). The
  // server is the sole allocator (C4), so refcount never crosses the process
  // boundary. Upgrade path: an in-SHM per-block refcount array + bitmap if a
  // second allocator ever needs to share it (spec §3.2). Ceiling: O(1) per op,
  // but the map grows/shrinks with outstanding blocks — fine for v1 dispatch
  // thread throughput.
  absl::flat_hash_map<uint64_t, int> refcounts_;

  // Pool-full blocking. Allocate holds this while waiting on a tier's free
  // list; Deallocate signals when it pushes a block back. SERVER-ONLY.
  absl::Mutex mu_;
  absl::CondVar cv_;
};

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SHM_BYTE_POOL_H_
