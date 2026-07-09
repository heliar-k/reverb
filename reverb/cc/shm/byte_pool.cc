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

#include "reverb/cc/shm/byte_pool.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace deepmind {
namespace reverb {
namespace shm {

namespace {

absl::Status ErrnoStatus(std::string_view op, std::string_view detail) {
  return absl::InternalError(
      absl::StrCat(op, " failed: ", detail, " (errno ", errno, ": ",
                   std::strerror(errno), ")"));
}

}  // namespace

ShmBytePool::ShmBytePool() = default;

ShmBytePool::~ShmBytePool() { Release(); }

ShmBytePool::ShmBytePool(ShmBytePool&& other) noexcept {
  *this = std::move(other);
}

ShmBytePool& ShmBytePool::operator=(ShmBytePool&& other) noexcept {
  if (this != &other) {
    Release();
    shm_name_ = std::move(other.shm_name_);
    base_ = other.base_;
    pool_size_ = other.pool_size_;
    header_ = other.header_;
    slabs_ = other.slabs_;
    slab_count_ = other.slab_count_;
    owner_ = other.owner_;
    refcounts_ = std::move(other.refcounts_);
    other.base_ = nullptr;
    other.header_ = nullptr;
    other.slabs_ = nullptr;
    other.slab_count_ = 0;
    other.pool_size_ = 0;
    other.owner_ = false;
  }
  return *this;
}

void ShmBytePool::Release() {
  if (base_) {
    munmap(base_, pool_size_);
  }
  if (owner_ && !shm_name_.empty()) {
    shm_unlink(shm_name_.c_str());
  }
  base_ = nullptr;
  header_ = nullptr;
  slabs_ = nullptr;
  slab_count_ = 0;
  pool_size_ = 0;
  owner_ = false;
}

// static
absl::StatusOr<ShmBytePool> ShmBytePool::Create(
    const std::string& shm_name, absl::Span<const size_t> slab_sizes,
    size_t blocks_per_slab) {
  // Use the default tiers if the caller passed none. Ponytail: hardcode the
  // default array rather than threading an options struct — one caller (the
  // server) and one default, no polymorphism needed.
  std::vector<size_t> sizes(slab_sizes.begin(), slab_sizes.end());
  if (sizes.empty()) {
    sizes.assign(std::begin(kDefaultSlabSizes), std::end(kDefaultSlabSizes));
  }
  if (sizes.empty() || blocks_per_slab == 0) {
    return absl::InvalidArgumentError("slab_sizes and blocks_per_slab required");
  }

  int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0600);
  if (fd < 0) {
    // R6: stale segment under the same name. Unlink and retry once.
    if (errno == EEXIST) {
      shm_unlink(shm_name.c_str());
      fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0600);
    }
    if (fd < 0) return ErrnoStatus("shm_open", shm_name);
  }

  // Layout: [PoolHeader][SlabMeta * N][slab 0 data][slab 1 data]...
  size_t meta_bytes =
      sizeof(PoolHeader) + sizeof(SlabMeta) * sizes.size();
  size_t data_bytes = 0;
  for (size_t s : sizes) data_bytes += s * blocks_per_slab;
  size_t total = meta_bytes + data_bytes;

  if (ftruncate(fd, total) != 0) {
    auto s = ErrnoStatus("ftruncate", shm_name);
    close(fd);
    shm_unlink(shm_name.c_str());
    return s;
  }

  void* base = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (base == MAP_FAILED) {
    auto s = ErrnoStatus("mmap", shm_name);
    shm_unlink(shm_name.c_str());
    return s;
  }

  ShmBytePool pool;
  pool.shm_name_ = shm_name;
  pool.base_ = base;
  pool.pool_size_ = total;
  pool.owner_ = true;
  pool.header_ = static_cast<PoolHeader*>(base);
  pool.slab_count_ = sizes.size();
  pool.slabs_ = reinterpret_cast<SlabMeta*>(
      static_cast<char*>(base) + sizeof(PoolHeader));

  // Init header.
  pool.header_->magic = kPoolMagic;
  pool.header_->version = kPoolVersion;
  pool.header_->slab_count = static_cast<uint32_t>(sizes.size());
  pool.header_->total_size = total;

  // Init each slab's metadata + free list. The free list links every block in
  // the tier, head -> block 0 -> block 1 -> ... -> kNullOffset. The "next free
  // offset" is stored in the first 8 bytes of each free block.
  uint64_t region = meta_bytes;
  for (size_t i = 0; i < sizes.size(); i++) {
    SlabMeta& m = pool.slabs_[i];
    m.block_size = sizes[i];
    m.block_count = blocks_per_slab;
    m.region_offset = region;
    m.free_head_offset = region;  // first block is the head

    // Chain all blocks: block j's next pointer is block j+1's offset (or
    // kNullOffset for the last).
    for (size_t j = 0; j < blocks_per_slab; j++) {
      uint64_t this_off = region + j * sizes[i];
      uint64_t next_off = (j + 1 < blocks_per_slab)
                              ? (region + (j + 1) * sizes[i])
                              : kNullOffset;
      *reinterpret_cast<uint64_t*>(static_cast<char*>(base) + this_off) =
          next_off;
    }
    region += sizes[i] * blocks_per_slab;
  }

  return pool;
}

// static
absl::StatusOr<ShmBytePool> ShmBytePool::Open(const std::string& shm_name) {
  int fd = shm_open(shm_name.c_str(), O_RDWR, 0600);
  if (fd < 0) return ErrnoStatus("shm_open", shm_name);

  // Read the header to learn the segment geometry before mapping the full size.
  PoolHeader hdr;
  if (read(fd, &hdr, sizeof(hdr)) != sizeof(hdr)) {
    auto s = ErrnoStatus("read header", shm_name);
    close(fd);
    return s;
  }
  if (hdr.magic != kPoolMagic) {
    close(fd);
    return absl::InternalError(absl::StrCat("bad pool magic in ", shm_name));
  }
  if (hdr.version != kPoolVersion) {
    close(fd);
    return absl::InternalError(
        absl::StrCat("pool version mismatch in ", shm_name, ": ", hdr.version,
                     " != ", kPoolVersion));
  }

  size_t total = hdr.total_size;
  void* base = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (base == MAP_FAILED) return ErrnoStatus("mmap", shm_name);

  ShmBytePool pool;
  pool.shm_name_ = shm_name;
  pool.base_ = base;
  pool.pool_size_ = total;
  pool.owner_ = false;  // client never unlinks
  pool.header_ = static_cast<PoolHeader*>(base);
  pool.slab_count_ = hdr.slab_count;
  pool.slabs_ = reinterpret_cast<SlabMeta*>(
      static_cast<char*>(base) + sizeof(PoolHeader));
  return pool;
}

size_t ShmBytePool::PickSlab(size_t bytes) const {
  for (size_t i = 0; i < slab_count_; i++) {
    if (slabs_[i].block_size >= bytes) return i;
  }
  return slab_count_;  // none fits
}

uint64_t ShmBytePool::PopFree(size_t slab_idx) {
  SlabMeta& m = slabs_[slab_idx];
  if (m.free_head_offset == kNullOffset) return kNullOffset;
  uint64_t off = m.free_head_offset;
  // The next-free offset lives in the first 8 bytes of this block.
  m.free_head_offset =
      *reinterpret_cast<uint64_t*>(static_cast<char*>(base_) + off);
  return off;
}

void ShmBytePool::PushFree(size_t slab_idx, uint64_t offset) {
  SlabMeta& m = slabs_[slab_idx];
  // New head points at the old head.
  *reinterpret_cast<uint64_t*>(static_cast<char*>(base_) + offset) =
      m.free_head_offset;
  m.free_head_offset = offset;
}

absl::StatusOr<uint64_t> ShmBytePool::Allocate(size_t bytes) {
  if (bytes == 0) bytes = 1;  // always hand out at least the smallest block
  size_t idx = PickSlab(bytes);
  if (idx >= slab_count_) {
    return absl::InvalidArgumentError(
        absl::StrCat("allocation too large: ", bytes, " bytes"));
  }

  // Block if this tier's free list is empty; Deallocate signals on push.
  // SERVER-ONLY: the dispatch thread is the sole caller.
  mu_.Lock();
  while (slabs_[idx].free_head_offset == kNullOffset) {
    cv_.Wait(&mu_);
  }
  uint64_t off = PopFree(idx);
  mu_.Unlock();

  // Refcount starts implicit at 0; Ref() bumps to 1 when the server marks a
  // block as outstanding (decision C3: sample path sets refcount=1 at memcpy).
  return off;
}

void ShmBytePool::Deallocate(uint64_t offset) {
  // Find the owning slab by region. Ponytail: linear scan over slab_count (<=9
  // tiers). A binary search or an offset->slab index side table would be O(1),
  // but 9 comparisons is already nothing; not worth the bookkeeping.
  size_t idx = 0;
  while (idx + 1 < slab_count_ &&
         slabs_[idx + 1].region_offset <= offset) {
    idx++;
  }

  mu_.Lock();
  bool was_full = (slabs_[idx].free_head_offset == kNullOffset);
  PushFree(idx, offset);
  if (was_full) cv_.Signal();
  mu_.Unlock();
}

size_t ShmBytePool::block_size_at(uint64_t offset) const {
  size_t idx = 0;
  while (idx + 1 < slab_count_ &&
         slabs_[idx + 1].region_offset <= offset) {
    idx++;
  }
  return slabs_[idx].block_size;
}

void ShmBytePool::Ref(uint64_t offset) { refcounts_[offset]++; }

bool ShmBytePool::Unref(uint64_t offset) {
  auto it = refcounts_.find(offset);
  if (it == refcounts_.end()) return true;  // not tracked: treat as zero
  if (--(it->second) <= 0) {
    refcounts_.erase(it);
    return true;
  }
  return false;
}

void ShmBytePool::ReleaseAll(const std::vector<uint64_t>& offsets) {
  for (uint64_t off : offsets) {
    if (Unref(off)) {
      Deallocate(off);
    }
  }
}

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
