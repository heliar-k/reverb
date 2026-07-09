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

#ifndef REVERB_CC_SHM_SHM_CONNECTION_H_
#define REVERB_CC_SHM_SHM_CONNECTION_H_

#include <string>
#include <utility>

#include "reverb/cc/shm/byte_pool.h"
#include "reverb/cc/shm/ring.h"

namespace deepmind {
namespace reverb {
namespace shm {

// The two SPSC rings wired between a server and one client, plus the shared
// byte pool. Server side owns/creates the segments; client side opens them.
// C2S is written by the client and read by the server; S2C the reverse.
//
// The `pool` handle is only meaningful on the client side (where it is
// `ShmBytePool::Open`'d read/write per decision C4); the server keeps its own
// `ShmBytePool` (the owner/allocator) inside `ShmServer` and does not share it
// through this struct. Both sides read sample bytes via `pool.At(offset)`.
//
// ponytail: a plain struct, no factory. Move-only (Ring/ShmBytePool are
// move-only).
struct ShmConnection {
  Ring c2s;  // client -> server
  Ring s2c;  // server -> client
  ShmBytePool pool;  // client-side RW mapping (C4); server keeps its own
  std::string pool_shm_name;
};

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SHM_SHM_CONNECTION_H_
