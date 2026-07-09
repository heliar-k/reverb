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

#include "reverb/cc/shm/ring.h"

namespace deepmind {
namespace reverb {
namespace shm {

// The two SPSC rings wired between a server and one client, plus the pool
// segment name for later use (ticket ② builds the byte pool; ticket ① only
// exercises the rings). Server side owns/creates the segments; client side
// opens them. C2S is written by the client and read by the server; S2C the
// reverse.
//
// ponytail: a plain struct, no factory. The pool handle is deferred to ticket
// ② (ShmBytePool); for now only the name is carried so bootstrap can pass it
// through. Add a ShmBytePool member when the pool exists.
struct ShmConnection {
  Ring c2s;  // client -> server
  Ring s2c;  // server -> client
  std::string pool_shm_name;
};

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SHM_SHM_CONNECTION_H_
