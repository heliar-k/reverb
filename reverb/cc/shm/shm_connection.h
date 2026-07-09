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
