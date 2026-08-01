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

#include "reverb/cc/shm/shm_server.h"

#include <csignal>
#include <cstring>
#include <mutex>
#include <poll.h>
#include <sched.h>
#include <string>
#include <sys/mman.h>  // shm_unlink
#include <unistd.h>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/platform/default/hash_map.h"
#include "reverb/cc/platform/default/logging.h"
#include "reverb/cc/platform/default/status_macros.h"
#include "reverb/cc/reverb_service.pb.h"  // ticket ⑩: MutatePrioritiesRequest/ResetRequest
#include "reverb/cc/shm/bootstrap.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/support/trajectory_util.h"
#include "reverb/cc/table.h"
#include "third_party/reverb_tensor/reverb_tensor.pb.h"

namespace deepmind {
namespace reverb {
namespace shm {

namespace {

// ticket ⑥ / R12: signal-driven shutdown. A signal handler may only touch
// async-signal-safe state, so it flips this atomic and the dispatch loop
// exits on its next pass; cleanup then runs via Stop()/~ShmServer (the owner
// calls Stop, or the destructor runs at process exit). ponytail: signal()+an
// atomic, not signalfd/evfd — the minimal correct approach. Ceiling: only one
// ShmServer per process is signal-driven; if multiple need independent
// shutdown, switch to a per-instance self-pipe. Upgrade path noted, not built.
std::atomic<bool> g_signal_stop{false};

void SignalHandler(int) { g_signal_stop.store(true, std::memory_order_relaxed); }

void InstallSignalHandlers() {
  static std::once_flag once;
  std::call_once(once, [] {
    // signal() is not async-signal-safe to call from a handler, but we only
    // call it once at Start; the handler itself only flips an atomic.
    std::signal(SIGTERM, SignalHandler);
    std::signal(SIGINT, SignalHandler);
  });
}

// Build the ShmColumn spec (proto) from a TensorBuffer.
::reverb::tensor::SignatureProto::TensorSpec TensorSpecFromBuffer(
    const TensorBuffer& buf) {
  ::reverb::tensor::SignatureProto::TensorSpec spec;
  spec.set_dtype(DataTypeToProto(buf.dtype()));
  auto* shape = spec.mutable_shape();
  for (int64_t d : buf.shape()) shape->add_dim(d);
  return spec;
}

}  // namespace

ShmServer::ShmServer(std::vector<std::shared_ptr<Table>> tables,
                     std::string socket_path, ShmBytePool pool,
                     ShmBootstrapServer bootstrap,
                     std::shared_ptr<Checkpointer> checkpointer,
                     std::string name_token)
    : socket_path_(std::move(socket_path)),
      name_token_(std::move(name_token)),
      pool_(std::move(pool)),
      bootstrap_(std::move(bootstrap)),
      checkpointer_(std::move(checkpointer)) {
  // ticket ⑨: build the name→Table map. Uniqueness is validated in Create,
  // so here we just move each table into the map by its own name.
  tables_.reserve(tables.size());
  for (auto& t : tables) {
    tables_.emplace(t->name(), std::move(t));
  }
}

// static
absl::StatusOr<std::unique_ptr<ShmServer>> ShmServer::Create(
    std::vector<std::shared_ptr<Table>> tables, const std::string& socket_path,
    std::shared_ptr<Checkpointer> checkpointer) {
  if (tables.empty()) {
    return absl::InvalidArgumentError("tables must not be empty");
  }
  // ticket ⑨: unique-name validation. Python `Server` also checks, but C++
  // defends itself (Create may be called from C++ directly).
  internal::flat_hash_set<std::string> seen;
  seen.reserve(tables.size());
  for (const auto& t : tables) {
    if (t == nullptr) {
      return absl::InvalidArgumentError("table must not be null");
    }
    if (!seen.insert(t->name()).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("Duplicate table name: ", t->name()));
    }
  }
  REVERB_ASSIGN_OR_RETURN(ShmBootstrapServer bootstrap,
                          ShmBootstrapServer::Create(socket_path));

  // The pool name is derived from the socket path (unique per server
  // instance) — NOT the PID alone, which made a second in-process ShmServer
  // unlink the first's live pool (scan #12). ticket #7: plus a per-server
  // epoch (PID + wall-clock nanos), so a crash-restarted server on the same socket
  // path never names a segment a live client still has mapped (Create's
  // EEXIST unlink-and-retry would orphan the client's writes). The client
  // learns every name from Welcome, so only the server needs the formula.
  std::string name_token = absl::StrCat(socket_path, "_", getpid(), "_",
                                        absl::ToUnixNanos(absl::Now()));
  std::string pool_name = MakePoolShmName(name_token);
  REVERB_ASSIGN_OR_RETURN(
      ShmBytePool pool, ShmBytePool::Create(pool_name));
  return absl::WrapUnique(
      new ShmServer(std::move(tables), socket_path, std::move(pool),
                    std::move(bootstrap), std::move(checkpointer),
                    std::move(name_token)));
}

ShmServer::~ShmServer() { Stop(); }

absl::Status ShmServer::Start() {
  if (running_.exchange(true)) {
    return absl::FailedPreconditionError("ShmServer already started");
  }
  InstallSignalHandlers();  // R12: SIGTERM/SIGINT -> graceful shutdown
  dispatch_thread_ = std::thread([this] { DispatchLoop(); });
  return absl::OkStatus();
}

void ShmServer::Stop() {
  if (!running_.exchange(false)) return;
  if (dispatch_thread_.joinable()) dispatch_thread_.join();

  // review #3: drain + join the checkpoint executor before stopping tables —
  // a Save still in flight reads tables_ and stashes its response into a
  // (still-alive) ClientState's outbox. TaskExecutor::Close runs pending
  // tasks on this thread and joins the executor.
  checkpoint_executor_.Close();

  // review #1 (shm-clientstate-lifetime): stop tables BEFORE touching
  // clients_. Table workers + callback executors fire insert/sample
  // completion callbacks that dereference ClientState; Table::Stop() (Close
  // + join worker + drain the callback executor) guarantees none can fire
  // after it returns, so clear()/~ClientState below cannot race a callback.
  // The HandleDisconnect path can't stop the (shared) table per-client — it
  // is covered by the callbacks' shared_ptr<ClientState> capture instead.
  for (const auto& [name, table] : tables_) {
    table->Stop();
  }

  // Clean up clients: ReleaseAll outstanding offsets (C3), close fds. The
  // rings are unlinked by their Ring destructors (owner_=true) when
  // `clients_` clears, so we pass unlink_rings=false to avoid a redundant
  // shm_unlink (the destructor's unlink is the canonical one).
  for (auto& c : clients_) {
    CleanupClient(*c, /*unlink_rings=*/false);
  }
  clients_.clear();
}

void ShmServer::CloseClientFdForTest() {
  // ticket ⑥ test-only (see header). Close client 0's accepted fd from the
  // caller thread, NOT the dispatch thread. The dispatch thread keeps running
  // (it does not own this fd-close); the client's control_fd peer is now gone
  // -> the client's ReadBlocking sees EOF via IsPeerClosed. The dispatch
  // thread's own IsClientDead will also see EOF next pass and run
  // HandleDisconnect, but that races harmlessly with the test's assertions.
  if (!clients_.empty() && clients_[0]->fd >= 0) {
    close(clients_[0]->fd);
    clients_[0]->fd = -1;
  }
}

void ShmServer::CleanupClient(ClientState& state, bool unlink_rings) {
  // C3 crash recovery: centrally release every outstanding pool offset the
  // client never RELEASEd. ReleaseAll decrements refcount and deallocates any
  // that hit 0 (the sample-path bytes the client never read back).
  if (!state.outstanding_offsets_.empty()) {
    pool_.ReleaseAll(
        std::vector<uint64_t>(state.outstanding_offsets_.begin(),
                              state.outstanding_offsets_.end()));
    state.outstanding_offsets_.clear();
  }
  if (unlink_rings) {
    // R6: unlink this client's FOUR ring segments so a restart doesn't see
    // stale segments (decision D: insert + sample pairs). Recompute the names
    // (A3: keyed by server token + client PID). The Ring destructor ALSO
    // unlinks (owner_=true), but explicit unlink here is safe (second unlink
    // is a harmless ENOENT) and makes the cleanup intent obvious at the
    // disconnect site.
    ShmSegmentNames names = MakeShmNames(name_token_, state.client_pid);
    shm_unlink(names.insert_c2s.c_str());
    shm_unlink(names.insert_s2c.c_str());
    shm_unlink(names.sample_c2s.c_str());
    shm_unlink(names.sample_s2c.c_str());
  }
  if (state.fd >= 0) {
    close(state.fd);
    state.fd = -1;
  }
}

void ShmServer::HandleDisconnect(size_t client_id) {
  ClientState& state = *clients_[client_id];
  REVERB_LOG(REVERB_INFO)
      << "ShmServer: disconnecting client " << client_id
      << " (pid " << state.client_pid
      << "): fd EOF/HUP or explicit CLOSE";
  CleanupClient(state, /*unlink_rings=*/true);
  clients_.erase(clients_.begin() + client_id);
}

bool ShmServer::IsClientDead(const ClientState& state) {
  if (state.close_requested) return true;  // explicit CLOSE (ticket ⑥)
  if (state.fd < 0) return true;  // already closed
  // probe the udsocket fd for EOF/HUP. Shares the poll/recv logic with the
  // client-side server-death check (IsPeerClosed, ticket ⑥ spec §8.8): a
  // healthy idle client has the fd open and nothing to send -> poll returns 0
  // (alive); POLLHUP/POLLERR or a 0-byte recv (EOF) => the client is gone.
  return IsPeerClosed(state.fd);
}

void ShmServer::DispatchLoop() {
  while (running_.load() && !g_signal_stop.load(std::memory_order_relaxed)) {
    TryAccept();
    // ticket ⑥: detect dead clients (udsocket EOF/HUP) each pass. Collect
    // dead indices and erase in REVERSE order so earlier indices stay valid
    // (erasing index i would shift i+1.. down). HandleClientRequests/Flush
    // a client AFTER confirming it is alive.
    std::vector<size_t> dead;
    for (size_t i = 0; i < clients_.size(); i++) {
      if (IsClientDead(*clients_[i])) {
        dead.push_back(i);
        continue;  // don't serve a dead client
      }
      // Decision D: drain insert and sample c2s rings independently so a
      // stalled insert request (e.g. waiting on the table worker) does not
      // block sample responses, and vice versa.
      HandleInsertRequests(i);
      HandleSampleRequests(i);
      DrainPendingSamples(*clients_[i]);
      FlushOutbox(*clients_[i]);
    }
    for (auto it = dead.rbegin(); it != dead.rend(); ++it) {
      HandleDisconnect(*it);
    }
    // ponytail: yield + short sleep instead of bare sched_yield. A bare
    // sched_yield does NOT release the core when no other thread is runnable
    // (the common idle case), so the dispatch loop pegged a core at ~100% CPU
    // spinning on poll(timeout=0) — this was the 98% CPU seen in the hang.
    // The 50us sleep releases the core; vs ring ops costing hundreds of us the
    // latency cost is negligible. A blocking poll on all ring fds would be
    // cheaper still but needs eventfd plumbing per ring (upgrade).
    sched_yield();
    usleep(50);
  }
}

bool ShmServer::TryAccept() {
  // Poll the listen fd with zero timeout (non-blocking): only accept when a
  // client is already waiting, so the dispatch loop never stalls on accept.
  struct pollfd pfd;
  pfd.fd = bootstrap_.listen_fd();
  pfd.events = POLLIN;
  if (poll(&pfd, 1, 0) <= 0) return false;  // nothing waiting / error

  auto a = bootstrap_.Accept();
  if (!a.ok()) return false;
  auto [client_fd, client_pid] = std::move(a).value();

  // Bounded handshake: RecvHello runs on the single dispatch thread — an
  // unbounded read lets a connect-and-stall client wedge accept AND service
  // for every client. 250ms is generous for a local udsocket peer (the
  // client writes HELLO immediately after connect); on timeout/malformed we
  // close the fd and keep serving.
  // ponytail: full async accept (pending-hello state machine) would remove
  // even this bounded stall; the 250ms bound caps a stall flood at 4/s.
  auto hello = RecvHello(client_fd, absl::Milliseconds(250));
  if (!hello.ok()) {
    REVERB_LOG(REVERB_WARNING)
        << "ShmServer: closing stalled/malformed HELLO: " << hello.status();
    close(client_fd);
    return false;
  }
  auto version_ok = CheckProtocolVersion(hello->protocol_version());
  if (!version_ok.ok()) {
    close(client_fd);
    return false;
  }

  ShmSegmentNames names = MakeShmNames(name_token_, client_pid);
  // Decision D: create FOUR rings per client — one SPSC pair for the insert
  // flow (TrajectoryWriter) and one for the sample flow (ShmSampler). Each
  // pair keeps the SPSC invariant intact (one client-thread producer per c2s)
  // so the two background workers never race a shared ring. (The pool was
  // created once at server init; its name is shared with all clients.)
  auto ins_c2s = Ring::Create(names.insert_c2s);
  if (!ins_c2s.ok()) {
    close(client_fd);
    return false;
  }
  auto ins_s2c = Ring::Create(names.insert_s2c);
  if (!ins_s2c.ok()) {
    close(client_fd);
    return false;
  }
  auto smp_c2s = Ring::Create(names.sample_c2s);
  if (!smp_c2s.ok()) {
    close(client_fd);
    return false;
  }
  auto smp_s2c = Ring::Create(names.sample_s2c);
  if (!smp_s2c.ok()) {
    close(client_fd);
    return false;
  }

  WelcomeResponse welcome;
  welcome.set_pool_shm_name(pool_.name());
  // Deprecated aliases mirror the sample flow for legacy readers.
  welcome.set_c2s_shm_name(names.sample_c2s);
  welcome.set_s2c_shm_name(names.sample_s2c);
  welcome.set_insert_c2s_shm_name(names.insert_c2s);
  welcome.set_insert_s2c_shm_name(names.insert_s2c);
  welcome.set_sample_c2s_shm_name(names.sample_c2s);
  welcome.set_sample_s2c_shm_name(names.sample_s2c);
  // ticket ⑧ step 1: piggyback TableInfo on the bootstrap handshake so the
  // client can serve server_info() from a bootstrap snapshot — no new
  // SERVER_INFO ring round-trip (that is step 2, deferred). ponytail:
  // bootstrap-time snapshot only; mid-session Table.replace / signature
  // changes are NOT reflected here until step 2 lands. Upgrade path: a
  // SERVER_INFO/SERVER_INFO_RESP MsgType + on-demand round-trip.
  // ticket ⑨: list ALL tables (was: just table_->info()). hash_map order is
  // unspecified; the client consumes server_info into a name→TableInfo dict.
  auto* server_info = welcome.mutable_server_info();
  for (const auto& [name, table] : tables_) {
    *server_info->add_table_info() = table->info();
  }
  auto send = SendWelcome(client_fd, welcome);
  if (!send.ok()) {
    close(client_fd);
    return false;
  }

  auto state = std::make_shared<ClientState>();
  state->fd = client_fd;
  state->client_pid = client_pid;
  state->conn.insert_c2s = std::move(*ins_c2s);
  state->conn.insert_s2c = std::move(*ins_s2c);
  state->conn.sample_c2s = std::move(*smp_c2s);
  state->conn.sample_s2c = std::move(*smp_s2c);
  state->conn.pool_shm_name = pool_.name();
  // The server keeps its own owner pool (`pool_`); the client-side pool handle
  // in `conn` is left default (the server never reads sample bytes via it).
  clients_.push_back(std::move(state));
  return true;
}

void ShmServer::HandleInsertRequests(size_t client_id) {
  ClientState& state = *clients_[client_id];
  // Drain everything currently readable from the INSERT flow's c2s ring
  // (non-blocking). Ring::Read returns NotFound("NOT_READY") when empty
  // (spec §3.1) — that is the normal "no work" signal, not an error.
  while (running_.load()) {
    MsgType type;
    std::string payload;
    absl::Status s = state.conn.insert_c2s.Read(&type, &payload);
    if (!s.ok()) {
      if (!absl::IsNotFound(s)) {
        REVERB_LOG(REVERB_WARNING)
            << "ShmServer: insert c2s read error for client " << client_id
            << ": " << s;
      }
      return;
    }
    switch (type) {
      case INSERT: {
        ShmInsertRequest req;
        if (!req.ParseFromString(payload)) {
          REVERB_LOG(REVERB_WARNING)
              << "ShmServer: malformed ShmInsertRequest from client "
              << client_id;
          break;
        }
        auto st = HandleInsert(clients_[client_id], req);
        if (!st.ok()) {
          REVERB_LOG(REVERB_WARNING)
              << "ShmServer: HandleInsert failed for client " << client_id
              << ": " << st;
        }
        break;
      }
      case ALLOCATE: {
        ShmAllocateRequest req;
        if (!req.ParseFromString(payload)) {
          REVERB_LOG(REVERB_WARNING)
              << "ShmServer: malformed ShmAllocateRequest from client "
              << client_id;
          break;
        }
        auto st = HandleAllocate(state, req);
        if (!st.ok()) {
          REVERB_LOG(REVERB_WARNING)
              << "ShmServer: HandleAllocate failed for client " << client_id
              << ": " << st;
        }
        break;
      }
      case RELEASE: {
        // RELEASE may arrive on either flow (insert chunks vs sample bytes).
        // HandleRelease is offset-keyed, flow-agnostic.
        ShmReleaseRequest req;
        if (!req.ParseFromString(payload)) break;
        auto st = HandleRelease(state, req);
        if (!st.ok()) {
          REVERB_LOG(REVERB_WARNING)
              << "ShmServer: HandleRelease failed for client " << client_id
              << ": " << st;
        }
        break;
      }
      case CLOSE:
        // ticket ⑥: graceful close. Mark for disconnect; the dispatch loop's
        // dead-collection erases it next pass (avoid erasing mid-iteration).
        state.close_requested = true;
        return;
      case MUTATE_PRIORITIES: {
        // ticket ⑩: control-plane rides the insert flow (the client holds
        // insert_flow_mu across the send→read-ACK pair, so this is drained in
        // order between RunShmWorker's INSERT round-trips).
        MutatePrioritiesRequest req;
        if (!req.ParseFromString(payload)) {
          REVERB_LOG(REVERB_WARNING)
              << "ShmServer: malformed MutatePrioritiesRequest from client "
              << client_id;
          break;
        }
        auto st = HandleMutatePriorities(state, req);
        if (!st.ok()) {
          REVERB_LOG(REVERB_WARNING)
              << "ShmServer: HandleMutatePriorities failed for client "
              << client_id << ": " << st;
        }
        break;
      }
      case RESET: {
        ResetRequest req;
        if (!req.ParseFromString(payload)) {
          REVERB_LOG(REVERB_WARNING)
              << "ShmServer: malformed ResetRequest from client " << client_id;
          break;
        }
        auto st = HandleReset(state, req);
        if (!st.ok()) {
          REVERB_LOG(REVERB_WARNING)
              << "ShmServer: HandleReset failed for client " << client_id
              << ": " << st;
        }
        break;
      }
      case CHECKPOINT: {
        // ticket ⑪: checkpoint rides the insert flow like ⑩'s control-plane
        // ops. CheckpointRequest is empty; HandleCheckpoint returns the path
        // in CHECKPOINT_RESP (or ShmError on failure).
        // review #3: HandleCheckpoint only SCHEDULES the Save (on
        // checkpoint_executor_) and returns immediately — the response comes
        // back out-of-band via the client's insert_outbox.
        auto st = HandleCheckpoint(clients_[client_id]);
        if (!st.ok()) {
          REVERB_LOG(REVERB_WARNING)
              << "ShmServer: HandleCheckpoint failed for client " << client_id
              << ": " << st;
        }
        break;
      }
      case SERVER_INFO: {
        // ticket ⑧ step 2: on-demand server_info round-trip. Request body is
        // empty (like CheckpointRequest); HandleServerInfo returns the live
        // ServerInfoResponse in SERVER_INFO_RESP.
        auto st = HandleServerInfo(state);
        if (!st.ok()) {
          REVERB_LOG(REVERB_WARNING)
              << "ShmServer: HandleServerInfo failed for client " << client_id
              << ": " << st;
        }
        break;
      }
      default:
        // Unknown msg type on this flow: ignore (forward-compat).
        break;
    }
  }
}

void ShmServer::HandleSampleRequests(size_t client_id) {
  ClientState& state = *clients_[client_id];
  // Drain the SAMPLE flow's c2s ring (non-blocking). Decision D: separate
  // from the insert ring so a slow insert does not delay samples.
  while (running_.load()) {
    MsgType type;
    std::string payload;
    absl::Status s = state.conn.sample_c2s.Read(&type, &payload);
    if (!s.ok()) {
      if (!absl::IsNotFound(s)) {
        REVERB_LOG(REVERB_WARNING)
            << "ShmServer: sample c2s read error for client " << client_id
            << ": " << s;
      }
      return;
    }
    switch (type) {
      case SAMPLE: {
        ShmSampleRequest req;
        if (!req.ParseFromString(payload)) {
          REVERB_LOG(REVERB_WARNING)
              << "ShmServer: malformed ShmSampleRequest from client "
              << client_id;
          break;
        }
        auto st = HandleSample(clients_[client_id], req);
        if (!st.ok()) {
          REVERB_LOG(REVERB_WARNING)
              << "ShmServer: HandleSample failed for client " << client_id
              << ": " << st;
        }
        break;
      }
      case RELEASE: {
        ShmReleaseRequest req;
        if (!req.ParseFromString(payload)) break;
        auto st = HandleRelease(state, req);
        if (!st.ok()) {
          REVERB_LOG(REVERB_WARNING)
              << "ShmServer: HandleRelease failed for client " << client_id
              << ": " << st;
        }
        break;
      }
      case CLOSE:
        state.close_requested = true;
        return;
      default:
        break;
    }
  }
}

void ShmServer::FlushOutbox(ClientState& state) {
  // Decision D: flush each flow's outbox to its own s2c ring so a full insert
  // ring does not block sample responses (and vice versa).
  absl::MutexLock ins_lock(&state.insert_outbox_mu);
  while (!state.insert_outbox.empty()) {
    auto& [type, body] = state.insert_outbox.front();
    absl::Status s =
        state.conn.insert_s2c.TryWrite(static_cast<MsgType>(type),
                                       absl::MakeSpan(body));
    if (!s.ok()) {
      // Still full (or oversized) — leave the front in place and try again
      // next pass (§8.7). ponytail: a full outbox eventually backpressures via
      // pool exhaustion; v1 does not bound the outbox size. Upgrade:
      // ring-buffer + drop policy.
      break;
    }
    state.insert_outbox.erase(state.insert_outbox.begin());
  }
  absl::MutexLock smp_lock(&state.sample_outbox_mu);
  while (!state.sample_outbox.empty()) {
    auto& [type, body] = state.sample_outbox.front();
    absl::Status s =
        state.conn.sample_s2c.TryWrite(static_cast<MsgType>(type),
                                       absl::MakeSpan(body));
    if (!s.ok()) {
      break;
    }
    state.sample_outbox.erase(state.sample_outbox.begin());
  }
}

namespace {
// TryWrite's InvalidArgument means the response can NEVER fit the ring.
// Stashing it for retry would spin forever (a >capacity SAMPLE_RESP /
// SERVER_INFO_RESP used to hang the client until its 60s cap). Send a small
// ERROR instead so the client fails fast.
absl::Status EnqueueOversizeError(
    Ring* ring, std::vector<std::pair<uint16_t, std::string>>* outbox,
    absl::Mutex* mu, MsgType type, size_t body_size) {
  REVERB_LOG(REVERB_WARNING)
      << "ShmServer: response type " << type << " (" << body_size
      << " bytes) exceeds ring capacity; sending ERROR instead";
  ShmError err;
  err.set_code(ShmError::INTERNAL);
  err.set_message(absl::StrCat("ShmServer: response type ", type,
                               " too large for SHM ring (", body_size,
                               " bytes)"));
  std::string ebody;
  err.SerializeToString(&ebody);
  absl::Status s = ring->TryWrite(ERROR, absl::MakeSpan(ebody));
  if (absl::IsResourceExhausted(s)) {
    absl::MutexLock lock(mu);
    outbox->emplace_back(static_cast<uint16_t>(ERROR), std::move(ebody));
    return absl::OkStatus();
  }
  return s;
}
}  // namespace

absl::Status ShmServer::EnqueueInsertS2C(ClientState& state, MsgType type,
                                         absl::string_view body) {
  // Non-blocking write on the INSERT flow's s2c ring (spec §8.7). TryWrite
  // returns ResourceExhausted("RING_FULL") immediately when full — we stash
  // in insert_outbox and FlushOutbox retries each dispatch pass.
  absl::Status s = state.conn.insert_s2c.TryWrite(type, absl::MakeSpan(body));
  if (s.ok()) return absl::OkStatus();
  if (absl::IsInvalidArgument(s)) {
    // PERMANENT: the message can never fit the ring. Stashing it would retry
    // forever (the old RING_FULL/oversize confusion). Send a small ERROR
    // instead so the client fails fast instead of hanging to its 60s cap.
    return EnqueueOversizeError(&state.conn.insert_s2c,
                                &state.insert_outbox, &state.insert_outbox_mu,
                                type, body.size());
  }
  if (!absl::IsResourceExhausted(s)) {
    return s;  // genuine error
  }
  {
    absl::MutexLock lock(&state.insert_outbox_mu);
    state.insert_outbox.emplace_back(static_cast<uint16_t>(type),
                                     std::string(body));
  }
  return absl::OkStatus();
}

absl::Status ShmServer::EnqueueSampleS2C(ClientState& state, MsgType type,
                                         absl::string_view body) {
  // Non-blocking write on the SAMPLE flow's s2c ring.
  absl::Status s = state.conn.sample_s2c.TryWrite(type, absl::MakeSpan(body));
  if (s.ok()) return absl::OkStatus();
  if (absl::IsInvalidArgument(s)) {
    // PERMANENT: see EnqueueInsertS2C.
    return EnqueueOversizeError(&state.conn.sample_s2c,
                                &state.sample_outbox, &state.sample_outbox_mu,
                                type, body.size());
  }
  if (!absl::IsResourceExhausted(s)) {
    return s;
  }
  {
    absl::MutexLock lock(&state.sample_outbox_mu);
    state.sample_outbox.emplace_back(static_cast<uint16_t>(type),
                                     std::string(body));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<Table>> ShmServer::FindTable(
    const std::string& name) const {
  // ticket ⑨: shared table-name routing; ⑩ (mutate_priorities/reset) reuses
  // this seam. Returns NotFoundError on miss; callers map that to a ShmError.
  auto it = tables_.find(name);
  if (it == tables_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Unknown table: ", name));
  }
  return it->second;
}

absl::Status ShmServer::HandleSample(std::shared_ptr<ClientState> state_sp,
                                     const ShmSampleRequest& req) {
  ClientState& state = *state_sp;
  // 0. ticket ⑨: route by table name. Unknown table -> ShmError::NOT_FOUND on
  //    the sample s2c flow (mirrors the DEADLINE_EXCEEDED error path below);
  //    the client's FetchOne maps it to absl::NotFoundError.
  auto table_or = FindTable(req.table());
  if (!table_or.ok()) {
    ShmError err;
    err.set_code(ShmError::NOT_FOUND);
    err.set_message(std::string(table_or.status().message()));
    std::string body;
    err.SerializeToString(&body);
    return EnqueueSampleS2C(state, ERROR, body);
  }
  std::shared_ptr<Table> table = *std::move(table_or);

  // ticket ⑩ 死锁修复（方向 A）：异步采样。原来的 table->Sample() 同步调用会在
  // rate limiter 上阻塞整个 dispatch 线程（timeout_ms<0 时无限等），造成队头阻塞——
  // 该 client 的所有后续 insert/mutate/sample ACK 永远排不进 ring，client
  // ReadBlocking 100% CPU 忙等、timeout 杀不掉。gdb 抓栈确认（见 ⑩ 「已确认根因」）。
  //
  // 改用 EnqueSampleRequest 入 table worker 异步队列，dispatch 立即返回不阻塞。
  // 完成回调在 table worker 线程触发，不能直接碰 pool_/outstanding_offsets_
  // （单线程 dispatch 不变式），故把 SampledItem + status 攒进
  // ClientState::pending_samples，dispatch 线程每轮 DrainPendingSamples 取出做
  // unpack+pool+SAMPLE_RESP。镜像 HandleInsert 的 callback→dispatch-drain 模式。
  //
  // timeout 传给 EnqueSampleRequest（table worker 的 GetExpiredRequests 处理超时，
  // callback 收到 DeadlineExceeded status）。timeout_ms<0 仍映射 InfiniteDuration
  // ——但现在阻塞发生在 table worker 线程，不卡 dispatch。
  absl::Duration timeout = (req.timeout_ms() < 0)
                               ? absl::InfiniteDuration()
                               : absl::Milliseconds(req.timeout_ms());
  // review #1: the callback captures state_sp (shared_ptr<ClientState>),
  // not a raw pointer — the state then stays alive until the callback
  // returns even if clients_.erase()/clear() ran first (HandleDisconnect /
  // Stop racing an in-flight callback).
  ShmSampleRequest req_copy = req;  // 回调攥住请求（table 名等在回调里不再用，
                                    // 但保留以备将来按请求配对）
  // Keepalive + 自清：shared_ptr<SamplingCallback> 存 pending_sample_callbacks
  //（入队时 push_back，使 EnqueSampleRequest 的 weak_ptr 可 lock）。回调触发时
  // 按裸指针 key 从 vector erase 自己的 shared_ptr 拷贝。
  //
  // 裸指针获取的 bootstrap 问题：cb_raw 在 make_shared 后才有值，lambda 捕获在
  // make_shared 时冻结，按值捕获会冻结为 nullptr，按引用捕获局部变量会悬空
  //（HandleSample 返回后栈帧销毁）。解法：把裸指针存在堆上 shared_ptr 控制块里
  //（cb_raw_box），按值捕获该 shared_ptr；make_shared 后赋值 *cb_raw_box =
  // cb.get()，lambda 触发时读到正确值。
  //
  // 安全性：erase 发生在回调体内，此时 FinalizeSampleRequest 持另一份 shared_ptr
  // 拷贝（来自 weak_ptr.lock()），erase vector 里的拷贝不会析构正在执行的回调
  // 对象。erase 后只剩 FinalizeSampleRequest 的拷贝，回调返回后拷贝销毁、对象析构。
  //
  // 这比「FIFO 弹出」正确——table worker 可能乱序完成请求（rate limiter 不满足
  // 时把请求放回 current_sampling，后入队的可能先完成），FIFO 会弹错 keepalive
  // 导致 weak_ptr.lock() 失败、callback 不触发、sample 永远不回来。
  // ponytail: O(n) scan erase，n=in-flight sample 数（受 sampler 队列容量限，<=8）。
  // Ceil: 高并发可改 hash_set 按指针查。Upgrade: 同。
  auto cb_raw_box = std::make_shared<Table::SamplingCallback*>();
  auto cb = std::make_shared<Table::SamplingCallback>(
      [state_sp, cb_raw_box, req_copy = std::move(req_copy)](
          Table::SampleRequest* sample) mutable {
        ClientState::PendingSample ps;
        ps.req = std::move(req_copy);
        ps.status = sample->status;
        if (sample->status.ok() && !sample->samples.empty()) {
          ps.item = std::move(sample->samples.front());
        }
        absl::MutexLock lock(&state_sp->pending_samples_mu);
        state_sp->pending_samples.push_back(std::move(ps));
        // 自清 keepalive：erase 指向自己的 shared_ptr 拷贝。
        Table::SamplingCallback* raw = *cb_raw_box;
        auto& cbs = state_sp->pending_sample_callbacks;
        for (auto it = cbs.begin(); it != cbs.end(); ++it) {
          if (it->get() == raw) {
            cbs.erase(it);
            break;
          }
        }
      });
  *cb_raw_box = cb.get();  // make_shared 后赋值，lambda 按值捕获 cb_raw_box 读到此值
  {
    absl::MutexLock lock(&state.pending_samples_mu);
    state.pending_sample_callbacks.push_back(cb);
  }
  table->EnqueSampleRequest(/*num_samples=*/1, cb, timeout);
  return absl::OkStatus();
}

void ShmServer::DrainPendingSamples(ClientState& state) {
  // ticket ⑩ 死锁修复（方向 A）：在 dispatch 线程消费异步完成的 sample。取出
  // pending_samples，每个做 unpack+pool memcpy+写 SAMPLE_RESP（或写 ERROR on
  // 失败/超时）。pool_/outstanding_offsets_ 仍只被 dispatch 线程碰，不变式保持。
  //
  // keepalive 清理：回调触发时按裸指针 key 自清 erase 自己的 shared_ptr 拷贝
  //（见 HandleSample 的 cb_raw_box 模式）。这里只 drain pending_samples。
  std::vector<ClientState::PendingSample> done;
  {
    absl::MutexLock lock(&state.pending_samples_mu);
    done.swap(state.pending_samples);
  }

  for (auto& ps : done) {
    // 失败/超时 → ShmError on sample s2c（镜像旧路径）。
    if (!ps.status.ok()) {
      ShmError err;
      err.set_code(absl::IsDeadlineExceeded(ps.status) ? ShmError::DEADLINE_EXCEEDED
                                                        : ShmError::INTERNAL);
      err.set_message(std::string(ps.status.message()));
      std::string body;
      err.SerializeToString(&body);
      (void)EnqueueSampleS2C(state, ERROR, body);
      continue;
    }
    // 成功：unpack + pool + SAMPLE_RESP。以下与原同步 HandleSample 第 2-3 步同。
    const Table::SampledItem& item = ps.item;
    if (item.ref == nullptr) {
      // 空采样项（不应发生，但防御）→ INTERNAL error。
      ShmError err;
      err.set_code(ShmError::INTERNAL);
      err.set_message("ShmServer: async sample returned empty item");
      std::string body;
      err.SerializeToString(&body);
      (void)EnqueueSampleS2C(state, ERROR, body);
      continue;
    }
    // 2. Unpack each column on the dispatch thread (A1: single-threaded alloc).
    //    Mirror AsSample (sampler.cc): walk flat_trajectory().columns(), for each
    //    chunk slice call UnpackChunkColumnAndSlice. Then concatenate the slices
    //    of a column into one TensorBuffer (the trajectory view), applying the
    //    squeeze flag — exactly as Sample::AsTrajectory does.
    internal::flat_hash_map<uint64_t, std::shared_ptr<ChunkStore::Chunk>> chunks;
    for (auto& chunk : item.ref->chunks()) chunks[chunk->key()] = chunk;

    ShmSampleResponse resp;
    ShmSample* sample = resp.add_samples();

    // SampleInfo: key, priority, times_sampled, probability, table_size,
    // rate_limited — mirrors AsSample (sampler.cc).
    auto* info = sample->mutable_info();
    info->mutable_item()->set_key(item.ref->key());
    info->mutable_item()->set_priority(item.priority);
    info->mutable_item()->set_times_sampled(item.times_sampled);
    info->set_probability(item.probability);
    info->set_table_size(item.table_size);
    info->set_rate_limited(item.rate_limited);

    const auto& columns = item.ref->flat_trajectory().columns();
    bool unpack_ok = true;
    absl::Status unpack_status;
    for (int ci = 0; ci < columns.size(); ci++) {
      const auto& column = columns[ci];
      // Gather the column's unpacked slices.
      std::vector<TensorBuffer> slices;
      slices.reserve(column.chunk_slices_size());
      for (const auto& slice : column.chunk_slices()) {
        auto it = chunks.find(slice.chunk_key());
        if (it == chunks.end()) {
          unpack_status = absl::InternalError(absl::StrCat(
              "ShmServer: chunk ", slice.chunk_key(),
              " not found when unpacking item ", item.ref->key()));
          unpack_ok = false;
          break;
        }
        slices.emplace_back();
        auto s = internal::UnpackChunkColumnAndSlice(
            it->second->data(), slice, &slices.back());
        if (!s.ok()) {
          unpack_status = s;
          unpack_ok = false;
          break;
        }
      }
      if (!unpack_ok) break;

      // Concatenate the column's slices into one batched trajectory tensor
      // (matches Sample::AsTrajectory: single-slice columns move, multi-slice
      // Concat along dim 0). The squeeze is applied CLIENT-side by
      // Sample::AsTrajectory (we send the batched tensor + the squeeze flag),
      // so the server and client mirror the local AsSample path exactly.
      TensorBuffer column_tensor;
      if (slices.size() == 1) {
        column_tensor = std::move(slices[0]);
      } else {
        auto concat = TensorBuffer::Concat(slices);
        if (!concat.status().ok()) {
          unpack_status = concat.status();
          unpack_ok = false;
          break;
        }
        column_tensor = *std::move(concat);
      }

      bool squeeze = column.squeeze();

      // 3. memcpy the result bytes into the pool (C3: refcount starts at 1).
      absl::string_view bytes = column_tensor.bytes();
      auto offset_or = pool_.Allocate(bytes.size());
      if (!offset_or.ok()) {
        unpack_status = offset_or.status();
        unpack_ok = false;
        break;
      }
      uint64_t offset = *offset_or;
      pool_.Ref(offset);  // C3: refcount = 1 for the outstanding sample bytes
      std::memcpy(pool_.At(offset), bytes.data(), bytes.size());

      state.outstanding_offsets_.insert(offset);

      ShmColumn* col = sample->add_columns();
      col->set_shm_offset(offset);
      col->set_length(bytes.size());
      *col->mutable_spec() = TensorSpecFromBuffer(column_tensor);
      col->set_squeeze(squeeze);
    }

    if (!unpack_ok) {
      // 回收本轮已 memcpy 进池的列块:它们以 refcount=1 挂在
      // outstanding_offsets_ 上,不回收会泄漏到客户端断连(喂养池耗尽)。
      for (const ShmColumn& col : sample->columns()) {
        if (state.outstanding_offsets_.erase(col.shm_offset()) > 0) {
          if (pool_.Unref(col.shm_offset())) {  // ->0
            pool_.Deallocate(col.shm_offset());
          }
        }
      }
      // unpack/alloc 失败：回收本轮已分配的 offset，写 ERROR。
      ShmError err;
      err.set_code(ShmError::INTERNAL);
      err.set_message(std::string(unpack_status.message()));
      std::string body;
      err.SerializeToString(&body);
      (void)EnqueueSampleS2C(state, ERROR, body);
      REVERB_LOG(REVERB_WARNING) << "ShmServer: drain sample failed: "
                                 << unpack_status;
      continue;
    }

    std::string body;
    resp.SerializeToString(&body);
    (void)EnqueueSampleS2C(state, SAMPLE_RESP, body);
  }
}

absl::Status ShmServer::HandleRelease(ClientState& state,
                                      const ShmReleaseRequest& req) {
  for (uint64_t offset : req.offsets()) {
    // Only release offsets this server granted to THIS client. Deallocating
    // an arbitrary offset corrupts the slab free lists (a double RELEASE
    // hands the same block out twice; offset 0 would free the PoolHeader).
    if (state.outstanding_offsets_.erase(offset) == 0) {
      REVERB_LOG(REVERB_WARNING)
          << "ShmServer: ignoring RELEASE of untracked offset " << offset;
      continue;
    }
    if (pool_.Unref(offset)) {  // ->0
      pool_.Deallocate(offset);
    }
  }
  return absl::OkStatus();
}

absl::Status ShmServer::HandleAllocate(ClientState& state,
                                       const ShmAllocateRequest& req) {
  // C4: the server is the sole pool allocator. Grant the offset, track it as
  // outstanding (refcount=1) so a client crash (⑥) reclaims it, and reply with
  // ALLOCATE_RESP. The client memcpy's insert bytes here, then sends INSERT and
  // waits for InsertAck.offsets_to_release before reusing the region (C2).
  // Pool exhaustion / oversize must reach the client as an ERROR — returning
  // a bare status only gets logged by the dispatcher and the writer hangs
  // until its timeout cap. (Exhaustion can't block-wait anymore: the dispatch
  // thread is the sole Deallocate caller, so waiting would deadlock.)
  auto offset_or = pool_.Allocate(req.num_bytes());
  if (!offset_or.ok()) {
    ShmError err;
    err.set_code(absl::IsResourceExhausted(offset_or.status())
                     ? ShmError::RESOURCE_EXHAUSTED
                     : ShmError::INVALID_ARGUMENT);
    err.set_message(std::string(offset_or.status().message()));
    std::string body;
    err.SerializeToString(&body);
    return EnqueueInsertS2C(state, ERROR, body);
  }
  uint64_t offset = *offset_or;
  pool_.Ref(offset);  // outstanding against client crash
  state.outstanding_offsets_.insert(offset);
  ShmAllocateResponse resp;
  resp.set_shm_offset(offset);
  std::string body;
  resp.SerializeToString(&body);
  return EnqueueInsertS2C(state, ALLOCATE_RESP, body);
}

absl::Status ShmServer::HandleMutatePriorities(
    ClientState& state, const MutatePrioritiesRequest& req) {
  // ticket ⑩: route by table name via FindTable (ticket ⑨'s seam). Unknown
  // table -> ShmError::NOT_FOUND on the insert s2c flow; the client maps that
  // to absl::NotFoundError -> Python FileNotFoundError. Mirrors InProcessClient
  // / Client::MutatePriorities (table->MutateItems). The MutatePriorities-
  // Response is empty; we send an empty MUTATE_ACK.
  auto table_or = FindTable(req.table());
  if (!table_or.ok()) {
    ShmError err;
    err.set_code(ShmError::NOT_FOUND);
    err.set_message(std::string(table_or.status().message()));
    std::string body;
    err.SerializeToString(&body);
    return EnqueueInsertS2C(state, ERROR, body);
  }
  // Materialize the repeated fields into locals so the spans point at stable
  // storage (a span over a temporary vector would dangle).
  std::vector<KeyWithPriority> updates(req.updates().begin(), req.updates().end());
  std::vector<uint64_t> deletes(req.delete_keys().begin(),
                                req.delete_keys().end());
  absl::Status s = (*table_or)->MutateItems(absl::MakeConstSpan(updates),
                                            absl::MakeConstSpan(deletes));
  if (!s.ok()) {
    ShmError err;
    err.set_code(ShmError::INTERNAL);
    err.set_message(std::string(s.message()));
    std::string body;
    err.SerializeToString(&body);
    return EnqueueInsertS2C(state, ERROR, body);
  }
  return EnqueueInsertS2C(state, MUTATE_ACK, "");  // empty ack
}

absl::Status ShmServer::HandleReset(ClientState& state,
                                    const ResetRequest& req) {
  // ticket ⑩: route by table name; unknown table -> NOT_FOUND. ResetResponse
  // is empty; we send an empty RESET_ACK.
  auto table_or = FindTable(req.table());
  if (!table_or.ok()) {
    ShmError err;
    err.set_code(ShmError::NOT_FOUND);
    err.set_message(std::string(table_or.status().message()));
    std::string body;
    err.SerializeToString(&body);
    return EnqueueInsertS2C(state, ERROR, body);
  }
  absl::Status s = (*table_or)->Reset();
  if (!s.ok()) {
    ShmError err;
    err.set_code(ShmError::INTERNAL);
    err.set_message(std::string(s.message()));
    std::string body;
    err.SerializeToString(&body);
    return EnqueueInsertS2C(state, ERROR, body);
  }
  return EnqueueInsertS2C(state, RESET_ACK, "");  // empty ack
}

absl::Status ShmServer::HandleCheckpoint(std::shared_ptr<ClientState> state) {
  // ticket ⑪: cross-table checkpoint, mirroring InProcessClient::Checkpoint /
  // ReverbServiceImpl::Checkpoint. No checkpointer -> FailedPreconditionError
  // surfaced as ShmError::INTERNAL (the ShmError::Code enum has no
  // FAILED_PRECONDITION; INTERNAL is the catch-all the client maps to
  // InternalError -> Python RuntimeError, close enough to LocalClient's
  // FailedPreconditionError surfacing). CheckpointResponse carries the path.
  if (checkpointer_ == nullptr) {
    ShmError err;
    err.set_code(ShmError::INTERNAL);
    err.set_message("ShmServer: no checkpointer provided");
    std::string body;
    err.SerializeToString(&body);
    return EnqueueInsertS2C(*state, ERROR, body);
  }
  // review #3: Save is unbounded disk I/O — schedule it on the checkpoint
  // executor and return immediately so a slow checkpoint never
  // head-of-line-blocks the dispatch thread. The response is stashed in the
  // client's insert_outbox (mutex-protected, off-dispatch-thread safe — the
  // same callback→drain pattern as async insert ACKs) and flushed next pass.
  checkpoint_executor_.Schedule([this, state] {
    std::vector<Table*> raw_tables;
    raw_tables.reserve(tables_.size());
    for (auto& [_, table] : tables_) {
      raw_tables.push_back(table.get());
    }
    CheckpointResponse resp;
    absl::Status s =
        checkpointer_->Save(std::move(raw_tables), /*keep_latest=*/1,
                            resp.mutable_checkpoint_path());
    uint16_t type;
    std::string body;
    if (s.ok()) {
      type = static_cast<uint16_t>(CHECKPOINT_RESP);
      resp.SerializeToString(&body);
    } else {
      type = static_cast<uint16_t>(ERROR);
      ShmError err;
      err.set_code(ShmError::INTERNAL);
      err.set_message(std::string(s.message()));
      err.SerializeToString(&body);
    }
    absl::MutexLock lock(&state->insert_outbox_mu);
    state->insert_outbox.emplace_back(type, std::move(body));
  });
  return absl::OkStatus();
}

absl::Status ShmServer::HandleServerInfo(ClientState& state) {
  // ticket ⑧ step 2: on-demand server_info round-trip. Gathers each table's
  // live TableInfo (current_size / signature / rate_limiter_info reflect
  // mid-session state, unlike the step-1 bootstrap snapshot) into a
  // ServerInfoResponse and returns it as SERVER_INFO_RESP on the insert s2c
  // flow. Cross-table and always succeeds — no FindTable routing, no error
  // path. Mirrors TryAccept's welcome.server_info fill but on demand.
  ServerInfoResponse resp;
  for (const auto& [name, table] : tables_) {
    *resp.add_table_info() = table->info();
  }
  std::string body;
  resp.SerializeToString(&body);
  return EnqueueInsertS2C(state, SERVER_INFO_RESP, body);
}

absl::Status ShmServer::HandleInsert(std::shared_ptr<ClientState> state_sp,
                                      const ShmInsertRequest& req) {
  ClientState& state = *state_sp;
  // 1. Deserialize each referenced ChunkData from the pool (C4: client wrote
  //    serialized bytes at the granted offset). ChunkData is self-describing,
  //    so ShmChunkRef.specs/sequence_range/delta_encoded are redundant metadata
  //    — the deserialized proto carries everything the ChunkStore needs.
  //    ponytail: skip populating the redundant ShmChunkRef metadata client-side.
  internal::flat_hash_map<uint64_t, std::shared_ptr<ChunkStore::Chunk>> chunks;
  for (const ShmChunkRef& ref : req.chunks()) {
    if (ref.total_length() == 0) {
      return absl::InvalidArgumentError(
          "ShmServer::HandleInsert: zero-length chunk");
    }
    // Validate the peer-supplied offset/length BEFORE touching the pool:
    // the offset must be a block this server granted to THIS client (via
    // ALLOCATE, still outstanding), and the length must fit in its block.
    // Otherwise ParseFromArray(pool_.At(offset), total_length) is an
    // out-of-bounds read on client-controlled input. Reject with an ERROR
    // response (not just a logged status) so the client's writer fails fast
    // instead of hanging until its timeout cap.
    if (!state.outstanding_offsets_.contains(ref.shm_offset()) ||
        ref.total_length() > pool_.block_size_at(ref.shm_offset())) {
      ShmError err;
      err.set_code(ShmError::INVALID_ARGUMENT);
      err.set_message(absl::StrCat(
          "ShmServer::HandleInsert: chunk ", ref.chunk_key(),
          " references offset ", ref.shm_offset(), " (len ",
          ref.total_length(),
          ") which is not an outstanding block granted to this client"));
      std::string body;
      err.SerializeToString(&body);
      return EnqueueInsertS2C(state, ERROR, body);
    }
    ChunkData cd;
    if (!cd.ParseFromArray(pool_.At(ref.shm_offset()),
                           static_cast<int>(ref.total_length()))) {
      return absl::InternalError(absl::StrCat(
          "ShmServer::HandleInsert: failed to parse ChunkData at offset ",
          ref.shm_offset(), " (len ", ref.total_length(), ")"));
    }
    if (cd.chunk_key() != ref.chunk_key()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "ShmServer::HandleInsert: chunk_key mismatch (ref ",
          ref.chunk_key(), " vs proto ", cd.chunk_key(), ")"));
    }
    // ChunkStore::Chunk owns a copy of the ChunkData proto. Multiple items
    // may reference the same chunk_key; dedup so storage is shared.
    // ponytail: 必须先取 key 再 std::move(cd)。try_emplace(key, factory)
    // 的实参求值顺序未指定(unspecified): gcc 可能先求值
    // make_shared<Chunk>(std::move(cd)) 移动掉 cd,再求值 key 参数,此时
    // cd.chunk_key() 读到被移动后的默认值 0,导致 map 以 0 为键插入,
    // 随后 items.flat_trajectory 的真实 chunk_key 查不到 -> "unknown chunk"。
    // clang(lld 链路的旧 hermetic cc)恰好先求值 key 故不触发。系统 gcc 必修。
    uint64_t map_key = cd.chunk_key();
    chunks.try_emplace(map_key,
                       std::make_shared<ChunkStore::Chunk>(std::move(cd)));
  }

  // 2. For each PrioritizedItem, gather its referenced chunks and dispatch via
  //    InsertOrAssignAsync. The table name comes from item.table(). The
  //    completion callback fires on the table worker's callback-executor thread
  //    — NOT the dispatch thread — so it must not write the S→C ring directly
  //    (Ring is SPSC, single producer = the dispatch thread). Instead it stashes
  //    an InsertAck into the client's mutex-protected outbox; FlushOutbox
  //    (called each dispatch pass) drains it to S→C. This preserves the
  //    single-producer invariant while honoring C2 (client waits for ACK before
  //    reusing offsets).
  //
  //    ponytail: one ACK per insert request, aggregating all item keys and all
  //    chunk offsets. The client correlates by waiting for the next INSERT_ACK
  //    after sending INSERT (SPSC ordering guarantees it matches). A per-item
  //    ACK would let the client release offsets earlier, but v1's
  //    one-INSERT-per-item writer means there is exactly one item per ACK
  //    anyway. Ceiling: a batched-INSERT client would hold all offsets until the
  //    whole batch acks. Upgrade: per-item ACKs keyed by item key.
  std::vector<uint64_t> chunk_offsets;
  chunk_offsets.reserve(req.chunks_size());
  for (const ShmChunkRef& ref : req.chunks()) {
    chunk_offsets.push_back(ref.shm_offset());
  }

  InsertAck ack_template;
  for (uint64_t off : chunk_offsets) ack_template.add_offsets_to_release(off);

  // Collect the keys to ACK. We count outstanding items; when all complete,
  // enqueue the aggregate ACK. ponytail: atomic counter + insert_outbox push
  // under insert_outbox_mu; the dispatch thread is the sole reader/drainer.
  int num_items = req.items_size();
  if (num_items == 0) {
    // No items: still ack so the client can release the chunk offsets it
    // allocated (e.g. a chunks-only prefetch — v1 writer doesn't do this, but
    // be defensive).
    InsertAck ack = ack_template;
    std::string body;
    ack.SerializeToString(&body);
    return EnqueueInsertS2C(state, INSERT_ACK, body);
  }

  auto remaining = std::make_shared<std::atomic<int>>(num_items);
  auto ack_keys = std::make_shared<std::vector<uint64_t>>();
  // review #1: capture state_sp (shared_ptr<ClientState>), not a raw
  // pointer — see HandleSample.
  auto offsets = std::make_shared<std::vector<uint64_t>>(std::move(chunk_offsets));

  // ticket ⑨: route each item to its named table via FindTable. If ANY item
  // references an unknown table, reject the WHOLE request with ERROR
  // (NOT_FOUND) on the insert s2c flow and insert nothing — stricter than the
  // old single-table warn-and-skip, which silently masked bad routing. ⑩
  // (mutate_priorities/reset) will reuse FindTable.
  for (const PrioritizedItem& item_proto : req.items()) {
    if (!FindTable(item_proto.table()).ok()) {
      ShmError err;
      err.set_code(ShmError::NOT_FOUND);
      err.set_message(absl::StrCat(
          "Unknown table: ", item_proto.table(),
          " (item key ", item_proto.key(), ")"));
      std::string body;
      err.SerializeToString(&body);
      return EnqueueInsertS2C(state, ERROR, body);
    }
  }

  // Mid-insert failure must BOTH reply ERROR and drop the request's pending
  // callbacks. A bare status return only gets logged by the dispatcher — the
  // client hangs until its timeout cap; and with `remaining` initialized to
  // num_items but fewer callbacks registered, no INSERT_ACK ever fires and
  // pending_insert_callbacks leaks. The table holds the callbacks as
  // weak_ptr, so clear() safely expires them (later completions are dropped
  // by design). The client kills the stream on ERROR; the request's pool
  // offsets are reclaimed at disconnect like any abandoned outstanding.
  auto fail_insert = [&state, this](ShmError::Code code,
                                    const std::string& msg) {
    {
      absl::MutexLock lock(&state.insert_outbox_mu);
      state.pending_insert_callbacks.clear();
    }
    ShmError err;
    err.set_code(code);
    err.set_message(msg);
    std::string body;
    err.SerializeToString(&body);
    return EnqueueInsertS2C(state, ERROR, body);
  };

  for (const PrioritizedItem& item_proto : req.items()) {
    const std::string& table_name = item_proto.table();
    // FindTable already validated above; safe to dereference.
    std::shared_ptr<Table> table = *FindTable(table_name);

    // Gather the chunks referenced by this item's trajectory.
    std::vector<std::shared_ptr<ChunkStore::Chunk>> item_chunks;
    std::vector<uint64_t> keys = internal::GetChunkKeys(item_proto.flat_trajectory());
    item_chunks.reserve(keys.size());
    for (uint64_t ck : keys) {
      auto it = chunks.find(ck);
      if (it == chunks.end()) {
        return fail_insert(
            ShmError::INTERNAL,
            absl::StrCat("ShmServer::HandleInsert: item ", item_proto.key(),
                         " references unknown chunk ", ck));
      }
      item_chunks.push_back(it->second);
    }

    TableItem table_item(item_proto, std::move(item_chunks));

    // The callback fires on the table callback-executor thread. It must not
    // touch the S→C ring directly; it pushes the completed key, and when the
    // last item completes it enqueues the aggregate ACK into the outbox.
    auto cb = std::make_shared<Table::InsertCallback>(
        [remaining, ack_keys, offsets, state_sp](uint64_t completed_key) {
          ack_keys->push_back(completed_key);
          if (remaining->fetch_sub(1) == 1) {
            InsertAck ack;
            for (uint64_t k : *ack_keys) ack.add_keys(k);
            for (uint64_t off : *offsets) ack.add_offsets_to_release(off);
            std::string body;
            ack.SerializeToString(&body);
            // EnqueueInsertS2C writes to the ring directly (dispatch thread)
            // or stashes in insert_outbox on ResourceExhausted. Called off the
            // dispatch thread, the direct write would race the dispatch
            // thread's S→C producer. Route through the outbox unconditionally
            // instead.
            absl::MutexLock lock(&state_sp->insert_outbox_mu);
            state_sp->insert_outbox.emplace_back(
                static_cast<uint16_t>(INSERT_ACK), std::move(body));
            // All inserts confirmed: drop the keepalive so the callbacks (and
            // what they capture) are reclaimed. This breaks the would-be
            // cycle (cb -> lambda -> ... ; the lambda does NOT capture cb).
            state_sp->pending_insert_callbacks.clear();
          }
        });
    // Keepalive: InsertOrAssignAsync stores a weak_ptr; the table worker fires
    // the callback AFTER HandleInsert returns, so the shared_ptr must outlive
    // this function. Stash it on the client; cleared by the last callback.
    {
      absl::MutexLock lock(&state.insert_outbox_mu);
      state.pending_insert_callbacks.push_back(cb);
    }

    bool can_insert_more = false;
    absl::Status s = table->InsertOrAssignAsync(std::move(table_item),
                                                &can_insert_more, cb);
    if (!s.ok()) {
      return fail_insert(
          ShmError::INTERNAL,
          absl::StrCat("ShmServer::HandleInsert: InsertOrAssignAsync failed "
                       "for item ",
                       item_proto.key(), ": ", s.message()));
    }
    // ponytail: v1 ignores can_insert_more on the server side — the dispatch
    // thread reads one INSERT at a time and the table's pending_inserts_ queue
    // absorbs bursts (max_enqueued_inserts). Backpressure is enforced
    // client-side via the ACK wait. If the table queue fills, the client's
    // outstanding items pile up in in_flight_items_ and the writer's
    // local_can_insert_more_ gate kicks in. Upgrade: honor can_insert_more by
    // deferring the read of the next INSERT until a completion fires.
  }

  return absl::OkStatus();
}

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
