# torch.Tensor integration research (reverb-torch)

Research date: 2026-07-25. All claims verified by fetching the cited primary
sources (pytorch.org "stable" docs currently resolve to the 2.13 doc tree;
numpy docs v2.4; CUDA 13.x programming guide). Sources are inline per claim.

Context: reverb fork with TF removed; data flows as numpy arrays serialized
into a protobuf TensorProto (raw bytes + dtype + shape); server treats bytes
as opaque. Goal: accept `torch.Tensor` (possibly CUDA-resident) as an
alternative to numpy.

---

## 1. CPU interop: numpy ↔ torch

### `torch.from_numpy(ndarray)`

- The returned tensor and the ndarray **share the same memory** (zero-copy):
  "Modifications to the tensor will be reflected in the ndarray and vice
  versa. The returned tensor is not resizable."
  <https://pytorch.org/docs/stable/generated/torch.from_numpy.html>
- Accepted dtypes (exact list from the docs): `numpy.float64`, `float32`,
  `float16`, `complex64`, `complex128`, `int64`, `int32`, `int16`, `int8`,
  `uint8`, `bool`. Notably absent from the list: `uint16/32/64`,
  `longdouble/float128`, `datetime64`, structured dtypes.
  <https://pytorch.org/docs/stable/generated/torch.from_numpy.html>
- Writing to a tensor created from a **read-only** numpy array is undefined
  behavior (docs warning).
  <https://pytorch.org/docs/stable/generated/torch.from_numpy.html>
- Non-contiguous arrays: arrays with **negative strides raise ValueError**
  ("tensors with negative strides are not currently supported ... work around
  by making a copy"); non-native byte order is also rejected. Ordinary
  positive-stride non-contiguous views are fine — strides are passed through
  and memory is still shared.
  <https://github.com/pytorch/pytorch/blob/main/torch/csrc/utils/tensor_numpy.cpp>
  (see the `TORCH_CHECK_VALUE(strides[i] >= 0, ...)` and
  `PyArray_EquivByteorders` checks in `from_numpy`).

### `tensor.numpy()`

- With `force=False` (default) the conversion is performed **only if the
  tensor is on CPU, does not require grad, has no conjugate/negative bit set,
  and has a NumPy-supported dtype/layout**; the returned ndarray **shares
  storage** with the tensor (zero-copy, mutations visible both ways).
  <https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html>
- `force=True` is equivalent to
  `t.detach().cpu().resolve_conj().resolve_neg().numpy()` — i.e. it **copies**
  to host memory when the tensor is not on CPU.
  <https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html>
- Source confirmation: with `force=False` a CUDA tensor raises
  "can't convert cuda:N device type tensor to numpy. Use Tensor.cpu() to copy
  the tensor to host memory first."
  <https://github.com/pytorch/pytorch/blob/main/torch/csrc/utils/tensor_numpy.cpp>
  (`tensor_to_numpy`).

**Bottom line:** on CPU, numpy↔torch is free (zero-copy) in both directions
for the common RL dtypes, with the dtype/stride caveats above.

---

## 2. DLPack

### The protocol

- DLPack "describes the memory layout of dense, strided, n-dimensional
  arrays". When a consumer calls `y = from_dlpack(x)`, "if possible, this
  must be zero-copy (i.e. `y` will be a view on `x`)". The producer keeps
  owning the memory. A `stream` keyword handles CUDA/ROCm stream sync.
  <https://dmlc.github.io/dlpack/latest/python_spec.html>
- The device model explicitly includes GPU memory: `kDLCUDA` (CUDA GPU
  device), `kDLCUDAHost` (pinned CUDA host memory from `cudaMallocHost`),
  `kDLCUDAManaged` (managed/unified memory), plus CPU, ROCm, Metal, etc.
  <https://dmlc.github.io/dlpack/latest/c_api.html>
- The Python array API standard chose DLPack as "the primary/recommended
  protocol" for data interchange (over the CPU-only buffer protocol and the
  CUDA-only `__cuda_array_interface__`).
  <https://data-apis.org/array-api/latest/design_topics/data_interchange.html>
- The array-object side of the protocol is the pair of methods
  `__dlpack__()` / `__dlpack_device__()`, consumed by a `from_dlpack()`
  function.
  <https://dmlc.github.io/dlpack/latest/python_spec.html>

### PyTorch side

- `torch.from_dlpack(ext_tensor)` (also exposed as
  `torch.utils.dlpack.from_dlpack`): "The returned PyTorch tensor will share
  the memory with the input tensor (which may have come from another
  library)". Accepts either an object implementing `__dlpack__` or a raw
  DLPack capsule; optional `device` and `copy` arguments.
  <https://pytorch.org/docs/stable/generated/torch.from_dlpack.html>
  <https://pytorch.org/docs/stable/dlpack.html>
- **Version:** converting a tensor object directly ("supported in PyTorch
  >= 1.10") — i.e. `torch.Tensor.__dlpack__`/`__dlpack_device__` exist since
  PyTorch 1.10.
  <https://pytorch.org/docs/stable/dlpack.html> (first example comment)
- `torch.utils.dlpack.to_dlpack(tensor)` returns an opaque `PyCapsule`
  sharing the tensor's memory, but the docs mark it "a **legacy** DLPack
  interface ... The more idiomatic use of DLPack is to call `from_dlpack`
  directly on the tensor object". Current `main` source shows no deprecation
  warning on `from_dlpack` itself (it is the implementation behind
  `torch.from_dlpack`, and negotiates `max_version=(1, 0)` of the versioned
  DLPack protocol).
  <https://pytorch.org/docs/stable/dlpack.html>
  <https://github.com/pytorch/pytorch/blob/main/torch/utils/dlpack.py>
- Capsule caveat: "Only call `from_dlpack` once per capsule produced with
  `to_dlpack`. Behavior when a capsule is consumed multiple times is
  undefined."
  <https://pytorch.org/docs/stable/dlpack.html>

### NumPy side

- NumPy **1.22** added `ndarray.__dlpack__()` (NEP 47-compatible) plus a
  private `np._from_dlpack(obj)`.
  <https://numpy.org/doc/stable/release/1.22.0-notes.html>
- NumPy **1.23** added the public `numpy.from_dlpack`, which "accepts Python
  objects that implement the `__dlpack__` and `__dlpack_device__` methods and
  returns a ndarray object which is generally the view of the data".
  <https://numpy.org/doc/stable/release/1.23.0-notes.html>
- Current `np.from_dlpack(x, *, device=None, copy=None)`: `device` "must be
  `"cpu"` if passed"; the result is "generally ... a view of the input
  object". NumPy has no CUDA support, so it can only consume CPU buffers.
  <https://numpy.org/doc/stable/reference/generated/numpy.from_dlpack.html>

### Is DLPack a viable zero-copy numpy↔torch bridge on CPU and GPU?

- **CPU: yes, trivially.** numpy (>=1.22) → `torch.from_dlpack(arr)` and
  torch → `np.from_dlpack(cpu_tensor)` are both zero-copy views. (For
  numpy↔torch specifically, `from_numpy`/`.numpy()` in §1 are just as good
  and older-version friendly.)
- **GPU: yes between two GPU-aware libraries** (e.g. torch ↔ CuPy) — the
  DLPack device model carries CUDA device pointers and stream info
  zero-copy. **No via numpy**: numpy is host-only, so a CUDA torch tensor
  cannot reach numpy without a D2H copy, regardless of protocol.
  (<https://dmlc.github.io/dlpack/latest/c_api.html>,
  <https://numpy.org/doc/stable/reference/generated/numpy.from_dlpack.html>)

---

## 3. CUDA tensors: serialization and transport

### Why you can't naively serialize a CUDA tensor's raw bytes for another process

- "Device pointers or event handles are **not valid outside the process that
  created them**, and therefore cannot be directly referenced by threads
  belonging to a different process." Cross-process access requires CUDA IPC
  or VMM APIs to create "process-portable handles".
  <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html>
- Consequences for a replay pipeline: a device pointer copied as bytes into a
  protobuf is meaningless in the server process (different CUDA context or a
  different machine entirely), and there is no CPU mapping to read unless the
  memory is pinned/managed. The only way to move the *contents* to a CPU
  byte-stream transport is a D2H copy.

### Official PyTorch cross-process sharing: `torch.multiprocessing`, `share_memory_()`, CUDA IPC

- `torch.multiprocessing` "registers custom reducers, that use shared memory
  to provide shared views on the same data in different processes. Once the
  tensor/storage is moved to shared_memory (see `share_memory_()`), it will
  be possible to send it to other processes without making any copies" — this
  is the CPU-tensor path, drop-in compatible with `multiprocessing`.
  <https://pytorch.org/docs/stable/multiprocessing.html>
- `Tensor.share_memory_()` "moves the underlying storage to shared memory ...
  This is a no-op if the underlying storage is already in shared memory **and
  for CUDA tensors**" — i.e. CUDA tensors don't go through host shared
  memory; the reducers use CUDA IPC for them.
  <https://pytorch.org/docs/stable/generated/torch.Tensor.share_memory_.html>
- **CUDA sharing requirements/limitations** (from the official docs):
  "Sharing CUDA tensors between processes is supported only in Python 3,
  using a `spawn` or `forkserver` start methods." The **sending process must
  keep the original tensor alive** as long as the receiver retains it; the
  consumer should release the tensor ASAP; received tensors must not be
  re-forwarded; if the consumer dies abnormally the memory can leak in the
  producer.
  <https://pytorch.org/docs/stable/multiprocessing.html>
- The low-level `storage._share_cuda_()` / `cudaIpcMemHandle` mechanism is a
  **private** API (underscore-prefixed, no doc page); the documented public
  entry point is `torch.multiprocessing`. Underneath it is CUDA IPC:
  `cudaIpcGetMemHandle()` to export, `cudaIpcOpenMemHandle()` to import, and
  the legacy CUDA IPC API is "only currently supported on Linux platforms",
  within a single OS instance.
  <https://pytorch.org/docs/stable/multiprocessing.html>
  <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html>

### Pinned memory (faster H2D/D2H)

- "Host to GPU copies are much faster when they originate from pinned
  (page-locked) memory. CPU tensors and storages expose a `pin_memory()`
  method ... Also, once you pin a tensor or storage, you can use asynchronous
  GPU copies. Just pass an additional `non_blocking=True` argument to a
  `to()` or a `cuda()` call. This can be used to overlap data transfers with
  computation." Caveat: pinning is expensive and overuse can exhaust RAM.
  `DataLoader(pin_memory=True)` returns pinned batches.
  <https://pytorch.org/docs/stable/notes/cuda.html> ("Use pinned memory
  buffers" section)
- `Tensor.pin_memory()` "copies the tensor to pinned memory, if it's not
  already pinned".
  <https://pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html>

### Zero-copy CUDA tensor → numpy?

- **Does not exist.** `.numpy(force=False)` hard-raises on non-CPU tensors
  ("Use Tensor.cpu() to copy the tensor to host memory first"); `force=True`
  performs a D2H **copy** via `.cpu()`.
  <https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html>
  <https://github.com/pytorch/pytorch/blob/main/torch/csrc/utils/tensor_numpy.cpp>
- `np.from_dlpack` only supports `device="cpu"`.
  <https://numpy.org/doc/stable/reference/generated/numpy.from_dlpack.html>

---

## 4. How established RL libraries handle torch tensors in replay buffers

### TorchRL (pytorch/rl) — priority

- TorchRL **officially supports CUDA-resident replay storage**: the docs have
  a "CUDA prioritized replay buffers" section showing `LazyTensorStorage(
  ..., device="cuda")` with all-CUDA storage and sampling, and state that
  "Prioritized replay buffers can keep the priority trees on CPU or CUDA
  independently from the data storage. By default, CUDA tensor storage
  selects a CUDA sampler and CPU storage selects a CPU sampler", with
  `sampler_device` to split them (e.g. CPU memmap storage + CUDA priority
  sampling, or CUDA storage + CPU sampling).
  <https://pytorch.org/rl/stable/reference/data.html>
- `ReplayBuffer` also exposes `pin_memory: bool` ("whether `pin_memory()`
  should be called on the rb samples") and `prefetch: int` ("number of next
  batches to be prefetched using multithreading"), and `shared: bool` for
  cross-process sharing — i.e. the built-in acceleration knobs are pinned
  host memory + prefetch, not implicit GPU residency.
  <https://pytorch.org/rl/stable/reference/generated/torchrl.data.ReplayBuffer.html>

### Stable-Baselines3

- `ReplayBuffer` storage is plain numpy (`observations: np.ndarray`, etc. on
  host); at sample time every field goes through `to_torch()` →
  `th.as_tensor(array, device=self.device)` (CPU storage + move-to-GPU at
  sample time).
  <https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py>

### Ray RLlib

- Replay buffers store `SampleBatch` objects backed by numpy arrays
  (`import numpy as np`, e.g. `self._hit_count = np.zeros(self.capacity)`);
  torch conversion happens later in the learner, not in the buffer.
  <https://github.com/ray-project/ray/blob/master/rllib/utils/replay_buffers/replay_buffer.py>

### Community-standard pattern

Store on CPU (numpy or CPU torch tensors), move to GPU **at sample/learner
time**, optionally accelerated with pinned memory + `non_blocking=True`
(async H2D, overlappable with compute on a dedicated copy stream). TorchRL is
the exception that *also* offers full-CUDA buffers — but that is a
same-process design, inapplicable to a network-separated replay server.

---

## 5. GPU-direct options (official/production only)

- **CUDA IPC**: same-node, cross-process sharing of device buffers via
  `cudaIpcGetMemHandle()` / `cudaIpcOpenMemHandle()`; legacy IPC API is
  Linux-only and single-OS-instance; this is what `torch.multiprocessing`
  uses for CUDA tensors.
  <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html>
- **NCCL**: "inter-GPU communication primitives that are topology-aware" —
  collectives plus point-to-point send/recv, over "PCIe, NVLINK, InfiniBand
  Verbs, and IP sockets", within and across nodes, compatible with
  multi-process (MPI-style) models.
  <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html>
  PyTorch ships the NCCL backend in `torch.distributed` ("By default for
  Linux, the Gloo and NCCL backends are built and included ... NCCL only when
  building with CUDA").
  <https://pytorch.org/docs/stable/distributed.html>
- **GPUDirect RDMA**: "a direct path for data exchange between the GPU and a
  third-party peer device" (e.g. NICs), introduced with Kepler/CUDA 5.0; "the
  most important" limitation: the two devices "must share the same upstream
  PCI Express root complex". Requires kernel/driver enablement.
  <https://docs.nvidia.com/cuda/gpudirect-rdma/index.html>

---

## Implications for reverb-torch integration

*(Synthesis/recommendation — not vendor documentation.)*

1. **Keep the server exactly as-is (opaque CPU bytes).** No official
   mechanism lets a CPU byte-store server hold or forward GPU-resident data:
   device pointers are process-local (CUDA guide §4.15), and every transport
   leg that crosses a process/machine boundary through a protobuf is
   fundamentally a host-memory copy. GPU residency cannot survive
   client→server→worker through TensorProto bytes.

2. **Put torch conversion at the two edges, where it's free.**
   - *Client (writer) side:* accept `torch.Tensor`. CPU tensors →
     `.numpy()` (zero-copy view, then existing numpy→TensorProto path).
     CUDA tensors → `.detach().to('cpu', non_blocking=...)` (one unavoidable
     D2H copy; use pinned staging if profiling shows it matters). Watch the
     `from_numpy` dtype list (no `uint16/32/64`) — cast or copy for those.
   - *Sampler/learner side:* TensorProto bytes → numpy view →
     `torch.from_numpy()` (zero-copy) → batch → `pin_memory()` +
     `.to('cuda', non_blocking=True)` at the learner. This is exactly the
     SB3/TorchRL community pattern (§4).

3. **Where GPU residency actually helps:** (a) GPU-side preprocessing on the
   client *before* the D2H copy (e.g. frame stacking, compression — fewer
   bytes over the wire); (b) learner-side batch assembly and training after
   the H2D copy; (c) pinned+async H2D overlapping transfer with compute.

4. **Where a CPU roundtrip is unavoidable:** the transport leg itself.
   DLPack gives zero-copy GPU interchange only between two GPU-aware
   libraries **in cooperating processes** (§2); CUDA IPC only within one
   machine/OS instance with the producer kept alive (§3); NCCL/GPUDirect RDMA
   target multi-GPU collective/topology-aware transfer, not an opaque-byte
   replay protocol (§5).

5. **Optional same-machine fast path (only if colocating server and
   learner):** `torch.multiprocessing` with CUDA tensors via CUDA IPC skips
   the D2H/H2D roundtrip entirely, at the cost of spawn/forkserver semantics,
   producer-lifetime coupling, Linux-only legacy IPC, and abandoning the
   protobuf transport. Treat as a separate optimization, not part of the
   general torch-tensor support.
