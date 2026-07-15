# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Top level import for Reverb.

Reverb supports two modes:

  * **In-process / numpy-only mode** (no TensorFlow required): construct a
    `reverb.Server(in_process=True)` and interact with it through
    `server.in_process_client` (an `InProcessClient` wrapped by `LocalClient`).
    Data flows as numpy arrays. Zero gRPC overhead; tables are held directly.

  * **gRPC mode** (numpy-only): the original `Client`/`Writer`/
    `StructuredWriter` path over a networked `Server(in_process=False)`. The
    C++ `client.cc`/`writer.cc`/`streaming_trajectory_writer.cc` were de-TF'd
    (data flows as numpy-backed `TensorBuffer`), so the gRPC bindings are
    restored and `Client`/`Writer` no longer raise `NotImplementedError`.

TensorFlow is intentionally NOT imported at module load time: TF is only
needed to encode `tf.TypeSpec`-based table signatures, and importing it
eagerly would load TF's bundled gRPC and conflict with Reverb's own gRPC
statically linked into `libreverb.so` (duplicate flag registration).
"""

# pylint: disable=g-import-notat-top
# pylint: disable=g-bad-import-order

from reverb import item_selectors as selectors

# Expose the in-process C++ client class directly for advanced users.
from reverb import pybind as _pybind  # noqa: E402
from reverb import rate_limiters
from reverb.client import Client, LocalClient, ShmClient, Writer
from reverb.errors import DeadlineExceededError, ReverbError
from reverb.platform.default import checkpointers
from reverb.replay_sample import ReplaySample, SampleInfo
from reverb.server import Server, Table
from reverb.trajectory_writer import TrajectoryColumn, TrajectoryWriter

InProcessClient = _pybind.InProcessClient
del _pybind

# pylint: enable=g-bad-import-order
# pylint: enable=g-import-not-at-top
