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
    Data flows as numpy arrays. This is the path surfaced by the de-TF refactor.

  * **gRPC mode** (TensorFlow required): the original `Client`/`Writer`/
    `StructuredWriter` path. These C++ bindings (`client.cc`/`writer.cc`/
    `structured_writer.cc`) still depend on TensorFlow and have been removed
    from the pybind module until those files are de-TF'd. The Python `Client`/
    `Writer` classes remain but raise `NotImplementedError` on construction.

TensorFlow is intentionally NOT imported at module load time, even when
installed: doing so would load TF's bundled gRPC and conflict with Reverb's own
gRPC statically linked into `libreverb.so` (duplicate flag registration).
"""

# pylint: disable=g-import-notat-top
# pylint: disable=g-bad-import-order

from reverb import item_selectors as selectors
from reverb import rate_limiters

from reverb.client import Client
from reverb.client import LocalClient
from reverb.client import Writer

from reverb.errors import DeadlineExceededError
from reverb.errors import ReverbError

from reverb.platform.default import checkpointers

from reverb.replay_sample import ReplaySample
from reverb.replay_sample import SampleInfo

from reverb.server import Server
from reverb.server import Table

from reverb.trajectory_writer import TrajectoryColumn
from reverb.trajectory_writer import TrajectoryWriter

# Expose the in-process C++ client class directly for advanced users.
from reverb import pybind as _pybind  # noqa: E402
InProcessClient = _pybind.InProcessClient
del _pybind

# pylint: enable=g-bad-import-order
# pylint: enable=g-import-not-at-top
