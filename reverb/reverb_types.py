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

"""Pytype helpers."""

import dataclasses
from typing import Any, Optional, Union

from reverb import pybind

from reverb.cc import schema_pb2

# TensorFlow is only required to (de)code table signatures as nested
# `tf.TypeSpec` structures, and only when a signature is actually provided.
# The in-process / numpy-only mode must not import TF at module load time
# (it would load TF's bundled gRPC and conflict with Reverb's own gRPC linked
# into libreverb.so), so TF / nested_structure_coder are imported lazily inside
# the methods that need them.
tf = None
nested_structure_coder = None


Fifo = pybind.FifoSelector
Heap = pybind.HeapSelector
Lifo = pybind.LifoSelector
Prioritized = pybind.PrioritizedSelector
Uniform = pybind.UniformSelector

SelectorType = Union[Fifo, Heap, Lifo, Prioritized, Uniform]

# Signatures are an opaque nested structure; without eager TF type hints we
# treat them as Any. Actual encoding/decoding happens lazily below.
SpecNest = Any


@dataclasses.dataclass
class TableInfo:
  """A tuple describing Table information.

  The main difference between this object and a `schema_pb2.TableInfo` message
  is that the signature is a nested structure of `tf.TypeSpec` objects,
  instead of a raw proto.

  It also has a `TableInfo.from_serialized_proto` classmethod, which is an
  alternate constructor for creating a `TableInfo` object from a serialized
  `schema_pb2.TableInfo` proto.
  """
  # LINT.IfChange
  name: str
  sampler_options: schema_pb2.KeyDistributionOptions
  remover_options: schema_pb2.KeyDistributionOptions
  max_size: int
  max_times_sampled: int
  rate_limiter_info: schema_pb2.RateLimiterInfo
  signature: Optional[SpecNest]
  current_size: int
  num_episodes: int
  num_deleted_episodes: int
  num_unique_samples: int
  table_worker_time: schema_pb2.TableWorkerTime
  # LINT.ThenChange(../../reverb/schema.proto)

  @classmethod
  def from_serialized_proto(cls, proto_string: bytes) -> 'TableInfo':
    """Constructs a TableInfo from a serialized `schema_pb2.TableInfo`."""
    proto = schema_pb2.TableInfo.FromString(proto_string)
    signature = None
    if proto.HasField('signature'):
      # Lazily import TF's nested_structure_coder to decode the signature into
      # a nested tf.TypeSpec structure. If TF is unavailable (in-process numpy
      # mode) we leave the signature as None rather than failing.
      try:
        # pylint: disable=g-import-not-at-top
        from tensorflow.python.saved_model import nested_structure_coder
        # pylint: enable=g-import-not-at-top
        signature = nested_structure_coder.decode_proto(proto.signature)
      except ImportError:
        signature = None
    return cls(
        name=proto.name,
        sampler_options=proto.sampler_options,
        remover_options=proto.remover_options,
        max_size=proto.max_size,
        max_times_sampled=proto.max_times_sampled,
        rate_limiter_info=proto.rate_limiter_info,
        signature=signature,
        current_size=proto.current_size,
        num_episodes=proto.num_episodes,
        num_deleted_episodes=proto.num_deleted_episodes,
        num_unique_samples=proto.num_unique_samples,
        table_worker_time=proto.table_worker_time,
        )
