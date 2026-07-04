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

"""Implementation of the Python client for Reverb.

`Client` is used to connect and interact with a Reverb server. The client
exposes direct methods for both inserting (i.e `insert`) and sampling (i.e
`sample`) but users should prefer to use `TrajectoryWriter` and
`TrajectoryDataset` directly whenever possible.
"""

import logging
from typing import Any, Dict, Generator, List, Optional, Sequence, Union

import numpy as np
from reverb import errors
from reverb import pybind
from reverb import replay_sample
from reverb import reverb_types
import tree


class Writer:
  """Writer is used for streaming data of arbitrary length.

  See Client.writer for documentation.
  """

  def __init__(self, internal_writer):
    """Constructor for Writer (must only be called by Client.writer)."""
    self._writer = internal_writer
    self._closed = False

  def __enter__(self) -> 'Writer':
    if self._closed:
      raise ValueError('Cannot reuse already closed Writer')
    return self

  def __exit__(self, *_):
    self.flush()
    self.close()

  def __del__(self):
    if not self._closed:
      logging.warning(
          'Writer-object deleted without calling .close explicitly.')
      self.close()

  def __repr__(self):
    return repr(self._writer) + ', closed: ' + str(self._closed)

  def append(self, data: Any):
    """Appends data to the internal buffer.

    NOTE: Calling this method alone does not result in anything being inserted
    into the replay. To trigger data insertion, `create_item`
    must be called so that the resulting sequence includes the data.

    Consider the following example:

    ```python

        A, B, C = ...
        client = Client(...)

        with client.writer(max_sequence_length=2) as writer:
          writer.append(A)  # A is added to the internal buffer.
          writer.append(B)  # B is added to the internal buffer.

          # The buffer is now full so when this is called C is added and A is
          # removed from the internal buffer and since A was never referenced by
          # a prioritized item it was never sent to the server.
          writer.append(C)

          # A sequence of length 1 is created referencing only C and thus C is
          # sent to the server.
          writer.create_item('my_table', 1, 5.0)

        # Writer is now closed and B was never referenced by a prioritized item
        # and thus never sent to the server.

    ```

    Args:
      data: The (possibly nested) structure to make available for new
        items to reference.
    """
    self._writer.Append(tree.flatten(data))

  def append_sequence(self, sequence: Any):
    """Appends sequence of data to the internal buffer.

    Each element in `sequence` must have the same leading dimension [T].

    A call to `append_sequence` is equivalent to splitting `sequence` along its
    first dimension and calling `append` once for each slice.

    For example:

    ```python

      with client.writer(max_sequence_length=2) as writer:
        sequence = np.array([[1, 2, 3],
                             [4, 5, 6]])

        # Insert two timesteps.
        writer.append_sequence([sequence])

        # Create an item that references the step [4, 5, 6].
        writer.create_item('my_table', num_timesteps=1, priority=1.0)

        # Create an item that references the steps [1, 2, 3] and [4, 5, 6].
        writer.create_item('my_table', num_timesteps=2, priority=1.0)

    ```

    Is equivalent to:

    ```python

      with client.writer(max_sequence_length=2) as writer:
        # Insert two timesteps.
        writer.append([np.array([1, 2, 3])])
        writer.append([np.array([4, 5, 6])])

        # Create an item that references the step [4, 5, 6].
        writer.create_item('my_table', num_timesteps=1, priority=1.0)

        # Create an item that references the steps [1, 2, 3] and [4, 5, 6].
        writer.create_item('my_table', num_timesteps=2, priority=1.0)

    ```

    Args:
      sequence: Batched (possibly nested) structure to make available for items
        to reference.
    """
    self._writer.AppendSequence(tree.flatten(sequence))

  def create_item(self, table: str, num_timesteps: int, priority: float):
    """Creates an item and sends it to the ReverbService.

    This method is what effectively makes data available for sampling. See the
    docstring of `append` for an illustrative example of the behavior.

    Note: The item is not always immediately pushed.  To ensure items
    are pushed to the service, call `writer.flush()` or `writer.close()`.

    Args:
      table: Name of the priority table to insert the item into.
      num_timesteps: The number of most recently added timesteps that the new
        item should reference.
      priority: The priority used for determining the sample probability of the
        new item.

    Raises:
      ValueError: If num_timesteps is < 1.
      StatusNotOk: If num_timesteps is > than the timesteps currently available
        in the buffer.
    """
    if num_timesteps < 1:
      raise ValueError('num_timesteps (%d) must be a positive integer')
    self._writer.CreateItem(table, num_timesteps, priority)

  def flush(self):
    """Flushes the stream to the ReverbService.

    This method sends any pending items from the local buffer to the service.

    Raises:
      tf.errors.OpError: If there is trouble packing or sending the data, e.g.
        if shapes are inconsistent or if there was data loss.
    """
    self._writer.Flush()

  def close(self, retry_on_unavailable=True):
    """Closes the stream to the ReverbService.

    The method is automatically called when existing the contextmanager scope.

    Note: Writer-object must be abandoned after this method called.

    Args:
      retry_on_unavailable: if true, it will keep trying to connect to the
        server if it's unavailable..

    Raises:
      ValueError: If `close` has already been called once.
      tf.errors.OpError: If there is trouble packing or sending the data, e.g.
        if shapes are inconsistent or if there was data loss.
    """
    if self._closed:
      raise ValueError('close() has already been called on Writer.')
    self._closed = True
    self._writer.Close(retry_on_unavailable)


class _ClientMethods:
  """共享转发方法:子类提供 self._client(pybind Client 或 InProcessClient)。

  ponytail: 收敛 Client/LocalClient 重复的 mutate_priorities/reset/
  checkpoint(三者仅浅转发到 self._client)。两 pybind 类型已对齐 PascalCase
  绑定名(Client 原生;InProcessClient 在 pybind.cc 加了别名),故转发逻辑一致。
  server_info 因 Client 有 timeout/signature_cache 差异,不在此共享,仅共享
  _table_infos_from_proto_strings 解析循环。
  """

  def mutate_priorities(self,
                        table: str,
                        updates: Optional[Dict[int, float]] = None,
                        deletes: Optional[List[int]] = None):
    """Updates and/or deletes existing items in a priority table."""
    if updates is None:
      updates = {}
    if deletes is None:
      deletes = []
    self._client.MutatePriorities(table, list(updates.items()), deletes)

  def reset(self, table: str):
    """Clears all items of the table and resets its RateLimiter."""
    self._client.Reset(table)

  def checkpoint(self) -> str:
    """Triggers a checkpoint to be created and returns its path."""
    return self._client.Checkpoint()

  def _table_infos_from_proto_strings(
      self, info_proto_strings) -> Dict[str, reverb_types.TableInfo]:
    table_infos = {}
    for proto_string in info_proto_strings:
      table_info = reverb_types.TableInfo.from_serialized_proto(proto_string)
      table_infos[table_info.name] = table_info
    return table_infos


class Client(_ClientMethods):
  """Client for interacting with a Reverb ReverbService from Python.

  Note: This client should primarily be used when inserting data or prototyping
  at very small scale.
  Whenever possible, prefer to use TFClient (see ./tf_client.py).
  """

  def __init__(self, server_address: str):
    """Constructor of Client.

    Args:
      server_address: Address to the Reverb ReverbService.
    """
    self._server_address = server_address
    self._client = pybind.Client(server_address)
    self._signature_cache = {}

  def __reduce__(self):
    return self.__class__, (self._server_address,)

  def __repr__(self):
    return f'Client, server_address={self._server_address}'

  @property
  def server_address(self) -> str:
    return self._server_address

  def insert(self, data, priorities: Dict[str, float]):
    """Inserts a "blob" (e.g. trajectory) into one or more priority tables.

    Note: The data is only stored once even if samples are inserted into
    multiple priority tables.

    Note: When possible, prefer to use the in graph version (see ./tf_client.py)
    to avoid stepping through Python.

    Args:
      data: A (possible nested) structure to insert.
      priorities: Mapping from table name to priority value.

    Raises:
      ValueError: If priorities is empty.
    """
    if not priorities:
      raise ValueError('priorities must contain at least one item')

    with self.writer(max_sequence_length=1) as writer:
      writer.append(data)
      for table, priority in priorities.items():
        writer.create_item(
            table=table, num_timesteps=1, priority=priority)

  def writer(self,
             max_sequence_length: int,
             delta_encoded: bool = False,
             chunk_length: Optional[int] = None,
             max_in_flight_items: Optional[int] = 25) -> Writer:
    """Constructs a writer with a `max_sequence_length` buffer.

    NOTE! This method will eventually be deprecated in favor of
    `trajectory_writer` so please prefer to use the latter.

    The writer can be used to stream data of any length. `max_sequence_length`
    controls the size of the internal buffer and ensures that prioritized items
    can be created of any length <= `max_sequence_length`.

    The writer is stateful and must be closed after the write has finished. The
    easiest way to manage this is to use it as a contextmanager:

    ```python

    with client.writer(10) as writer:
       ...  # Write data of any length.

    ```

    If not used as a contextmanager then `.close()` must be called explicitly.

    Args:
      max_sequence_length: Size of the internal buffer controlling the upper
        limit of the number of timesteps which can be referenced in a single
        prioritized item. Note that this is NOT a limit of how many timesteps or
        items that can be inserted.
      delta_encoded: If `True` (False by default)  tensors are delta encoded
        against the first item within their respective batch before compressed.
        This can significantly reduce RAM at the cost of a small amount of CPU
        for highly correlated data (e.g frames of video observations).
      chunk_length: Number of timesteps grouped together before delta encoding
        and compression. Set by default to `min(10, max_sequence_length)` but
        can be overridden to achieve better compression rates when using longer
        sequences with a small overlap.
      max_in_flight_items: The maximum number of items allowed to be "in flight"
        at the same time. An item is considered to be "in flight" if it has been
        sent to the server but the response confirming that the operation
        succeeded has not yet been received. Note that "in flight" items does
        NOT include items that are in the client buffer due to the current chunk
        not having reached its desired length yet. None results in an unlimited
        number of "in flight" items.

    Returns:
      A `Writer` with `max_sequence_length`.

    Raises:
      ValueError: If max_sequence_length < 1.
      ValueError: if chunk_length > max_sequence_length.
      ValueError: if chunk_length < 1.
      ValueError: If max_in_flight_items < 1.
    """
    if max_sequence_length < 1:
      raise ValueError('max_sequence_length (%d) must be a positive integer' %
                       max_sequence_length)

    if chunk_length is None:
      chunk_length = min(10, max_sequence_length)

    if chunk_length < 1 or chunk_length > max_sequence_length:
      raise ValueError(
          'chunk_length (%d) must be a positive integer le to max_sequence_length (%d)'
          % (chunk_length, max_sequence_length))

    if max_in_flight_items is None:
      # Mimic 'unlimited' number of "in flight" items with a big value.
      max_in_flight_items = 1_000_000

    if max_in_flight_items < 1:
      raise ValueError(
          f'max_in_flight_items ({max_in_flight_items}) must be a '
          f'positive integer')

    return Writer(
        self._client.NewWriter(chunk_length, max_sequence_length, delta_encoded,
                               max_in_flight_items))

  def sample(
      self,
      table: str,
      num_samples: int = 1,
      *,
      emit_timesteps: bool = True,
      unpack_as_table_signature: bool = False,
  ) -> Generator[Union[List[replay_sample.ReplaySample],
                       replay_sample.ReplaySample], None, None]:
    """Samples `num_samples` items from table `table` of the Server.

    NOTE: This method should NOT be used for real training. TrajectoryDataset
    and TimestepDataset should always be preferred over this method.

    Note: If data was written using `insert` (e.g when inserting complete
    trajectories) then the returned "sequence" will be a list of length 1
    containing the trajectory as a single item.

    If `num_samples` is greater than the number of items in `table`, (or
    a rate limiter is used to control sampling), then the returned generator
    will block when an item past the sampling limit is requested.  It will
    unblock when sufficient additional items have been added to `table`.

    Example:

    ```python

    server = Server(..., tables=[queue("queue", ...)])
    client = Client(...)

    # Don't insert anything into "queue"
    generator = client.sample("queue")
    generator.next()  # Blocks until another thread/process writes to queue.

    ```

    Args:
      table: Name of the priority table to sample from.
      num_samples: (default to 1) The number of samples to fetch.
      emit_timesteps: If True then trajectories are returned as a list of
        `ReplaySample`, each representing a single step within the trajectory.
      unpack_as_table_signature: If True then the sampled data is unpacked
        according to the structure of the table signature. If the table does
        not have a signature then flat data is returned.

    Yields:
      If `emit_timesteps` is `True`:

        Lists of timesteps (lists of instances of `ReplaySample`).
        If data was inserted into the table via `insert`, then each element
        of the generator is a length 1 list containing a `ReplaySample`.
        If data was inserted via a writer, then each element is a list whose
        length is the sampled trajectory's length.

      If emit_timesteps is False:

        An instance of `ReplaySample` where the data is unpacked according to
        the signature of the table. If the table does not have any signature
        then the data is flat, i.e each element is a leaf node of the full
        trajectory.

    Raises:
      ValueError: If `emit_timestep` is True but the trajectory cannot be
        decomposed into timesteps.
    """
    buffer_size = 1

    if unpack_as_table_signature:
      signature = self._get_signature_for_table(table)
    else:
      signature = None

    if signature:
      unflatten = lambda x: tree.unflatten_as(signature, x)
    else:
      unflatten = lambda x: x

    sampler = self._client.NewSampler(table, num_samples, buffer_size)

    for _ in range(num_samples):
      sample = sampler.GetNextTrajectory()

      info = replay_sample.SampleInfo(
          key=int(sample[0]),
          probability=float(sample[1]),
          table_size=int(sample[2]),
          priority=float(sample[3]),
          times_sampled=int(sample[4]))
      data = sample[len(info):]

      if emit_timesteps:
        if len(set([len(col) for col in data])) != 1:
          raise ValueError(
              'Can\'t split non timestep trajectory into timesteps.')

        timesteps = []
        for i in range(data[0].shape[0]):
          timestep = replay_sample.ReplaySample(
              info=info,
              data=unflatten([np.asarray(col[i], col.dtype) for col in data]))
          timesteps.append(timestep)

        yield timesteps
      else:
        yield replay_sample.ReplaySample(info, unflatten(data))

  def server_info(self,
                  timeout: Optional[int] = None
                 ) -> Dict[str, reverb_types.TableInfo]:
    """Get table metadata information.

    Args:
      timeout: Timeout in seconds to wait for server response. By default no
        deadline is set and call will block indefinetely until server responds.

    Returns:
      A dictionary mapping table names to their associated `TableInfo`
      instances, which contain metadata about the table.

    Raises:
      errors.DeadlineExceededError: If timeout provided and exceeded.
    """
    try:
      info_proto_strings = self._client.ServerInfo(timeout or 0)
    except RuntimeError as e:
      if 'Deadline Exceeded' in str(e) and timeout is not None:
        raise errors.DeadlineExceededError(
            f'ServerInfo call did not complete within provided timeout of '
            f'{timeout}s')
      raise

    table_infos = self._table_infos_from_proto_strings(info_proto_strings)

    # Populate the signature cache if this is the first time server_info is
    # (successfully) called.
    if not self._signature_cache:
      self._signature_cache = {
          table: info.signature
          for table, info in table_infos.items()
      }

    return table_infos

  def trajectory_writer(self,
                        num_keep_alive_refs: int,
                        *,
                        validate_items: bool = True):
    """Constructs a new `TrajectoryWriter`.

    Note: The chunk length is auto tuned by default. Use
      `TrajectoryWriter.configure` to override this behaviour.

    See `TrajectoryWriter` for more detailed documentation about the writer
    itself.

    Args:
      num_keep_alive_refs: The size of the circular buffer which each column
        maintains for the most recent data appended to it. When a data reference
        popped from the buffer it can no longer be referenced by new items. The
        value `num_keep_alive_refs` can therefore be interpreted as maximum
        number of steps which a trajectory can span.
      validate_items: Whether to validate items against the table signature
        before they are sent to the server. This requires table signature to be
        fetched from the server and cached locally.

    Returns:
      A `TrajectoryWriter` with auto tuned chunk lengths in each column.

    Raises:
      ValueError: If num_keep_alive_refs < 1.
    """
    if num_keep_alive_refs < 1:
      raise ValueError(
          f'num_keep_alive_refs ({num_keep_alive_refs}) must be a positive '
          f'integer'
      )

    chunker_options = pybind.AutoTunedChunkerOptions(num_keep_alive_refs, 1.0)
    cpp_writer = self._client.NewTrajectoryWriter(chunker_options,
                                                  validate_items)
    from reverb import trajectory_writer as trajectory_writer_lib  # pylint: disable=g-import-not-at-top
    return trajectory_writer_lib.TrajectoryWriter(cpp_writer)

  def structured_writer(self, configs):
    """Constructs a new `StructuredWriter`.

    See `StructuredWriter` for more detailed documentation.

    Args:
      configs: Configurations describing how the writer should transform the
        sequence of steps into table insertions.

    Returns:
      A `StructuredWriter` that inserts items according to `configs`.

    Raises:
      ValueError: If `configs` is empty or contains an invalid config.
    """
    if not configs:
      raise ValueError('At least one config must be provided.')

    serialized_configs = [config.SerializeToString() for config in configs]
    cpp_writer = self._client.NewStructuredWriter(serialized_configs)
    from reverb import structured_writer as structured_writer_lib  # pylint: disable=g-import-not-at-top
    return structured_writer_lib.StructuredWriter(cpp_writer)

  def _get_signature_for_table(self, table: str):
    if not self._signature_cache:
      self.server_info()  # Populates the cache.

    if table not in self._signature_cache:
      raise ValueError(
          f'Could not find table "{table}". The following tables exists: '
          f'{", ".join(self._signature_cache.keys())}.')

    return self._signature_cache[table]


class LocalClient(_ClientMethods):
  """Python wrapper around the C++ `InProcessClient` for embedded mode.

  Provides a numpy-friendly API mirroring the historical `Client`: writers are
  context managers returned by `trajectory_writer`, and `sample` yields
  `ReplaySample` objects. All data is exchanged as numpy arrays; no TensorFlow
  is required.
  """

  def __init__(self, internal_client: 'pybind.InProcessClient'):
    self._client = internal_client

  def __repr__(self):
    return 'LocalClient (in-process, numpy)'

  def trajectory_writer(self,
                        table: str,
                        num_keep_alive_refs: int,
                        *,
                        max_chunk_length: Optional[int] = None):
    """Constructs a `TrajectoryWriter` bound to `table` in local mode.

    Args:
      table: Name of the table that items created by this writer are inserted
        into.
      num_keep_alive_refs: Size of the circular buffer of recent data
        references; the maximum trajectory length.
      max_chunk_length: Optional constant chunk length. If None, the chunk
        length is auto-tuned.

    Returns:
      A `TrajectoryWriter` context manager.
    """
    if num_keep_alive_refs < 1:
      raise ValueError(
          f'num_keep_alive_refs ({num_keep_alive_refs}) must be a positive '
          f'integer'
      )
    if max_chunk_length is None:
      chunker_options = pybind.AutoTunedChunkerOptions(num_keep_alive_refs, 1.0)
    else:
      chunker_options = pybind.ConstantChunkerOptions(
          max_chunk_length=max_chunk_length,
          num_keep_alive_refs=num_keep_alive_refs)
    cpp_writer = self._client.new_trajectory_writer(table, chunker_options)
    # Imported here to avoid a circular import (trajectory_writer imports
    # pybind, not client) and to keep the module import TF-free.
    from reverb import trajectory_writer as trajectory_writer_lib  # pylint: disable=g-import-not-at-top
    return trajectory_writer_lib.TrajectoryWriter(cpp_writer)

  def new_sampler(self, table: str, num_samples: int = 1, buffer_size: int = 1,
                   timeout_ms: Optional[int] = None):
    """Constructs a `Sampler` over `table` in local mode.

    Args:
      table: Name of the table to sample from.
      num_samples: Maximum number of samples the sampler will yield.
      buffer_size: Max in-flight samples per worker.
      timeout_ms: Per-sample rate-limiter timeout in milliseconds. `None` (or a
        negative value) waits forever (the historical default). A positive
        value surfaces a `reverb.errors.DeadlineExceededError` when the table's
        rate limiter blocks longer than `timeout_ms`.
    """
    # -1 is the C++ sentinel for InfiniteDuration (see sampler.h).
    timeout_ms_arg = -1 if timeout_ms is None or timeout_ms < 0 else timeout_ms
    return self._client.new_sampler(
        table, num_samples, buffer_size, timeout_ms_arg)

  def sample(self, table: str, num_samples: int = 1,
             timeout_ms: Optional[int] = None):
    """Yields `ReplaySample` objects sampled from `table`.

    Args:
      table: Name of the table to sample from.
      num_samples: Number of samples to yield.
      timeout_ms: Rate-limiter timeout in milliseconds per sample. `None` waits
        forever. A positive value raises `reverb.errors.DeadlineExceededError`
        if the table's rate limiter blocks longer than `timeout_ms`.
    """
    sampler = self.new_sampler(table, num_samples, timeout_ms=timeout_ms)
    for _ in range(num_samples):
      try:
        flat = sampler.GetNextTrajectory()
      except RuntimeError as e:
        # The C++ layer returns absl::DeadlineExceededError, which pybind maps
        # to RuntimeError (no DeadlineExceeded mapping in MaybeRaiseFromStatus).
        # Mirror the server_info() conversion below.
        if 'Deadline Exceeded' in str(e) or 'Timeout exceeded' in str(e):
          raise errors.DeadlineExceededError(
              f'Rate limiter blocked longer than timeout_ms={timeout_ms}ms')
        raise
      info = replay_sample.SampleInfo(
          key=int(flat[0]),
          probability=float(flat[1]),
          table_size=int(flat[2]),
          priority=float(flat[3]),
          times_sampled=int(flat[4]),
      )
      data = flat[len(info):]
      yield replay_sample.ReplaySample(info=info, data=data)

  def server_info(self) -> Dict[str, reverb_types.TableInfo]:
    return self._table_infos_from_proto_strings(self._client.server_info())
