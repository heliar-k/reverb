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
`sample`) but users should prefer to use `TrajectoryWriter` directly whenever
possible.
"""

import logging
from typing import Any, Dict, Generator, List, Literal, Optional, Union, overload

import numpy as np
import tree

from reverb import pybind, replay_sample, reverb_types, torch_support


class Writer:
    """Writer is used for streaming data of arbitrary length.

    See Client.writer for documentation.
    """

    def __init__(self, internal_writer):
        """Constructor for Writer (must only be called by Client.writer)."""
        self._writer = internal_writer
        self._closed = False

    def __enter__(self) -> "Writer":
        if self._closed:
            raise ValueError("Cannot reuse already closed Writer")
        return self

    def __exit__(self, *_):
        # close() must run even when flush() raises — otherwise the C++
        # writer's background thread isn't joined until GC.
        try:
            self.flush()
        finally:
            self.close()

    def __del__(self):
        if not self._closed:
            logging.warning("Writer-object deleted without calling .close explicitly.")
            self.close()

    def __repr__(self):
        return repr(self._writer) + ", closed: " + str(self._closed)

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
        # Accept torch.Tensor leaves (see TrajectoryWriter.append).
        self._writer.Append(tree.flatten(torch_support.to_numpy_tree(data)))

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
        # Accept torch.Tensor leaves (see TrajectoryWriter.append).
        self._writer.AppendSequence(tree.flatten(torch_support.to_numpy_tree(sequence)))

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
            raise ValueError(
                f"num_timesteps ({num_timesteps}) must be a positive integer"
            )
        self._writer.CreateItem(table, num_timesteps, priority)

    def flush(self):
        """Flushes the stream to the ReverbService.

        This method sends any pending items from the local buffer to the service.
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
        """
        if self._closed:
            raise ValueError("close() has already been called on Writer.")
        self._closed = True
        self._writer.Close(retry_on_unavailable)


class _BaseClient:
    """Shared logic between gRPC `Client` and in-process `LocalClient`.

    Both clients implement the same `sample`/`mutate_priorities`/`reset`/
    `server_info`/`checkpoint`/`_get_signature_for_table`/`trajectory_writer`/
    `structured_writer` semantics; only the underlying C++ call signatures
    differ (gRPC `ServerInfo(timeout)` vs. in-process `server_info()`, gRPC
    `NewSampler` has no rate-limiter timeout, and gRPC `NewTrajectoryWriter`
    takes a `validate_items` flag the in-process/SHM bindings omit). Those
    differences are captured in three hooks implemented by each subclass:
    `_fetch_server_info_proto`, `_new_sampler`, and `_new_trajectory_writer`.
    `structured_writer` needs no hook: all three C++ clients expose an identical
    `NewStructuredWriter(vector<string>)` binding.

    `output_format` selects the leaf type yielded by `sample`: "numpy"
    (default) returns numpy arrays; "torch" converts each leaf to a zero-copy
    `torch.Tensor` where possible (unsupported dtypes stay numpy).
    """

    _VALID_OUTPUT_FORMATS = ("numpy", "torch")

    def __init__(self, *, output_format: str = "numpy"):
        # Subclasses assign `self._client` and `self._server_address` (gRPC only).
        if output_format not in self._VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"output_format must be one of {self._VALID_OUTPUT_FORMATS}, "
                f"got {output_format!r}"
            )
        self._output_format = output_format
        self._signature_cache: Dict[str, Any] = {}

    # Default for `sample(emit_timesteps=...)` when the caller omits the arg.
    # Both gRPC `Client` and the in-process `LocalClient` default to True now
    # that the local API mirrors the gRPC surface (insert/writer available,
    # trajectory_writer unchained from a single table).
    _default_emit_timesteps: bool = True

    def _fetch_server_info_proto(self, timeout: Optional[int]):
        """Fetches serialized TableInfo proto strings from the C++ client.

        Subclasses implement the C++ call difference (gRPC passes a timeout,
        in-process ignores it). Returns a sequence of `bytes`.
        """
        raise NotImplementedError

    def _new_sampler(
        self, table: str, num_samples: int, buffer_size: int, timeout_ms: Optional[int]
    ):
        """Constructs a C++ `Sampler`.

        The gRPC `NewSampler` does not take a rate-limiter timeout; the in-process
        `NewSampler` does. Subclasses pass `timeout_ms` through or ignore it.
        """
        raise NotImplementedError

    def _get_signature_for_table(self, table: str):
        if not self._signature_cache:
            self.server_info()  # Populates the cache.

        if table not in self._signature_cache:
            raise ValueError(
                f'Could not find table "{table}". The following tables exists: '
                f"{', '.join(self._signature_cache.keys())}."
            )

        return self._signature_cache[table]

    def server_info(
        self, timeout: Optional[int] = None
    ) -> Dict[str, reverb_types.TableInfo]:
        """Get table metadata information.

        Args:
          timeout: Timeout in seconds to wait for server response. By default no
            deadline is set and call will block indefinetely until server responds.
            Ignored by the in-process `LocalClient` (which has no C++ timeout).

        Returns:
          A dictionary mapping table names to their associated `TableInfo`
          instances, which contain metadata about the table.

        Raises:
          errors.DeadlineExceededError: If timeout provided and exceeded.
        """
        info_proto_strings = self._fetch_server_info_proto(timeout)

        table_infos = {}
        for proto_string in info_proto_strings:
            table_info = reverb_types.TableInfo.from_serialized_proto(proto_string)
            table_infos[table_info.name] = table_info

        # Refresh the signature cache on every call so that table replacements
        # (e.g. Table.replace) are reflected. Mirrors the C++ Client::ServerInfo
        # "Forces an update of internal signature caches" semantics.
        self._signature_cache = {
            table: info.signature for table, info in table_infos.items()
        }

        return table_infos

    def mutate_priorities(
        self,
        table: str,
        updates: Optional[Dict[int, float]] = None,
        deletes: Optional[List[int]] = None,
    ):
        """Updates and/or deletes existing items in a priority table.

        NOTE: Prefer `TrajectoryWriter` for bulk priority management.

        Actions are executed in the same order as the arguments are specified.

        Args:
          table: Name of the priority table to update.
          updates: Mapping from priority item key to new priority value. If a key
            cannot be found then it is ignored.
          deletes: List of keys for priority items to delete. If a key cannot be
            found then it is ignored.
        """
        if updates is None:
            updates = {}
        if deletes is None:
            deletes = []
        self._client.MutatePriorities(table, list(updates.items()), deletes)

    def reset(self, table: str):
        """Clears all items of the table and resets its RateLimiter.

        Args:
          table: Name of the priority table to reset.
        """
        self._client.Reset(table)

    def checkpoint(self) -> str:
        """Triggers a checkpoint to be created.

        Returns:
          Absolute path to the saved checkpoint.
        """
        return self._client.Checkpoint()

    @overload
    def sample(
        self,
        table: str,
        num_samples: int = ...,
        *,
        emit_timesteps: Literal[True],
        unpack_as_table_signature: bool = ...,
        timeout_ms: Optional[int] = ...,
    ) -> Generator[List[replay_sample.ReplaySample], None, None]: ...

    @overload
    def sample(
        self,
        table: str,
        num_samples: int = ...,
        *,
        emit_timesteps: Literal[False],
        unpack_as_table_signature: bool = ...,
        timeout_ms: Optional[int] = ...,
    ) -> Generator[replay_sample.ReplaySample, None, None]: ...

    @overload
    def sample(
        self,
        table: str,
        num_samples: int = ...,
        *,
        emit_timesteps: None = ...,
        unpack_as_table_signature: bool = ...,
        timeout_ms: Optional[int] = ...,
    ) -> Generator[
        Union[List[replay_sample.ReplaySample], replay_sample.ReplaySample], None, None
    ]: ...

    def sample(
        self,
        table: str,
        num_samples: int = 1,
        *,
        emit_timesteps: Optional[bool] = None,
        unpack_as_table_signature: bool = False,
        timeout_ms: Optional[int] = None,
    ) -> Generator[
        Union[List[replay_sample.ReplaySample], replay_sample.ReplaySample], None, None
    ]:
        """Samples `num_samples` items from table `table` of the Server.

        NOTE: This method is not optimized for high-throughput training; prefer
        `TrajectoryWriter` for bulk inserts.

        Note: If data was written using `insert` (e.g when inserting complete
        trajectories) then the returned "sequence" will be a list of length 1
        containing the trajectory as a single item.

        If `num_samples` is greater than the number of items in `table`, (or
        a rate limiter is used to control sampling), then the returned generator
        will block when an item past the sampling limit is requested.  It will
        unblock when sufficient additional items have been added to `table`.

        Args:
          table: Name of the priority table to sample from.
          num_samples: (default to 1) The number of samples to fetch.
          emit_timesteps: If True then trajectories are returned as a list of
            `ReplaySample`, each representing a single step within the trajectory.
            If False, a single `ReplaySample` per sampled item is yielded. If
            `None` (default), falls back to the client's
            `_default_emit_timesteps`, which is `True` for all built-in clients
            (`Client`, `LocalClient`, `ShmClient`).
          unpack_as_table_signature: If True then the sampled data is unpacked
            according to the structure of the table signature. If the table does
            not have a signature then flat data is returned.
          timeout_ms: Per-sample rate-limiter timeout in milliseconds. `None` waits
            forever. A positive value raises `reverb.errors.DeadlineExceededError`
            if the table's rate limiter blocks longer than `timeout_ms`. Ignored by
            the gRPC `Client` (whose C++ `NewSampler` has no timeout parameter).

        Note:
          SHM transport (`ShmClient`): at most ONE live sampler per connection.
          Each live (not-yet-exhausted) `sample()` generator holds a sampler,
          so starting a second `sample()` before the previous generator is
          exhausted raises an error immediately — this guards the shared-memory
          ring's single-producer invariant (a second sampler could otherwise
          receive the first sampler's responses, silently mixing tables).
          Exhaust or close the previous generator first, or use a second
          connection for concurrent sampling. gRPC and in-process clients are
          not restricted.

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
        if emit_timesteps is None:
            emit_timesteps = self._default_emit_timesteps

        buffer_size = 1

        if unpack_as_table_signature:
            signature = self._get_signature_for_table(table)
        else:
            signature = None

        if signature:
            unflatten = lambda x: tree.unflatten_as(signature, x)
        else:
            unflatten = lambda x: x

        sampler = self._new_sampler(table, num_samples, buffer_size, timeout_ms)

        if self._output_format == "torch":
            convert = torch_support.from_numpy_leaf
        else:
            convert = lambda x: x

        for _ in range(num_samples):
            sample = sampler.GetNextTrajectory()

            info = replay_sample.SampleInfo(
                key=int(sample[0]),
                probability=float(sample[1]),
                table_size=int(sample[2]),
                priority=float(sample[3]),
                times_sampled=int(sample[4]),
            )
            data = sample[len(info) :]

            if emit_timesteps:
                if len(set([len(col) for col in data])) != 1:
                    raise ValueError(
                        "Can't split non timestep trajectory into timesteps."
                    )

                timesteps = []
                for i in range(data[0].shape[0]):
                    timestep = replay_sample.ReplaySample(
                        info=info,
                        data=unflatten(
                            [convert(np.asarray(col[i], col.dtype)) for col in data]
                        ),
                    )
                    timesteps.append(timestep)

                yield timesteps
            else:
                yield replay_sample.ReplaySample(info, unflatten([convert(x) for x in data]))

    def writer(
        self,
        max_sequence_length: int,
        delta_encoded: bool = False,
        chunk_length: Optional[int] = None,
        max_in_flight_items: Optional[int] = 25,
    ) -> Writer:
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

        Note: these range checks are NOT redundant with a C++ `Validate()`. The
        legacy `Writer` path has no `options.Validate()` (that lives on
        `TrajectoryWriter::Options`, a different class); `Client::NewWriter`
        (gRPC) does zero validation and the `Writer` ctor only does
        `CHECK_GT(max_in_flight_items_, 0)` (a hard abort, not a Python
        exception). `InProcessClient::NewWriter` validates the three `< 1`
        cases but NOT `chunk_length > max_sequence_length`. So removing these
        checks would turn `ValueError` into either a C++ abort (gRPC,
        `max_in_flight_items < 1`) or a silently-constructed broken writer
        (gRPC, `max_sequence_length < 1` / `chunk_length > max`). They are
        kept deliberately; see ticket "合并 Python 层客户端 API 三份副本".
        """
        if max_sequence_length < 1:
            raise ValueError(
                "max_sequence_length (%d) must be a positive integer"
                % max_sequence_length
            )

        if chunk_length is None:
            chunk_length = min(10, max_sequence_length)

        if chunk_length < 1 or chunk_length > max_sequence_length:
            raise ValueError(
                "chunk_length (%d) must be a positive integer le to max_sequence_length (%d)"
                % (chunk_length, max_sequence_length)
            )

        if max_in_flight_items is None:
            # Mimic 'unlimited' number of "in flight" items with a big value.
            max_in_flight_items = 1_000_000

        if max_in_flight_items < 1:
            raise ValueError(
                f"max_in_flight_items ({max_in_flight_items}) must be a "
                f"positive integer"
            )

        # Both gRPC `Client` and `InProcessClient` bind `NewWriter` (PascalCase);
        # the local path dispatches by `item.table()` into the client's tables.
        return Writer(
            self._client.NewWriter(
                chunk_length, max_sequence_length, delta_encoded, max_in_flight_items
            )
        )

    def insert(self, data, priorities: Dict[str, float]):
        """Inserts a "blob" (e.g. trajectory) into one or more priority tables.

        Note: The data is only stored once even if samples are inserted into
        multiple priority tables.

        Note: Prefer `TrajectoryWriter` for streaming inserts to avoid stepping
        through Python per item.

        Args:
          data: A (possible nested) structure to insert.
          priorities: Mapping from table name to priority value.

        Raises:
          ValueError: If priorities is empty.
        """
        if not priorities:
            raise ValueError("priorities must contain at least one item")

        with self.writer(max_sequence_length=1) as writer:
            writer.append(data)
            for table, priority in priorities.items():
                writer.create_item(table=table, num_timesteps=1, priority=priority)

    def trajectory_writer(
        self,
        num_keep_alive_refs: int,
        *,
        max_chunk_length: Optional[int] = None,
        validate_items: bool = True,
    ):
        """Constructs a new `TrajectoryWriter`.

        The chunker is auto-tuned by default; pass `max_chunk_length` for a
        constant chunk length. Use `TrajectoryWriter.configure` to override
        per-column after construction.

        Args:
          num_keep_alive_refs: Size of the circular buffer of recent data
            references; the maximum trajectory length.
          max_chunk_length: Optional constant chunk length. If `None` (the
            default) the chunk length is auto-tuned. Honoured by all three
            clients; previously the gRPC `Client` only auto-tuned (it did not
            expose this kwarg), but its C++ `NewTrajectoryWriter` accepts any
            `ChunkerOptions`, so the merged base now honours it uniformly.
          validate_items: Whether to validate items against the table signature
            before they are sent to the server. Honoured by the gRPC `Client`
            (where it gates a signature-fetch round-trip); `LocalClient` and
            `ShmClient` always validate when a signature is available (their
            in-process/SHM paths populate `flat_signature_map` directly), so
            this flag is accepted for signature parity but has no effect there.

        Returns:
          A `TrajectoryWriter` context manager.

        Raises:
          ValueError: If `num_keep_alive_refs < 1`.
        """
        if num_keep_alive_refs < 1:
            raise ValueError(
                f"num_keep_alive_refs ({num_keep_alive_refs}) must be a positive "
                f"integer"
            )
        if max_chunk_length is None:
            chunker_options = pybind.AutoTunedChunkerOptions(num_keep_alive_refs, 1.0)
        else:
            chunker_options = pybind.ConstantChunkerOptions(
                max_chunk_length=max_chunk_length,
                num_keep_alive_refs=num_keep_alive_refs,
            )
        cpp_writer = self._new_trajectory_writer(chunker_options, validate_items)
        # Imported here to avoid a circular import (trajectory_writer imports
        # pybind, not client) and to keep the module import TF-free.
        from reverb import (
            trajectory_writer as trajectory_writer_lib,  # pylint: disable=g-import-not-at-top
        )

        return trajectory_writer_lib.TrajectoryWriter(cpp_writer)

    def _new_trajectory_writer(self, chunker_options, validate_items: bool):
        """Constructs the C++ `TrajectoryWriter` from `chunker_options`.

        Subclasses dispatch to the transport-specific C++ binding. The gRPC
        `Client` passes `validate_items` through (its `NewTrajectoryWriter`
        fetches signatures when True); the in-process `LocalClient` and
        `ShmClient` ignore `validate_items` (their paths always populate
        `flat_signature_map` from the local/SHM tables).
        """
        raise NotImplementedError

    def structured_writer(self, configs):
        """Constructs a new `StructuredWriter`.

        Args:
          configs: Configurations describing how the writer should transform the
            sequence of steps into table insertions.

        Returns:
          A `StructuredWriter` that inserts items according to `configs`.

        Raises:
          ValueError: If `configs` is empty or contains an invalid config.
        """
        if not configs:
            raise ValueError("At least one config must be provided.")
        # All three C++ clients (`Client`, `InProcessClient`, `ShmClient`)
        # expose a `NewStructuredWriter(vector<string>)` binding that
        # re-parses the serialized protos internally, so the Python proto
        # objects are never handed to C++ directly.
        serialized_configs = [config.SerializeToString() for config in configs]
        cpp_writer = self._client.NewStructuredWriter(serialized_configs)
        from reverb import (
            structured_writer as structured_writer_lib,  # pylint: disable=g-import-not-at-top
        )

        return structured_writer_lib.StructuredWriter(cpp_writer)


class Client(_BaseClient):
    """Client for interacting with a Reverb ReverbService from Python.

    Note: This client should primarily be used when inserting data or prototyping
    at very small scale.
    """

    def __init__(self, server_address: str, *, output_format: str = "numpy"):
        """Constructor of Client.

        Args:
          server_address: Address to the Reverb ReverbService.
          output_format: "numpy" (default) or "torch"; see `_BaseClient`.
        """
        super().__init__(output_format=output_format)
        self._server_address = server_address
        self._client = pybind.Client(server_address)

    def __reduce__(self):
        return self.__class__, (self._server_address,)

    def __repr__(self):
        return f"Client, server_address={self._server_address}"

    @property
    def server_address(self) -> str:
        return self._server_address

    def _fetch_server_info_proto(self, timeout: Optional[int]):
        # gRPC `ServerInfo` takes a timeout in seconds (0 == wait forever).
        return self._client.ServerInfo(timeout or 0)

    def _new_sampler(
        self, table: str, num_samples: int, buffer_size: int, timeout_ms: Optional[int]
    ):
        # gRPC `NewSampler` has no rate-limiter timeout parameter; `timeout_ms`
        # is accepted for parity with `LocalClient` but ignored here.
        return self._client.NewSampler(table, num_samples, buffer_size)

    def _new_trajectory_writer(self, chunker_options, validate_items: bool):
        # gRPC `NewTrajectoryWriter` takes `(chunker_options, validate_items)`;
        # when `validate_items` is True it fetches table signatures with an
        # infinite timeout before constructing the writer.
        return self._client.NewTrajectoryWriter(chunker_options, validate_items)


class LocalClient(_BaseClient):
    """Python wrapper around the C++ `InProcessClient` for embedded mode.

    Provides a numpy-friendly API mirroring the gRPC `Client`: writers are
    context managers returned by `trajectory_writer`/`writer`, `sample` yields
    `ReplaySample` objects, and `insert`/`writer` are inherited from
    `_BaseClient` (single shared implementation, dispatched via `self.writer`
    / `self._client.NewWriter`). All data is exchanged as numpy arrays; no
    TensorFlow is required.

    The local writer is *not* bound to a single table: items are routed by
    their `table` field into the client's tables, so `trajectory_writer`/
    `structured_writer`/`writer`/`insert` all match the gRPC `Client` signatures
    exactly. `LocalClient` is not picklable (it holds in-process Table pointers).
    """

    def __init__(
        self, internal_client: "pybind.InProcessClient", *, output_format: str = "numpy"
    ):
        super().__init__(output_format=output_format)
        self._client = internal_client

    def __repr__(self):
        return "LocalClient (in-process, numpy)"

    def _new_trajectory_writer(self, chunker_options, validate_items: bool):
        # In-process `new_trajectory_writer` takes only `chunker_options`; it
        # always populates `flat_signature_map` from the local tables, so
        # `validate_items` is accepted for parity but has no effect here.
        return self._client.new_trajectory_writer(chunker_options)

    def new_sampler(
        self,
        table: str,
        num_samples: int = 1,
        buffer_size: int = 1,
        timeout_ms: Optional[int] = None,
    ):
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
        return self._client.new_sampler(table, num_samples, buffer_size, timeout_ms_arg)

    def _fetch_server_info_proto(self, timeout: Optional[int]):
        # In-process `ServerInfo` has no C++ timeout; `timeout` is accepted for
        # parity with the gRPC `Client` hook but ignored here. Uses the PascalCase
        # alias added in pybind.cc to exercise the unified naming.
        return self._client.ServerInfo()

    def _new_sampler(
        self, table: str, num_samples: int, buffer_size: int, timeout_ms: Optional[int]
    ):
        # -1 is the C++ sentinel for InfiniteDuration (see sampler.h). Uses the
        # PascalCase alias added in pybind.cc to exercise the unified naming.
        timeout_ms_arg = -1 if timeout_ms is None or timeout_ms < 0 else timeout_ms
        return self._client.NewSampler(table, num_samples, buffer_size, timeout_ms_arg)


class ShmClient(_BaseClient):
    """SHM (POSIX shared memory) client for same-machine cross-process access.

    Connects to a `Server(shm=True)` over a udsocket and mmaps the five SHM
    segments (pool + two ring pairs: insert C→S/S→C + sample C→S/S→C, one
    pair per flow so each flow keeps its own SPSC producer). `sample`/`trajectory_writer`/`structured_writer`
    match `Client`/`LocalClient` exactly, so user code is transport-agnostic.

    v1 scope (spec §6 + ticket ⑤): the C++ `ShmServer` dispatch handles
    SAMPLE/RELEASE/INSERT/ALLOCATE plus the control-plane ops
    `mutate_priorities`/`reset` (ticket ⑩) and `checkpoint` (ticket ⑪), all
    riding the insert flow under a client mutex. `server_info` returns a
    bootstrap-time snapshot (ticket ⑧); the plain `Writer` is unimplemented
    over SHM (use `trajectory_writer`/`structured_writer`). `ShmClient` is
    picklable: `__reduce__` stores `socket_path` and reconnects on load
    (ticket ⑫).
    """

    def __init__(self, socket_path: str, *, output_format: str = "numpy"):
        super().__init__(output_format=output_format)
        self._socket_path = socket_path
        # pybind.ShmClient.__init__ calls ShmClient::Connect (bootstrap + mmap);
        # raises on connection/handshake failure.
        self._client = pybind.ShmClient(socket_path)

    def __repr__(self):
        return f"ShmClient(socket_path={self._socket_path})"

    def __reduce__(self):
        # Pickle stores only the socket_path; unpickle re-connects via __init__
        # (which calls ShmClient::Connect: bootstrap + fresh mmap of the five
        # SHM segments). The old in-process mmap/ring/fd state stays in the
        # pickling process and is released when that process tears down the
        # original ShmClient — no leak across the pickle boundary.
        return self.__class__, (self._socket_path,)

    def writer(self, *args, **kwargs):
        # Legacy `Writer` (writer.h) has no SHM transport seam: its local ctor
        # takes a tables map the SHM client doesn't hold, and wiring SHM would
        # duplicate RunShmWorker for a deprecated API. Raise a clear Python-side
        # error instead of letting it reach the C++ UnimplementedError. Use
        # `trajectory_writer` / `structured_writer` for SHM inserts.
        raise NotImplementedError(
            "ShmClient does not support the legacy writer/insert; use "
            "trajectory_writer or structured_writer instead."
        )

    def insert(self, data, priorities: Dict[str, float]):
        # `insert` is sugar over `writer` (see _BaseClient.insert); the legacy
        # writer path has no SHM seam, so block it here with the same message.
        raise NotImplementedError(
            "ShmClient does not support the legacy writer/insert; use "
            "trajectory_writer or structured_writer instead."
        )

    def _fetch_server_info_proto(self, timeout: Optional[int]):
        # ticket ⑧ step 2: on-demand SERVER_INFO ring round-trip. `timeout` is
        # accepted for parity with the gRPC hook but ignored (the SHM round-trip
        # uses a hard cap, see ShmClient::ServerInfo). Returns live TableInfo
        # reflecting mid-session state (inserts, signature changes).
        return self._client.ServerInfo()

    def _new_sampler(
        self, table: str, num_samples: int, buffer_size: int, timeout_ms: Optional[int]
    ):
        timeout_ms_arg = -1 if timeout_ms is None or timeout_ms < 0 else timeout_ms
        return self._client.NewSampler(table, num_samples, buffer_size, timeout_ms_arg)

    def _new_trajectory_writer(self, chunker_options, validate_items: bool):
        # SHM `new_trajectory_writer` mirrors the in-process binding: it takes
        # only `chunker_options` and always populates `flat_signature_map`
        # from the cached bootstrap server_info (ticket ⑧-2b), so
        # `validate_items` is accepted for parity but has no effect here.
        return self._client.new_trajectory_writer(chunker_options)
