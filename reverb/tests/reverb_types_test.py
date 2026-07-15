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

"""Tests for reverb.reverb_types.TableInfo."""

import numpy as np
from absl.testing import absltest

from reverb import item_selectors, rate_limiters, reverb_types, server, signature_codec

TensorSpec = signature_codec.TensorSpec


class TableInfoTest(absltest.TestCase):
    def _make_table(self, signature=None):
        return server.Table(
            name="t",
            sampler=item_selectors.Fifo(),
            remover=item_selectors.Fifo(),
            max_size=10,
            rate_limiter=rate_limiters.MinSize(1),
            signature=signature,
        )

    def test_from_serialized_proto_round_trip(self):
        table = self._make_table()
        info = table.info
        serialized = info.sampler_options.SerializeToString()
        # Reconstruct via the classmethod from a serialized TableInfo proto string.
        proto_string = table.internal_table.info()
        info2 = reverb_types.TableInfo.from_serialized_proto(proto_string)
        self.assertEqual(info2.name, "t")
        self.assertEqual(info2.max_size, 10)
        self.assertEqual(info2.sampler_options.SerializeToString(), serialized)

    def test_from_serialized_proto_with_signature(self):
        sig = {"a": TensorSpec([3], np.float32, "a")}
        table = self._make_table(signature=sig)
        info = table.info
        self.assertEqual(info.signature, sig)
        # Round-trip through the serialized form.
        proto_string = table.internal_table.info()
        info2 = reverb_types.TableInfo.from_serialized_proto(proto_string)
        self.assertEqual(info2.signature, sig)

    def test_from_serialized_proto_no_signature_is_none(self):
        table = self._make_table()
        info = table.info
        self.assertIsNone(info.signature)
        proto_string = table.internal_table.info()
        info2 = reverb_types.TableInfo.from_serialized_proto(proto_string)
        self.assertIsNone(info2.signature)

    def test_dataclass_fields(self):
        # TableInfo is a dataclass; check the expected fields exist.
        names = {f.name for f in reverb_types.TableInfo.__dataclass_fields__.values()}
        expected = {
            "name",
            "sampler_options",
            "remover_options",
            "max_size",
            "max_times_sampled",
            "rate_limiter_info",
            "signature",
            "current_size",
            "num_episodes",
            "num_deleted_episodes",
            "num_unique_samples",
            "table_worker_time",
        }
        self.assertEqual(names, expected)


if __name__ == "__main__":
    absltest.main()
