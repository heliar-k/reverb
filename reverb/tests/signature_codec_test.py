# Copyright 2024 DeepMind Technologies Limited.
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

"""Exhaustive tests for reverb.signature_codec (pure-Python, no TF)."""

from absl.testing import absltest
from absl.testing import parameterized
import collections
import numpy as np

from reverb import signature_codec

TensorSpec = signature_codec.TensorSpec


class TensorSpecTest(absltest.TestCase):

  def test_construct_basic(self):
    spec = TensorSpec([3, 3], np.float32, 'a')
    self.assertEqual(spec.shape, (3, 3))
    self.assertEqual(spec.dtype, np.dtype(np.float32))
    self.assertEqual(spec.name, 'a')

  def test_name_defaults_none(self):
    spec = TensorSpec([], np.int32)
    self.assertIsNone(spec.name)

  def test_shape_normalized_to_tuple(self):
    spec = TensorSpec([1, 2, 3], np.float32)
    self.assertEqual(spec.shape, (1, 2, 3))

  def test_unknown_dim_none(self):
    spec = TensorSpec([None, 2], np.int64)
    self.assertEqual(spec.shape, (None, 2))

  def test_dtype_accepts_string(self):
    spec = TensorSpec([], 'float32')
    self.assertEqual(spec.dtype, np.dtype(np.float32))

  def test_dtype_accepts_np_scalar_type(self):
    spec = TensorSpec([], np.int32)
    self.assertEqual(spec.dtype, np.dtype(np.int32))

  def test_equality(self):
    a = TensorSpec([2], np.float32, 'a')
    b = TensorSpec([2], np.float32, 'a')
    c = TensorSpec([2], np.float32, 'b')
    d = TensorSpec([3], np.float32, 'a')
    e = TensorSpec([2], np.float64, 'a')
    self.assertEqual(a, b)
    self.assertNotEqual(a, c)
    self.assertNotEqual(a, d)
    self.assertNotEqual(a, e)

  def test_equality_not_tensor_spec(self):
    self.assertNotEqual(TensorSpec([], np.float32), 'not a spec')

  def test_hashable(self):
    a = TensorSpec([2], np.float32, 'a')
    s = {a}
    s.add(TensorSpec([2], np.float32, 'a'))
    self.assertEqual(len(s), 1)

  def test_repr(self):
    spec = TensorSpec([2], np.float32, 'a')
    r = repr(spec)
    self.assertIn('TensorSpec', r)
    self.assertIn('float32', r)

  def test_slots_no_dict(self):
    spec = TensorSpec([], np.float32)
    with self.assertRaises(AttributeError):
      spec.foo = 1  # __slots__ forbids new attrs.


class EncodeDecodeRoundTripTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('scalar', TensorSpec([], np.float32, 's')),
      ('vector', TensorSpec([3], np.int32, 'v')),
      ('matrix', TensorSpec([2, 2], np.float64, 'm')),
      ('unknown_dim', TensorSpec([None, 4], np.int64, 'u')),
      ('all_unknown', TensorSpec([None, None], np.float32, 'a')),
  )
  def test_single_leaf(self, spec):
    encoded = signature_codec.encode_signature(spec)
    decoded = signature_codec.decode_signature(encoded)
    self.assertEqual(decoded, spec)

  @parameterized.parameters(
      np.float32, np.float64, np.int8, np.int16, np.int32, np.int64,
      np.uint8, np.uint16, np.uint32, np.uint64, np.bool_,
      np.complex64, np.complex128,
  )
  def test_each_dtype_round_trips(self, np_dtype):
    spec = TensorSpec([2], np_dtype, 'd')
    decoded = signature_codec.decode_signature(
        signature_codec.encode_signature(spec))
    self.assertEqual(decoded.dtype, np.dtype(np_dtype))

  def test_dict_nested(self):
    spec = {
        'a': TensorSpec([3, 3], np.float32, 'a'),
        'b': {
            'c': TensorSpec([], np.int32, 'b/c'),
        },
    }
    decoded = signature_codec.decode_signature(
        signature_codec.encode_signature(spec))
    self.assertEqual(decoded, spec)

  def test_list_nested(self):
    spec = [TensorSpec([1], np.int32), TensorSpec([2], np.float32)]
    decoded = signature_codec.decode_signature(
        signature_codec.encode_signature(spec))
    self.assertEqual(decoded, spec)

  def test_tuple_nested(self):
    spec = (TensorSpec([1], np.int32), TensorSpec([2], np.float32))
    decoded = signature_codec.decode_signature(
        signature_codec.encode_signature(spec))
    self.assertEqual(decoded, spec)

  def test_mixed_nested(self):
    spec = {
        'a': TensorSpec([3, 3], np.float32, 'a'),
        'b': {
            'c': TensorSpec([], np.int32, 'b/c'),
            'd': [TensorSpec([None, 2], np.int64, 'b/d/0'),
                  TensorSpec([6], np.uint8, 'b/d/1')],
        },
        'e': (TensorSpec([1], np.bool_), TensorSpec([], np.float64)),
    }
    decoded = signature_codec.decode_signature(
        signature_codec.encode_signature(spec))
    self.assertEqual(decoded, spec)

  def test_namedtuple_leaf(self):
    nt = collections.namedtuple('Step', ['x', 'y'])
    spec = nt(x=TensorSpec([2], np.float32, 'x'),
              y=TensorSpec([], np.int32, 'y'))
    decoded = signature_codec.decode_signature(
        signature_codec.encode_signature(spec))
    # namedtuple decodes to an equivalent namedtuple with same fields/values.
    self.assertEqual(decoded._fields, spec._fields)
    self.assertEqual(decoded.x, spec.x)
    self.assertEqual(decoded.y, spec.y)


class EncodeDecodeErrorTest(absltest.TestCase):

  def test_encode_unsupported_leaf_raises(self):
    # A dict value that is neither TensorSpec nor a container falls through
    # to the 'Unsupported signature node' path (not _encode_leaf, since the
    # value isn't a TensorSpec to begin with).
    with self.assertRaisesRegex(ValueError, 'Unsupported signature node'):
      signature_codec.encode_signature({'a': 123})

  def test_encode_unsupported_node_raises(self):
    with self.assertRaisesRegex(ValueError, 'Unsupported signature node'):
      # A bare int is neither TensorSpec nor a container.
      signature_codec.encode_signature(123)

  def test_encode_unsupported_dtype_raises(self):
    # float16 is intentionally NOT in the dtype table.
    spec = TensorSpec([], np.float16)
    with self.assertRaisesRegex(ValueError, 'Unsupported numpy dtype'):
      signature_codec.encode_signature(spec)


if __name__ == '__main__':
  absltest.main()
