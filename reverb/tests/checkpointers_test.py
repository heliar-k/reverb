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

"""Tests for reverb.platform.default.checkpointers."""

import os
import tempfile

from absl.testing import absltest

from reverb import pybind
from reverb.platform.default import checkpointers


class CheckpointerBaseTest(absltest.TestCase):
    def test_base_is_abstract(self):
        # CheckpointerBase is ABC; internal_checkpointer is abstract.
        with self.assertRaises(TypeError):
            checkpointers.CheckpointerBase()


class TempDirCheckpointerTest(absltest.TestCase):
    def test_constructs_internal_checkpointer(self):
        cp = checkpointers.TempDirCheckpointer()
        self.assertIsInstance(cp.internal_checkpointer(), pybind.Checkpointer)

    def test_path_is_tempdir(self):
        cp = checkpointers.TempDirCheckpointer()
        self.assertTrue(os.path.isdir(cp.path))
        # A fresh temp dir should differ between instances.
        cp2 = checkpointers.TempDirCheckpointer()
        self.assertNotEqual(cp.path, cp2.path)


class DefaultCheckpointerTest(absltest.TestCase):
    def test_constructs_internal_checkpointer(self):
        root = tempfile.mkdtemp()
        cp = checkpointers.DefaultCheckpointer(path=root)
        self.assertIsInstance(cp.internal_checkpointer(), pybind.Checkpointer)

    def test_path_stored(self):
        root = tempfile.mkdtemp()
        cp = checkpointers.DefaultCheckpointer(path=root)
        self.assertEqual(cp.path, root)

    def test_group_default_empty(self):
        root = tempfile.mkdtemp()
        cp = checkpointers.DefaultCheckpointer(path=root)
        self.assertEqual(cp.group, "")

    def test_group_custom(self):
        root = tempfile.mkdtemp()
        cp = checkpointers.DefaultCheckpointer(path=root, group="mygroup")
        self.assertEqual(cp.group, "mygroup")

    def test_fallback_path_default_none(self):
        root = tempfile.mkdtemp()
        cp = checkpointers.DefaultCheckpointer(path=root)
        self.assertIsNone(cp.fallback_checkpoint_path)

    def test_fallback_path_custom(self):
        root = tempfile.mkdtemp()
        cp = checkpointers.DefaultCheckpointer(
            path=root, fallback_checkpoint_path="/some/ckpt"
        )
        self.assertEqual(cp.fallback_checkpoint_path, "/some/ckpt")

    def test_is_checkpointer_base(self):
        root = tempfile.mkdtemp()
        cp = checkpointers.DefaultCheckpointer(path=root)
        self.assertIsInstance(cp, checkpointers.CheckpointerBase)


class TempDirIsDefaultCheckpointerTest(absltest.TestCase):
    def test_tempdir_inherits_default(self):
        cp = checkpointers.TempDirCheckpointer()
        self.assertIsInstance(cp, checkpointers.DefaultCheckpointer)


class DefaultCheckpointerFacadeTest(absltest.TestCase):
    def test_default_checkpointer_returns_tempdir(self):
        cp = checkpointers.default_checkpointer()
        self.assertIsInstance(cp, checkpointers.TempDirCheckpointer)
        self.assertIsInstance(cp, checkpointers.CheckpointerBase)


if __name__ == "__main__":
    absltest.main()
