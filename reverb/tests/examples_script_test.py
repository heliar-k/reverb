# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression test: runs the standalone example scripts end-to-end.

Executes every self-contained `examples/*.py` script; each carries assertions
or prints expected output, so a regression surfaces as an exception.
Mirrors `examples_notebook_test.py`'s exec-the-source approach.
"""

import os
from absl.testing import absltest, parameterized

import numpy as np  # pylint: disable=unused-import
import reverb
import tree

_EXAMPLES_DIR = os.path.join(os.environ['TEST_SRCDIR'], 'reverb', 'examples')

_EXAMPLES = [
    'demo.py',
    'grpc_client.py',
    'multi_server.py',
    'production_patterns.py',
    'shm_client.py',
    'structured_writer.py',
    'structured_writer_advanced.py',
    'table_signature.py',
    'training_loop.py',
]


class ExamplesScriptTest(parameterized.TestCase):
  """Runs each standalone example script in-process and asserts it succeeds."""

  @parameterized.parameters(_EXAMPLES)
  def test_example_runs(self, script):
    path = os.path.join(_EXAMPLES_DIR, script)
    with open(path) as f:
      src = f.read()
    # Execute the module body (imports + def main) in a fresh namespace, then
    # call main(). __name__ is set so the `if __name__ == '__main__'` guard does
    # not auto-run main (we invoke it explicitly afterwards).
    g = {'__name__': 'examples_script', 'np': np, 'reverb': reverb, 'tree': tree}
    exec(compile(src, path, 'exec'), g)  # pylint: disable=exec-used
    g['main']()


if __name__ == '__main__':
  absltest.main()
