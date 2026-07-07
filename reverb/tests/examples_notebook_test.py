"""Validates that the example notebooks' code runs end-to-end (numpy, no TF).

Extracts code cells from examples/demo.ipynb and examples/frame_stacking.ipynb
and executes them in order. Temporary; can be deleted once the notebooks are
confirmed.
"""

import json
import os
import sys
from absl.testing import absltest

import numpy as np  # pylint: disable=unused-import

import reverb

_EXAMPLES_DIR = os.path.join(
    os.environ['TEST_SRCDIR'], 'reverb', 'examples')


def _code_cells(nb_path, skip_install=True):
  nb = json.load(open(nb_path))
  cells = []
  for c in nb['cells']:
    if c['cell_type'] != 'code':
      continue
    src = ''.join(c['source'])
    if skip_install and src.lstrip().startswith('!pip'):
      continue
    cells.append(src)
  return cells


class DemoNotebookTest(absltest.TestCase):

  def test_demo_notebook_cells_run(self):
    cells = _code_cells(os.path.join(_EXAMPLES_DIR, 'demo.ipynb'))
    self.assertGreaterEqual(len(cells), 10)
    g = {'np': np, 'reverb': reverb}
    for i, src in enumerate(cells):
      with self.subTest(cell=i):
        exec(compile(src, f'<demo cell {i}>', 'exec'), g)  # pylint: disable=exec-used


class FrameStackingNotebookTest(absltest.TestCase):

  def test_frame_stacking_notebook_cells_run(self):
    cells = _code_cells(os.path.join(_EXAMPLES_DIR, 'frame_stacking.ipynb'))
    self.assertGreaterEqual(len(cells), 4)
    g = {'np': np, 'reverb': reverb}
    for i, src in enumerate(cells):
      with self.subTest(cell=i):
        exec(compile(src, f'<frame cell {i}>', 'exec'), g)  # pylint: disable=exec-used


if __name__ == '__main__':
  absltest.main()
