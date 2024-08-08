import tempfile
import os

import time
from orbax import checkpoint as ocp
from etils.epath import Path
from absl.testing import parameterized, absltest

from jax.random import normal, PRNGKey
from jax import numpy as jnp, tree


def max_file_with_size(path):
  """Return a tuple of (file name, size in bytes) of largest file in path."""
  root_path = Path(path)
  paths = sum([[Path(root) / fname for fname in fnames] 
                for root, _, fnames in os.walk(root_path)], [])
  paths_by_size = sorted({path.relative_to(root_path): path.stat().length 
                          for path in paths}.items(), key=lambda x: x[1])
  return paths_by_size[-1]
  
_UNIT_SIZE = 1024 ** 2
                      
# 4 * _UNITE_SIZE, use random to defeat compression if present
_DATA = {"r": normal(PRNGKey(round(time.time())), (_UNIT_SIZE,), 
                     dtype=jnp.float32)}


class OrbaxFileSizeTest(parameterized.TestCase):
  @parameterized.parameters([
    (False, None, None),
    (True, None, None),
    (False, 3 * _UNIT_SIZE, None),
    (False, 3 * _UNIT_SIZE, 2 * _UNIT_SIZE),
    (False, 1 * _UNIT_SIZE, 2 * _UNIT_SIZE),
    (False, None, 1 * _UNIT_SIZE),
    (True, 3 * _UNIT_SIZE, None),
    (True, 3 * _UNIT_SIZE, 2 * _UNIT_SIZE),
    (True, 1 * _UNIT_SIZE, 2 * _UNIT_SIZE),
    (True, None, 1 * _UNIT_SIZE),
  ])
  def test_file_size_created(self, use_zarr3, chunk_byte_size, ocdbt_target_byte_size):
    with tempfile.TemporaryDirectory() as tmpdir:
      mngr = ocp.CheckpointManager(Path(tmpdir), item_names=("items",), 
                                   item_handlers=dict(
                                     items=ocp.PyTreeCheckpointHandler(
                                       use_zarr3=use_zarr3)))

      if chunk_byte_size is not None:
        save_args = tree.map(lambda x: ocp.SaveArgs(chunk_byte_size=chunk_byte_size), _DATA)
      else:
        save_args = None
      args = ocp.args.Composite(
        items=ocp.args.PyTreeSave(item=_DATA, save_args=save_args, 
                                  ocdbt_target_data_file_size=ocdbt_target_byte_size))
      mngr.save(0, args=args)
      mngr.wait_until_finished()
      filename, filesize = max_file_with_size(tmpdir)
      msg = (f"\n\n{use_zarr3=} {chunk_byte_size=} {ocdbt_target_byte_size=}" 
             "\nFile: %60s, size (GB): %f\n")
      args = (filename, filesize / (1024 ** 3))
      absltest.logging.error(msg, *args)
      if ocdbt_target_byte_size is not None:
        target_size = 3 * ocdbt_target_byte_size 
      else:
        target_size = 3 * 1025 ** 3
      assert filesize < target_size, "File does not obey ocdbt_target_byte_size"


if __name__ == "__main__":
  absltest.logging.set_verbosity("error")
  absltest.main(testLoader=absltest.TestLoader())