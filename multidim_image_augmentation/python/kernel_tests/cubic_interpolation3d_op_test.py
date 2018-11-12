# Copyright 2018 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from multidim_image_augmentation import augmentation_ops


class CubicInterpolationTest(tf.test.TestCase):

  def test_3DInterpolation(self):
    with self.session():
      grid = np.ndarray([5, 5, 5, 2], dtype=np.float32)
      c = 0
      for x0 in range(grid.shape[0]):
        for x1 in range(grid.shape[1]):
          for x2 in range(grid.shape[2]):
            for channel in range(grid.shape[3]):
              grid[x0, x1, x2, channel] = c
              c += 1

      dense = augmentation_ops.cubic_interpolation3d(
          input=grid,
          factors=[10, 10, 10],
          output_spatial_shape=[21, 21, 21]).eval()
      precision = 4
      self.assertAlmostEqual(grid[1, 1, 1, 0], dense[0, 0, 0, 0], precision)
      self.assertAlmostEqual(grid[1, 1, 3, 0], dense[0, 0, 20, 0], precision)
      self.assertAlmostEqual(grid[1, 3, 1, 0], dense[0, 20, 0, 0], precision)
      self.assertAlmostEqual(grid[3, 1, 1, 0], dense[20, 0, 0, 0], precision)
      self.assertAlmostEqual(grid[2, 2, 2, 0], dense[10, 10, 10, 0], precision)
      self.assertAlmostEqual(grid[3, 3, 3, 0], dense[20, 20, 20, 0], precision)
      self.assertAlmostEqual(grid[1, 1, 1, 1], dense[0, 0, 0, 1], precision)
      self.assertAlmostEqual(grid[1, 1, 3, 1], dense[0, 0, 20, 1], precision)
      self.assertAlmostEqual(grid[1, 3, 1, 1], dense[0, 20, 0, 1], precision)
      self.assertAlmostEqual(grid[3, 1, 1, 1], dense[20, 0, 0, 1], precision)
      self.assertAlmostEqual(grid[2, 2, 2, 1], dense[10, 10, 10, 1], precision)
      self.assertAlmostEqual(grid[3, 3, 3, 1], dense[20, 20, 20, 1], precision)

  def test_3DInterpolationSingleSlice(self):
    with self.session():
      grid = np.ndarray([3, 5, 5, 2], dtype=np.float32)
      c = 0
      for x0 in range(grid.shape[0]):
        for x1 in range(grid.shape[1]):
          for x2 in range(grid.shape[2]):
            for channel in range(grid.shape[3]):
              grid[x0, x1, x2, channel] = c
              c += 1

      dense = augmentation_ops.cubic_interpolation3d(
          input=grid,
          factors=[1, 10, 10],
          output_spatial_shape=[1, 21, 21],
      ).eval()
      precision = 4
      self.assertAlmostEqual(grid[1, 1, 1, 0], dense[0, 0, 0, 0], precision)
      self.assertAlmostEqual(grid[1, 1, 3, 0], dense[0, 0, 20, 0], precision)
      self.assertAlmostEqual(grid[1, 3, 1, 0], dense[0, 20, 0, 0], precision)


if __name__ == "__main__":
  tf.test.main()
