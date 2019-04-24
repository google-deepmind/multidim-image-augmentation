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

  def test_2DInterpolation(self):
    with self.session():
      grid = np.ndarray([5, 5, 2], dtype=np.float32)
      c = 0
      for x0 in range(grid.shape[0]):
        for x1 in range(grid.shape[1]):
          for channel in range(grid.shape[2]):
            grid[x0, x1, channel] = c
            c += 1

      dense = augmentation_ops.cubic_interpolation2d(
          input=grid, factors=[10, 10], output_spatial_shape=[21, 21]).eval()
      precision = 5
      self.assertAlmostEqual(grid[1, 1, 0], dense[0, 0, 0], precision)
      self.assertAlmostEqual(grid[2, 2, 0], dense[10, 10, 0], precision)
      self.assertAlmostEqual(grid[3, 3, 0], dense[20, 20, 0], precision)
      self.assertAlmostEqual(grid[1, 1, 1], dense[0, 0, 1], precision)
      self.assertAlmostEqual(grid[2, 2, 1], dense[10, 10, 1], precision)
      self.assertAlmostEqual(grid[3, 3, 1], dense[20, 20, 1], precision)

  def testFactorAttrLengthErrors(self):
    with self.session():
      with self.assertRaisesWithPredicateMatch(ValueError,
                                               "factors must be rank 2, got 3"):
        augmentation_ops.cubic_interpolation2d(
            np.ndarray([1, 1, 1]),
            factors=[3, 4, 5],
            output_spatial_shape=[8, 9]).eval()

  def testOutputSpatialLengthAttrLengthErrors(self):
    with self.session():
      with self.assertRaisesWithPredicateMatch(
          ValueError, "output_spatial_shape must be rank 2, got 3"):
        augmentation_ops.cubic_interpolation2d(
            np.ndarray([1, 1, 1]),
            factors=[3, 4],
            output_spatial_shape=[7, 8, 9]).eval()


if __name__ == "__main__":
  tf.test.main()
