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

  def test_1DInterpolation(self):
    with self.session():
      grid = np.ndarray([5, 2], dtype=np.float32)
      for x0 in range(grid.shape[0]):
        for channel in range(grid.shape[1]):
          grid[x0, channel] = x0 * grid.shape[1] + channel

      dense = augmentation_ops.cubic_interpolation1d(
          input=grid, factor=10, output_length=21).eval()
      precision = 5
      self.assertAlmostEqual(grid[1, 0], dense[0, 0], precision)
      self.assertAlmostEqual(grid[2, 0], dense[10, 0], precision)
      self.assertAlmostEqual(grid[3, 0], dense[20, 0], precision)
      self.assertAlmostEqual(grid[1, 1], dense[0, 1], precision)
      self.assertAlmostEqual(grid[2, 1], dense[10, 1], precision)
      self.assertAlmostEqual(grid[3, 1], dense[20, 1], precision)

  def test_1DInterpolationFull(self):
    with self.session():
      grid = np.ndarray([5, 2], dtype=np.float32)
      for x0 in range(grid.shape[0]):
        for channel in range(grid.shape[1]):
          grid[x0, channel] = x0 * grid.shape[1] + channel

      dense_op = augmentation_ops.cubic_interpolation1d(grid, 10)
      self.assertAllEqual([41, 2], dense_op.get_shape().as_list())
      dense = dense_op.eval()
      precision = 5
      self.assertAlmostEqual(grid[0, 0], dense[0, 0], precision)
      self.assertAlmostEqual(grid[1, 0], dense[10, 0], precision)
      self.assertAlmostEqual(grid[2, 0], dense[20, 0], precision)
      self.assertAlmostEqual(grid[3, 0], dense[30, 0], precision)
      self.assertAlmostEqual(grid[4, 0], dense[40, 0], precision)
      self.assertAlmostEqual(grid[3, 1], dense[30, 1], precision)
      self.assertAlmostEqual(grid[2, 1], dense[20, 1], precision)
      self.assertAlmostEqual(grid[0, 1], dense[0, 1], precision)

  def test_OddEvenError(self):
    with self.session():
      odd_even = augmentation_ops.cubic_interpolation1d(
          np.ndarray([1, 1]), 2, output_length=2)
      with self.assertRaisesWithPredicateMatch(
          tf.errors.InvalidArgumentError,
          "output size and input size must both be odd or both be even"):
        odd_even.eval()

  def test_EvenOddError(self):
    with self.session():
      even_odd = augmentation_ops.cubic_interpolation1d(
          np.ndarray([2, 1]), 2, output_length=1)
      with self.assertRaisesWithPredicateMatch(
          tf.errors.InvalidArgumentError,
          "output size and input size must both be odd or both be even"):
        even_odd.eval()

  def test_AllEvenError(self):
    with self.session():
      all_even = augmentation_ops.cubic_interpolation1d(
          np.ndarray([2, 1]), 2, output_length=2)
      with self.assertRaisesWithPredicateMatch(
          tf.errors.InvalidArgumentError,
          "factor must be odd as input and output size is even"):
        all_even.eval()


if __name__ == "__main__":
  tf.test.main()
