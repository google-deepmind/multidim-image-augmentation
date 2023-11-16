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
import tensorflow.compat.v1 as tf

from multidim_image_augmentation import augmentation_ops


class ApplyTabulatedFunctionTest(tf.test.TestCase):

  def testShape(self):
    image = np.random.random([10, 7, 3]).astype(np.float32)
    tabulated_functions = np.random.random([3, 256]).astype(np.float32)
    out_image = augmentation_ops.apply_tabulated_functions(
        image, tabulated_functions)
    self.assertAllEqual(out_image.get_shape(), image.shape)

  def testBasicUsage(self):
    with self.session():
      image = np.array([[[5], [2], [4], [0], [3], [5], [2], [4], [0],
                         [3]]]).astype(np.float32)
      tabulated_functions = np.array([[10, 11, 12, 13, 14, 15, 16,
                                       17]]).astype(np.float32)
      out_image = augmentation_ops.apply_tabulated_functions(
          image, tabulated_functions)
      self.assertAllEqual(
          np.array([[[15], [12], [14], [10], [13], [15], [12], [14], [10],
                     [13]]]), out_image.eval())

  def testInterpolationExtrapolation(self):
    with self.session():
      image = np.array([[[-1], [2.7], [4.2], [0.3], [8.5]]]).astype(np.float32)
      tabulated_functions = np.array([[10, 11, 12, 13, 14, 15, 16,
                                       17]]).astype(np.float32)
      out_image = augmentation_ops.apply_tabulated_functions(
          image, tabulated_functions)
      self.assertAllClose(
          np.array([[[9], [12.7], [14.2], [10.3], [18.5]]]), out_image.eval())

  def testMult(self):
    with self.session():
      image = np.array([[[5, 2], [2, 4], [4, 1], [0, 1],
                         [3, 0]]]).astype(np.float32)
      tabulated_functions = np.array([[10, 11, 12, 13, 14, 15, 16, 17],
                                      [0, 10, 20, 30, 40, 50, 60,
                                       70]]).astype(np.float32)
      out_image = augmentation_ops.apply_tabulated_functions(
          image, tabulated_functions)
      self.assertAllEqual(
          np.array([[[15, 20], [12, 40], [14, 10], [10, 10], [13, 0]]]),
          out_image.eval())


if __name__ == "__main__":
  tf.test.main()
