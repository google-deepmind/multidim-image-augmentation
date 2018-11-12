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


class ApplyDeformation2DTest(tf.test.TestCase):

  def test_IdentityTransform(self):
    with self.session():
      src = np.random.random([10, 7, 3]).astype(np.float32)
      deformation = np.ndarray([10, 7, 2], dtype=np.float32)

      for x0 in range(deformation.shape[0]):
        for x1 in range(deformation.shape[1]):
          deformation[x0, x1, 0] = x0
          deformation[x0, x1, 1] = x1

      result = augmentation_ops.apply_deformation2d(
          src, deformation, [])
      self.assertEqual(result.get_shape(), src.shape)
      trg = result.eval()

      self.assertAllEqual(trg, src)

  def test_ExtrapolationMirror(self):
    with self.session():
      src = np.array([[[0], [1], [2], [3], [4]]]).astype(np.float32)
      deform = np.array([[[0, -10], [0, -9], [0, -8], [0, -7], [0, -6],
                          [0, -5], [0, -4], [0, -3], [0, -2], [0, -1], [0, 0],
                          [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6],
                          [0, 7], [0, 8], [0, 9], [0, 10]]]).astype(np.float32)
      trg = augmentation_ops.apply_deformation2d(
          src, deform, []).eval()
      self.assertAllEqual(
          np.array([[[2], [1], [0], [1], [2], [3], [4], [3], [2], [1], [0],
                     [1], [2], [3], [4], [3], [2], [1], [0], [1], [2]]]), trg)

  def test_ExtrapolationZero(self):
    with self.session():
      src = np.array([[[10], [11], [12], [13], [14]]]).astype(np.float32)
      deform = np.array([[[0, -10], [0, -9], [0, -8], [0, -7], [0, -6],
                          [0, -5], [0, -4], [0, -3], [0, -2], [0, -1], [0, 0],
                          [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6],
                          [0, 7], [0, 8], [0, 9], [0, 10]]]).astype(np.float32)
      trg = augmentation_ops.apply_deformation2d(
          src, deform, [], extrapolation="zero_padding").eval()
      self.assertAllEqual(
          np.array([[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [10],
                     [11], [12], [13], [14], [0], [0], [0], [0], [0], [0]]]),
          trg)

  def test_ExtrapolationZeroMultichannel(self):
    with self.session():
      src = np.array([[[10, 9, 8, 7], [11, 10, 9, 8], [12, 11, 10, 9],
                       [13, 12, 11, 10], [14, 13, 12, 11]]]).astype(np.float32)
      deform = np.array([[[0, -10], [0, -9], [0, -8], [0, -7], [0, -6],
                          [0, -5], [0, -4], [0, -3], [0, -2], [0, -1], [0, 0],
                          [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6],
                          [0, 7], [0, 8], [0, 9], [0, 10]]]).astype(np.float32)
      trg = augmentation_ops.apply_deformation2d(
          src, deform, [], extrapolation="zero_padding").eval()
      self.assertAllEqual(
          np.array([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                     [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                     [0, 0, 0, 0], [0, 0, 0, 0], [10, 9, 8, 7], [11, 10, 9, 8],
                     [12, 11, 10, 9], [13, 12, 11, 10], [14, 13, 12, 11],
                     [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                     [0, 0, 0, 0], [0, 0, 0, 0]]]), trg)

  def test_ExtrapolationConst(self):
    with self.session():
      src = np.array([[[10], [11], [12], [13], [14]]]).astype(np.float32)
      deform = np.array([[[0, -10], [0, -9], [0, -8], [0, -7], [0, -6],
                          [0, -5], [0, -4], [0, -3], [0, -2], [0, -1], [0, 0],
                          [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6],
                          [0, 7], [0, 8], [0, 9], [0, 10]]]).astype(np.float32)
      trg = augmentation_ops.apply_deformation2d(
          src,
          deform,
          padding_constant=np.array([42]),

          extrapolation="const_padding").eval()
      self.assertAllEqual(
          np.array([[[42], [42], [42], [42], [42], [42], [42], [42], [42],
                     [42], [10], [11], [12], [13], [14], [42], [42], [42],
                     [42], [42], [42]]]), trg)

  def test_ExtrapolationConstMultichannel(self):
    with self.session():
      src = np.array([[[10, 9, 8, 7], [11, 10, 9, 8], [12, 11, 10, 9],
                       [13, 12, 11, 10], [14, 13, 12, 11]]]).astype(np.float32)
      deform = np.array([[[0, -10], [0, -9], [0, -8], [0, -7], [0, -6],
                          [0, -5], [0, -4], [0, -3], [0, -2], [0, -1], [0, 0],
                          [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6],
                          [0, 7], [0, 8], [0, 9], [0, 10]]]).astype(np.float32)
      trg = augmentation_ops.apply_deformation2d(
          src,
          deform,

          extrapolation="const_padding",
          padding_constant=np.array([1, 2, 3, 4])).eval()
      self.assertAllEqual(
          np.array([[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                     [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                     [1, 2, 3, 4], [1, 2, 3, 4], [10, 9, 8, 7], [11, 10, 9, 8],
                     [12, 11, 10, 9], [13, 12, 11, 10], [14, 13, 12, 11],
                     [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                     [1, 2, 3, 4], [1, 2, 3, 4]]]), trg)


class ApplyDeformation3DTest(tf.test.TestCase):

  def test_IdentityTransform(self):
    with self.session():
      src = np.random.random([4, 10, 7, 3]).astype(np.float32)
      deformation = np.ndarray([4, 10, 7, 3], dtype=np.float32)

      for x0 in range(deformation.shape[0]):
        for x1 in range(deformation.shape[1]):
          for x2 in range(deformation.shape[2]):
            deformation[x0, x1, x2, 0] = x0
            deformation[x0, x1, x2, 1] = x1
            deformation[x0, x1, x2, 2] = x2

      result = augmentation_ops.apply_deformation3d(
          src, deformation, [])
      self.assertEqual(result.get_shape(), src.shape)
      trg = result.eval()
      self.assertAllEqual(trg, src)

  def test_InterpolationNearest(self):
    with self.session():
      src = np.array([[[[0], [10], [20], [30]]]]).astype(np.float32)
      deform = np.array([[[[0, 0, 0.5], [0, 0, 2.7]]]]).astype(np.float32)
      trg = augmentation_ops.apply_deformation3d(
          src, deform, [], interpolation="nearest").eval()
      self.assertAllEqual(
          np.array([[[[10], [30]]]]), trg)

  def test_InterpolationMixedNearestLinear(self):
    with self.session():
      src = np.array([[[[0], [10], [20], [30]]],
                      [[[5], [15], [25], [35]]]]).astype(np.float32)
      deform = np.array([[[[0, 0, 0.5], [0, 0, 2.7]],
                          [[0, 1, 1.5], [1, 0, 2.1]]]]).astype(np.float32)
      trg = augmentation_ops.apply_deformation3d(
          src,
          deform, [],
          interpolation="mixed_nearest_linear",
          extrapolation="zero_padding").eval()
      self.assertAllClose(np.array([[[[5], [27]], [[0], [26]]]]), trg)

  def test_ExtrapolationMirror(self):
    with self.session():
      src = np.array([[[[0], [1], [2], [3], [4]]]]).astype(np.float32)
      deform = np.array([[[[0, 0, -10], [0, 0, -9], [0, 0, -8], [0, 0, -7],
                           [0, 0, -6], [0, 0, -5], [0, 0, -4], [0, 0, -3],
                           [0, 0, -2], [0, 0, -1], [0, 0, 0], [0, 0, 1],
                           [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5],
                           [0, 0, 6], [0, 0, 7], [0, 0, 8], [0, 0, 9],
                           [0, 0, 10]]]]).astype(np.float32)
      trg = augmentation_ops.apply_deformation3d(
          src, deform, []).eval()
      self.assertAllEqual(
          np.array([[[[2], [1], [0], [1], [2], [3], [4], [3], [2], [1], [0],
                      [1], [2], [3], [4], [3], [2], [1], [0], [1], [2]]]]), trg)

  def test_ExtrapolationZero(self):
    with self.session():
      src = np.array([[[[10], [11], [12], [13], [14]]]]).astype(np.float32)
      deform = np.array([[[[0, 0, -10], [0, 0, -9], [0, 0, -8], [0, 0, -7],
                           [0, 0, -6], [0, 0, -5], [0, 0, -4], [0, 0, -3],
                           [0, 0, -2], [0, 0, -1], [0, 0, 0], [0, 0, 1],
                           [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5],
                           [0, 0, 6], [0, 0, 7], [0, 0, 8], [0, 0, 9],
                           [0, 0, 10]]]]).astype(np.float32)
      trg = augmentation_ops.apply_deformation3d(
          src, deform, [], extrapolation="zero_padding").eval()
      self.assertAllEqual(
          np.array([[[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [10],
                      [11], [12], [13], [14], [0], [0], [0], [0], [0], [0]]]]),
          trg)

  def test_ExtrapolationConst(self):
    with self.session():
      src = np.array([[[[10], [11], [12], [13], [14]]]]).astype(np.float32)
      deform = np.array([[[[0, 0, -10], [0, 0, -9], [0, 0, -8], [0, 0, -7],
                           [0, 0, -6], [0, 0, -5], [0, 0, -4], [0, 0, -3],
                           [0, 0, -2], [0, 0, -1], [0, 0, 0], [0, 0, 1],
                           [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5],
                           [0, 0, 6], [0, 0, 7], [0, 0, 8], [0, 0, 9],
                           [0, 0, 10]]]]).astype(np.float32)

      trg = augmentation_ops.apply_deformation3d(
          src,
          deform,
          padding_constant=np.array([42]),
          extrapolation="const_padding").eval()
      self.assertAllEqual(
          np.array([[[[42], [42], [42], [42], [42], [42], [42], [42], [42],
                      [42], [10], [11], [12], [13], [14], [42], [42], [42],
                      [42], [42], [42]]]]), trg)

  def test_One_Hot_Encoding(self):
    with self.session():
      src = np.array([[[[4], [3], [1], [0], [2]]]]).astype(np.float32)
      deform = np.array([[[[0, 0, -.5], [0, 0, 0], [0, 0, 0.3], [0, 0, 1],
                           [0, 0, 1.5], [0, 0, 2.5], [0, 0, 4],
                           [0, 0, 5]]]]).astype(np.float32)

      trg_graph = augmentation_ops.apply_deformation3d(
          src,
          deform, [],
          extrapolation="zero_padding",
          conversion="indexed_to_one_hot",
          output_num_channels=5)
      trg = trg_graph.eval()

      self.assertAllEqual([1, 1, 8, 5], trg_graph.shape)
      self.assertAllEqual([1, 1, 8, 5], trg.shape)
      expected = np.array([[[[0.5, 0, 0, 0, 0.5], [0, 0, 0, 0, 1],
                             [0, 0, 0, 0.3, 0.7], [0, 0, 0, 1, 0],
                             [0, 0.5, 0, 0.5, 0], [0.5, 0.5, 0, 0, 0],
                             [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]]]]).astype(float)
      for x2 in range(8):
        for ch in range(5):
          self.assertAlmostEqual(
              expected[0, 0, x2, ch],
              trg[0, 0, x2, ch],
              msg="expected {}, but got {} at x2={}, ch={}".format(
                  expected[0, 0, x2, ch], trg[0, 0, x2, ch], x2, ch))

  def test_outputSpatialShape(self):
    with self.session():
      src = np.random.random([4, 10, 7, 3]).astype(np.float32)
      deformation = np.ndarray([4, 10, 7, 3], dtype=np.float32)

      for x0 in range(deformation.shape[0]):
        for x1 in range(deformation.shape[1]):
          for x2 in range(deformation.shape[2]):
            deformation[x0, x1, x2, 0] = x0
            deformation[x0, x1, x2, 1] = x1
            deformation[x0, x1, x2, 2] = x2

      result = augmentation_ops.apply_deformation3d(
          src, deformation, [],
          output_spatial_shape=[-1, 6, 5])
      self.assertEqual(result.get_shape(), [4, 6, 5, 3])
      trg = result.eval()
      self.assertAllEqual(trg, src[:, 2:-2, 1:-1, :])


if __name__ == "__main__":
  tf.test.main()
