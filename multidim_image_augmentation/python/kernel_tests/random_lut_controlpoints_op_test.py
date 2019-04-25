# Lint as: python2, python3
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

from six.moves import range
import tensorflow as tf

from multidim_image_augmentation import augmentation_ops


class RandomLUTControlPointsTest(tf.test.TestCase):

  def testBasic1(self):
    with self.session():
      graph = augmentation_ops.random_lut_control_points(
          new_black_range=[-0.1, 0.1],
          new_white_range=[0.9, 1.1],
          slope_min=0.7,
          slope_max=1.4,
          num_control_point_insertions=2)

      for _ in range(10):
        lut_control_points = graph.eval()
        self.assertEqual(lut_control_points.shape[0], 5)
        tf.logging.info(lut_control_points)
        slopes = (lut_control_points[1:] - lut_control_points[0:-1]) / 0.25
        for i in range(4):
          self.assertGreaterEqual(slopes[i], 0.7)
          self.assertLessEqual(slopes[i], 1.4)

  def testBasic2(self):
    with self.session():
      graph = augmentation_ops.random_lut_control_points(
          new_black_range=[-0.1, 0.1],
          new_white_range=[0.9, 1.1],
          slope_min=0.7,
          slope_max=1.4,
          num_control_point_insertions=3)

      for _ in range(10):
        lut_control_points = graph.eval()
        self.assertEqual(lut_control_points.shape[0], 9)
        slopes = (lut_control_points[1:] - lut_control_points[0:-1]) / 0.125
        for i in range(4):
          self.assertGreaterEqual(slopes[i], 0.7)
          self.assertLessEqual(slopes[i], 1.4)

  def testNotOptimizedAway(self):
    with self.session() as sess:
      lut = augmentation_ops.random_lut_control_points(
          new_black_range=[-0.1, 0.1],
          new_white_range=[0.9, 1.1],
          slope_min=0.7,
          slope_max=1.4,
          num_control_point_insertions=2)
      graph = lut + 1
      for _ in range(10):
        lut_control_points1 = sess.run(graph)
        lut_control_points2 = sess.run(graph)
        for i in range(len(lut_control_points2)):
          self.assertNotEqual(lut_control_points1[i], lut_control_points2[i])


if __name__ == "__main__":
  tf.test.main()
