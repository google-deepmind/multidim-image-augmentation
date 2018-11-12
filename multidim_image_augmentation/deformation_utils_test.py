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

import tensorflow as tf
from multidim_image_augmentation import deformation_utils


class ControlGridTest(tf.test.TestCase):

  def test_create_control_grid_for_cubic_interp_2d(self):
    with self.session():
      grid = deformation_utils.create_control_grid_for_cubic_interp(
          transformed_image_shape=[20, 30],
          transformed_image_spacings_um=tf.constant([0.1, 0.1]),
          control_grid_spacings_pix=[9, 9])
      self.assertAllEqual([6, 8, 2], grid.eval().shape)

  def test_create_control_grid_for_cubic_interp_3d(self):
    with self.session():
      grid = deformation_utils.create_control_grid_for_cubic_interp(
          transformed_image_shape=[10, 20, 30],
          transformed_image_spacings_um=tf.constant([0.1, 0.1, 0.1]),
          control_grid_spacings_pix=[9, 9, 9])
      self.assertAllEqual([4, 6, 8, 3], grid.eval().shape)

  def test_create_control_grid_for_cubic_interp_3d_single_slice(self):
    with self.session():
      grid = deformation_utils.create_control_grid_for_cubic_interp(
          transformed_image_shape=[1, 20, 30],
          transformed_image_spacings_um=tf.constant([0.1, 0.1, 0.1]),
          control_grid_spacings_pix=[1, 9, 9])
      self.assertAllEqual([3, 6, 8, 3], grid.eval().shape)


if __name__ == "__main__":
  tf.test.main()
