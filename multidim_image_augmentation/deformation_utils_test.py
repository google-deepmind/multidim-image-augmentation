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

import numpy as np
import tensorflow.compat.v1 as tf
from multidim_image_augmentation import deformation_utils


_ARRAY_COMPARE_TOLERANCE = 1e-5


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


class Create2DDeformationFieldTest(tf.test.TestCase):

  def test_applies_cropping_offset(self):
    deformation_field = deformation_utils.create_2d_deformation_field(
        raw_image_center_pos_pix=tf.constant([1.0, 1.0]),
        raw_image_element_size_um=tf.constant([1.0, 1.0]),
        net_input_spatial_shape=[3, 3],
        net_input_element_size_um=tf.constant([1.0, 1.0]),
        control_grid_spacings_pix=[2.0, 2.0],
        deformations_magnitudes_um=tf.constant([0.0, 0.0]),
        rotation_angle=tf.constant(0.0),
        scale_factors=tf.constant([1.0, 1.0]),
        mirror_factors=tf.constant([1, 1]),
        shearing_coefs=tf.constant([0.0, 0.0]),
        cropping_offset_pix=tf.constant([2.0, 3.0]))

    expected_output = np.array([[[2, 3], [2, 4], [2, 5]],
                                [[3, 3], [3, 4], [3, 5]],
                                [[4, 3], [4, 4], [4, 5]]])

    with self.session() as sess:
      np.testing.assert_allclose(
          expected_output,
          sess.run(deformation_field),
          atol=_ARRAY_COMPARE_TOLERANCE)

  def test_applies_rotation(self):
    deformation_field = deformation_utils.create_2d_deformation_field(
        raw_image_center_pos_pix=tf.constant([1.0, 1.0]),
        raw_image_element_size_um=tf.constant([1.0, 1.0]),
        net_input_spatial_shape=[3, 3],
        net_input_element_size_um=tf.constant([1.0, 1.0]),
        control_grid_spacings_pix=[2.0, 2.0],
        deformations_magnitudes_um=tf.constant([0.0, 0.0]),
        rotation_angle=tf.constant(np.pi / 4.),
        scale_factors=tf.constant([1.0, 1.0]),
        mirror_factors=tf.constant([1, 1]),
        shearing_coefs=tf.constant([0.0, 0.0]),
        cropping_offset_pix=tf.constant([0.0, 0.0]))

    expected_output = np.array([[[-0.4142135624, 1.],
                                 [0.2928932188, 1.7071067812],
                                 [1., 2.4142135624]],
                                [[0.2928932188, 0.2928932188],
                                 [1., 1.],
                                 [1.7071067812, 1.7071067812]],
                                [[1., -0.4142135624],
                                 [1.7071067812, 0.2928932188],
                                 [2.4142135624, 1]]])

    with self.session() as sess:
      np.testing.assert_allclose(
          expected_output,
          sess.run(deformation_field),
          atol=_ARRAY_COMPARE_TOLERANCE)

  def test_applies_shear(self):
    deformation_field = deformation_utils.create_2d_deformation_field(
        raw_image_center_pos_pix=tf.constant([1.0, 1.0]),
        raw_image_element_size_um=tf.constant([1.0, 1.0]),
        net_input_spatial_shape=[3, 3],
        net_input_element_size_um=tf.constant([1.0, 1.0]),
        control_grid_spacings_pix=[2.0, 2.0],
        deformations_magnitudes_um=tf.constant([0.0, 0.0]),
        rotation_angle=tf.constant(0.0),
        scale_factors=tf.constant([1.0, 1.0]),
        mirror_factors=tf.constant([1, 1]),
        shearing_coefs=tf.constant([0.0, 0.1]),
        cropping_offset_pix=tf.constant([0.0, 0.0]))

    expected_output = np.array([[[-0.1, 0], [0, 1], [0.1, 2]],
                                [[0.9, 0], [1, 1], [1.1, 2]],
                                [[1.9, 0], [2, 1], [2.1, 2]]])

    with self.session() as sess:
      np.testing.assert_allclose(
          expected_output,
          sess.run(deformation_field),
          atol=_ARRAY_COMPARE_TOLERANCE)

  def test_applies_mirror(self):
    deformation_field = deformation_utils.create_2d_deformation_field(
        raw_image_center_pos_pix=tf.constant([1.0, 1.0]),
        raw_image_element_size_um=tf.constant([1.0, 1.0]),
        net_input_spatial_shape=[3, 3],
        net_input_element_size_um=tf.constant([1.0, 1.0]),
        control_grid_spacings_pix=[2.0, 2.0],
        deformations_magnitudes_um=tf.constant([0.0, 0.0]),
        rotation_angle=tf.constant(0.0),
        scale_factors=tf.constant([1.0, 1.0]),
        mirror_factors=tf.constant([-1, 1]),
        shearing_coefs=tf.constant([0.0, 0.0]),
        cropping_offset_pix=tf.constant([0.0, 0.0]))

    expected_output = np.array([[[2., 0.], [2., 1.], [2., 2.]],
                                [[1., 0.], [1., 1.], [1., 2.]],
                                [[0., 0.], [0., 1.], [0., 2.]]])

    with self.session() as sess:
      np.testing.assert_allclose(
          expected_output,
          sess.run(deformation_field),
          atol=_ARRAY_COMPARE_TOLERANCE)

  def test_applies_scale(self):
    deformation_field = deformation_utils.create_2d_deformation_field(
        raw_image_center_pos_pix=tf.constant([1.0, 1.0]),
        raw_image_element_size_um=tf.constant([1.0, 1.0]),
        net_input_spatial_shape=[3, 3],
        net_input_element_size_um=tf.constant([1.0, 1.0]),
        control_grid_spacings_pix=[2.0, 2.0],
        deformations_magnitudes_um=tf.constant([0.0, 0.0]),
        rotation_angle=tf.constant(0.0),
        scale_factors=tf.constant([2.0, 1.0]),
        mirror_factors=tf.constant([1, 1]),
        shearing_coefs=tf.constant([0.0, 0.0]),
        cropping_offset_pix=tf.constant([0.0, 0.0]))

    expected_output = np.array([[[-1., 0.], [-1., 1.], [-1., 2.]],
                                [[1., 0.], [1., 1.], [1., 2.]],
                                [[3., 0.], [3., 1.], [3., 2.]]])

    with self.session() as sess:
      np.testing.assert_allclose(
          expected_output,
          sess.run(deformation_field),
          atol=_ARRAY_COMPARE_TOLERANCE)

  def test_applies_multiple_transforms_together(self):
    deformation_field = deformation_utils.create_2d_deformation_field(
        raw_image_center_pos_pix=tf.constant([1.0, 1.0]),
        raw_image_element_size_um=tf.constant([1.0, 1.0]),
        net_input_spatial_shape=[3, 3],
        net_input_element_size_um=tf.constant([1.0, 1.0]),
        control_grid_spacings_pix=[2.0, 2.0],
        deformations_magnitudes_um=tf.constant([0.0, 0.0]),
        rotation_angle=tf.constant(np.pi / 2.),
        scale_factors=tf.constant([1.0, 2.0]),
        mirror_factors=tf.constant([1, -1]),
        shearing_coefs=tf.constant([0.1, 0.0]),
        cropping_offset_pix=tf.constant([3.0, 5.0]))

    expected_output = np.array([[[3., 3.9], [4., 4.], [5., 4.1]],
                                [[3., 5.9], [4., 6.], [5., 6.1]],
                                [[3., 7.9], [4., 8.], [5., 8.1]]])

    with self.session() as sess:
      np.testing.assert_allclose(
          expected_output,
          sess.run(deformation_field),
          atol=_ARRAY_COMPARE_TOLERANCE)

  def test_oddEvenErrorHandling(self):
    with tf.Session():
      deform = deformation_utils.create_2d_deformation_field(
          np.array([101, 101]) / 2,
          raw_image_element_size_um=tf.constant([1., 1.]),
          net_input_spatial_shape=[50, 101],
          net_input_element_size_um=tf.constant([2., 1.]),
          control_grid_spacings_pix=[10, 10],
          deformations_magnitudes_um=tf.constant((0., 0.)),
          rotation_angle=tf.constant(0.),
          scale_factors=tf.constant((1., 1.)),
          mirror_factors=tf.constant((1., 1.)),
          shearing_coefs=tf.constant((0., 0., 0., 0.)),
          cropping_offset_pix=tf.constant((0., 0.)))

      with self.assertRaisesWithPredicateMatch(
          tf.errors.InvalidArgumentError,
          "factor must be odd as input and output size is even"):
        deform.eval()


if __name__ == "__main__":
  tf.test.main()
