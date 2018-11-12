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

"""Helper functions for deformation augmentation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from multidim_image_augmentation import augmentation_ops


def vectorized_random_uniform(minvals, maxvals, name=None):
  """creates a tensor with uniform random values.

  Args:
    minvals: 1-D Tensor with minimum values
    maxvals: 1-D Tensor with maximum values
    name: optional name for the operation

  Returns:
    1-D Tensor with uniform random values
  """

  with tf.variable_scope(name, "vectorized_random_uniform", [minvals, maxvals]):
    ranges = tf.subtract(maxvals, minvals, name="ranges")
    samples = tf.random_uniform(
        ranges.shape_as_list(), dtype=ranges.dtype, name="samples")
    samples_scaled = tf.multiply(ranges, samples, name="samples_scaled")
    samples_scaled_offset = tf.add(samples_scaled,
                                   minvals,
                                   name="samples_scaled_offset")
  return samples_scaled_offset


def create_centered_identity_transformation_field(shape, spacings):
  """Create centered identity transformation field."""
  coords = []
  for i in xrange(len(shape)):
    size = shape[i]
    spacing = spacings[i]
    coords.append(tf.linspace(
        -(size - 1) / 2 * spacing,
        (size - 1) / 2 * spacing,
        size))
  permutation = np.roll(np.arange(len(coords) + 1), -1)
  return tf.transpose(tf.meshgrid(*coords, indexing="ij"), permutation)


def create_control_grid_for_cubic_interp(transformed_image_shape,
                                         transformed_image_spacings_um,
                                         control_grid_spacings_pix):
  """Create a control grid with optimal size for cubic interpolation.

  The control grid will have two extra points in every direction to allow an
  interpolation without border artefacts

  Args:
    transformed_image_shape: 2- or 3-element list describing the shape of the
      target image
    transformed_image_spacings_um: 2- or 3-element tensor describing the spacing
      of the target image
    control_grid_spacings_pix: 2- or 3-element list describing the control grid
      spacings

  Returns:
    2D case: 3-D Tensor (x0,x1,comp) describing a 2D vector field
    3D case: 4-D Tensor (x0,x1,x2,comp)  describing a 3D vector field
  """

  grid_shape = np.zeros(len(transformed_image_shape))
  for comp in range(len(transformed_image_shape)):
    spacing_pix = float(control_grid_spacings_pix[comp])
    num_elem = float(transformed_image_shape[comp])
    if num_elem % 2 == 0:
      grid_shape[comp] = np.ceil((num_elem - 1) / (2 * spacing_pix) +
                                 0.5) * 2 + 2
    else:
      grid_shape[comp] = np.ceil((num_elem - 1) / (2 * spacing_pix)) * 2 + 3
  control_grid_spacings_um = tf.multiply(
      tf.constant(control_grid_spacings_pix, dtype=tf.float32),
      transformed_image_spacings_um)
  control_grid = create_centered_identity_transformation_field(
      grid_shape, control_grid_spacings_um)
  control_grid.set_shape(np.append(grid_shape, len(control_grid_spacings_pix)))
  return control_grid


def create_2x2_rotation_matrix(radians):
  rotation = [[tf.cos(radians), -tf.sin(radians)],
              [tf.sin(radians), tf.cos(radians)]]
  rotation = tf.convert_to_tensor(rotation, name="rotation_matrix")
  return rotation


def create_2x2_shearing_matrix(shearing_coefs):
  shearing = [[1, shearing_coefs[0]], [shearing_coefs[1], 1]]
  shearing = tf.convert_to_tensor(shearing, name="shearing_matrix")
  return shearing


def create_3x3_rotation_matrix(radians):
  """Creates a 3D rotation matrix.

  Args:
    radians: 1-D Tensor with 3 elements, (a0, a1, a2) with the 3 rotation
      angles, where a0 is the rotation around the x0 axis, etc.

  Returns:
    2-D Tensor with 3x3 elements, the rotation matrix
  """

  with tf.variable_scope("rotation_dim_0"):
    rotation_dim_0 = [[1.0, 0.0, 0.0],
                      [0.0, tf.cos(radians[0]), -tf.sin(radians[0])],
                      [0.0, tf.sin(radians[0]), tf.cos(radians[0])]]
    rotation_dim_0 = tf.convert_to_tensor(
        rotation_dim_0, name="rotation_matrix")
  with tf.variable_scope("rotation_dim_1"):
    rotation_dim_1 = [[tf.cos(radians[1]), 0.0, tf.sin(radians[1])],
                      [0.0, 1.0, 0.0],
                      [-tf.sin(radians[1]), 0.0, tf.cos(radians[1])]]
    rotation_dim_1 = tf.convert_to_tensor(
        rotation_dim_1, name="rotation_matrix")
  with tf.variable_scope("rotation_dim_2"):
    rotation_dim_2 = [[tf.cos(radians[2]), -tf.sin(radians[2]), 0.0],
                      [tf.sin(radians[2]), tf.cos(radians[2]), 0.0],
                      [0.0, 0.0, 1.0]]
    rotation_dim_2 = tf.convert_to_tensor(
        rotation_dim_2, name="rotation_matrix")
  with tf.variable_scope("rotation"):
    rotation = tf.matmul(rotation_dim_0, rotation_dim_1)
    rotation = tf.matmul(rotation, rotation_dim_2)
  return rotation


def create_3x3_shearing_matrix(shearing_coefs):
  shearing = [[1., shearing_coefs[0], shearing_coefs[1]],
              [shearing_coefs[2], 1., shearing_coefs[3]],
              [shearing_coefs[4], shearing_coefs[5], 1.]]
  shearing = tf.convert_to_tensor(shearing, name="shearing_matrix")
  return shearing


def create_3d_deformation_field(
    raw_image_center_pos_pix, raw_image_element_size_um,
    net_input_spatial_shape, net_input_element_size_um,
    control_grid_spacings_pix, deformations_magnitudes_um, rotation_angles,
    scale_factors, mirror_factors, shearing_coefs, cropping_offset_pix):
  """Create a 3D deformation field."""
  # Set up the centered control grid for identity transform in real world
  # coordinates.
  control_grid = create_control_grid_for_cubic_interp(
      net_input_spatial_shape, net_input_element_size_um,
      control_grid_spacings_pix)

  # Add random deformation.
  control_grid += deformations_magnitudes_um * tf.random_normal(
      shape=control_grid.shape_as_list())

  # Apply affine transformation and transform units to raw image pixels.
  scale_to_pix = 1. / raw_image_element_size_um
  affine = tf.matmul(
      create_3x3_rotation_matrix(rotation_angles),
      tf.diag(scale_factors * mirror_factors * scale_to_pix))
  affine_shearing = tf.matmul(
      affine, create_3x3_shearing_matrix(shearing_coefs))

  control_grid = tf.reshape(
      tf.matmul(tf.reshape(control_grid, [-1, 3]), affine_shearing),
      control_grid.shape_as_list())

  # Translate to cropping position.
  control_grid += raw_image_center_pos_pix + cropping_offset_pix

  # Create the dense deformation field for the image.
  dense_deformation_field = augmentation_ops.cubic_interpolation3d(
      control_grid, control_grid_spacings_pix, net_input_spatial_shape)

  return dense_deformation_field
