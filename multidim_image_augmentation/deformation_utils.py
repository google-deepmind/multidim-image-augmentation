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

"""Helper functions for deformation augmentation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf
from multidim_image_augmentation import augmentation_ops


def vectorized_random_uniform(minvals, maxvals, name=None):
  """creates a tensor with uniform random values.

  Args:
    minvals: 1-D Tensor with minimum values.
    maxvals: 1-D Tensor with maximum values.
    name: (optional) Name for the operation.

  Returns:
    1-D Tensor with uniform random values.
  """

  with tf.variable_scope(name, "vectorized_random_uniform", [minvals, maxvals]):
    ranges = tf.subtract(maxvals, minvals, name="ranges")
    samples = tf.random.uniform(
        ranges.shape, dtype=ranges.dtype, name="samples")
    samples_scaled = tf.multiply(ranges, samples, name="samples_scaled")
    samples_scaled_offset = tf.add(samples_scaled,
                                   minvals,
                                   name="samples_scaled_offset")
  return samples_scaled_offset


def create_centered_identity_transformation_field(shape, spacings):
  """Create 2D or 3D centered identity transformation field.

  Args:
    shape: 2- or 3-element list. The shape of the transformation field.
    spacings: 2- or 3-element list. The spacings of the transformation field.

  Returns:
    2D case: 3-D Tensor (x0, x1, comp) describing a 2D vector field
    3D case: 4-D Tensor (x0, x1, x2, comp)  describing a 3D vector field
  """
  coords = []
  for i, size in enumerate(shape):
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
  interpolation without border artefacts.

  Args:
    transformed_image_shape: 2- or 3-element list describing the shape of the
      target image.
    transformed_image_spacings_um: 2- or 3-element tensor describing the spacing
      of the target image.
    control_grid_spacings_pix: 2- or 3-element list describing the control grid
      spacings.

  Returns:
    2D case: 3-D Tensor (x0, x1, comp) describing a 2D vector field.
    3D case: 4-D Tensor (x0, x1, x2, comp)  describing a 3D vector field.
  """

  grid_shape = np.zeros(len(transformed_image_shape), dtype=int)
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
  """Creates a 2D rotation matrix.

  For an angle a this is
  [[cos(a), -sin(a)],
   [sin(a),  cos(a)]]

  Args:
    radians: rotation angle in radians.

  Returns:
    2-D Tensor with 2x2 elements, the rotation matrix.
  """
  rotation = [[tf.cos(radians), -tf.sin(radians)],
              [tf.sin(radians), tf.cos(radians)]]
  rotation = tf.convert_to_tensor(rotation, name="rotation_matrix")
  return rotation


def create_2x2_shearing_matrix(shearing_coefs):
  """Creates a 2D shearing matrix.

  Args:
    shearing_coefs: 2-element list with the shearing coefficients
      (off-diagonal elements of the matrix: s01, s10) to create the matrix
      [[ 1 , s01],
       [s10,  1 ]]

  Returns:
    2-D Tensor with 2x2 elements, the shearing matrix
  """
  shearing = [[1, shearing_coefs[0]], [shearing_coefs[1], 1]]
  shearing = tf.convert_to_tensor(shearing, name="shearing_matrix")
  return shearing


def create_2d_deformation_field(
    raw_image_center_pos_pix, raw_image_element_size_um,
    net_input_spatial_shape, net_input_element_size_um,
    control_grid_spacings_pix, deformations_magnitudes_um, rotation_angle,
    scale_factors, mirror_factors, shearing_coefs, cropping_offset_pix):
  """Creates a 2D deformation field.

  Creates a dense 2D deformation field for affine and elastic deformations. The
  created 2D vector field (represented as a 3-D Tensor with (x0, x1, comp))
  has the same spatial shape as the output (net_input) image and contains the
  absolute positions of the corresponding pixels in the input (raw) image. The
  process of creating the deformation field has four steps:
    1. Setup a grid of control points.
    2. Add a random offset to each control point drawn from a normal
       distribution to model the random elastic deformation.
    3. Apply the affine transformation to the control points.
    4. Compute a dense transformation field using cubic bspline interpolation.
  A more detailed description of the process can be found in the doc directory.

  Args:
    raw_image_center_pos_pix: 1-D Tensor with 2 elements of type tf.float32. The
      position of the center of the raw image in pixels from the upper, left
      corner.
    raw_image_element_size_um: 1-D Tensor with 2 elements of type tf.float32.
      The pixel spacing (in micrometers) of the raw image.
    net_input_spatial_shape: List with 2 elements. The shape of the image that
      will be fed into the network (excluding channel dimension).
    net_input_element_size_um: Tensor with 2 elements. The pixel spacing (in
      micrometers) of the image that will be fed into the network.
    control_grid_spacings_pix: List with 2 elements. The control grid spacing in
      pixels.
    deformations_magnitudes_um: 1-D Tensor with 2 elements. The magnitudes for
      the random deformations. Will set the standard deviation (in micrometers)
      of a random normal distribution from which deformations will be generated.
    rotation_angle: Rotation angle in radians as a float (or single element
      Tensor of floating point type). In the absence of mirroring, a positive
      angle produces a counter-clockwise rotation of image contents.
    scale_factors: 1-D Tensor with 2 elements of type tf.float32. Scale factors
      in x0, x1 directions.
    mirror_factors: 1-D Tensor with 2 elements. Mirror factors in x0, x1
      directions. Each factor should be 1 or -1.
    shearing_coefs: 1-D Tensor with 2 elements of type tf.float32. The shearing
      coefficients (s01, s10) to create the shearing matrix:
      [[ 1 , s01], [s10,  1]].
    cropping_offset_pix: 1-D Tensor with 2 elements of type tf.float32. Cropping
      position (center of the cropped patch in the raw image) in pixels relative
      to the image origin (the origin is specified above as
      raw_image_center_pos_pix).

  Returns:
    3-D Tensor (x0, x1, comp) containing a 2D vector field.
  """
  # Set up the centered control grid for identity transform in real world
  # coordinates.
  control_grid = create_control_grid_for_cubic_interp(
      transformed_image_shape=net_input_spatial_shape,
      transformed_image_spacings_um=net_input_element_size_um,
      control_grid_spacings_pix=control_grid_spacings_pix)

  # Add random deformation.
  control_grid += deformations_magnitudes_um * tf.random.normal(
      shape=control_grid.shape)

  # Apply affine transformation and transform units to raw image pixels.
  scale_to_pix = 1. / raw_image_element_size_um
  affine = tf.matmul(
      create_2x2_rotation_matrix(rotation_angle),
      tf.diag(scale_factors * tf.to_float(mirror_factors) * scale_to_pix))
  affine_shearing = tf.matmul(affine,
                              create_2x2_shearing_matrix(shearing_coefs))

  control_grid = tf.reshape(
      tf.matmul(tf.reshape(control_grid, [-1, 2]), affine_shearing),
      control_grid.get_shape().as_list())

  # Translate to cropping position.
  control_grid += raw_image_center_pos_pix + cropping_offset_pix

  # Create the dense deformation field for the image.
  dense_deformation_field = augmentation_ops.cubic_interpolation2d(
      control_grid, control_grid_spacings_pix, net_input_spatial_shape)

  return dense_deformation_field


def create_3x3_rotation_matrix(radians):
  """Creates a 3D rotation matrix.

  Args:
    radians: 1-D Tensor with 3 elements, (a0, a1, a2) with the 3 rotation
      angles in radians, where a0 is the rotation around the x0 axis, etc.

  Returns:
    2-D Tensor with 3x3 elements, the rotation matrix.
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
  """Creates a 3D shearing matrix.

  Args:
    shearing_coefs: 6-element list with the shearing coefficients
      (off-diagonal elements of the matrix: s01, s02, s10, s12, s20, s21) to
      create the matrix
      [[ 1 , s01, s02],
       [s10,  1 , s12],
       [s20, s21,  1 ]]

  Returns:
    2-D Tensor with 3x3 elements, the shearing matrix.
  """
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
  """Create a 3D deformation field.

  Creates a dense 3D deformation field for affine and elastic deformations. The
  created 3D vector field (represented as a 4-D Tensor with (x0, x1, x2, comp))
  has the same spatial shape as the output image and contains the absolute
  position of the corresponding voxel in the input (raw) image. The process of
  creating the deformation field has four steps:
    1. Setup a grid of control points
    2. Add a random offset to each control point drawn from a normal
       distribution to model the random elastic deformation
    3. Apply the affine transformation to the control points
    4. Compute a dense transformation field using cubic bspline interpolation
  A more detailled description of the process can be found in the doc
  directory.

  Args:
    raw_image_center_pos_pix: 1-D Tensor with 3 elements. The position of the
      origin in the raw image in pixels from the upper, left, front corner.
    raw_image_element_size_um: 1-D Tensor with 3 elements. The pixel spacing
      (in micrometers) of the raw image.
    net_input_spatial_shape: 1-D Tensor with 3 elements. The shape of the
      image that will be fed into the network.
    net_input_element_size_um: 1-D Tensor with 3 elements. The pixel spacing
      (in micrometers) of the image that will be fed into the network.
    control_grid_spacings_pix: 1-D Tensor with 3 elements. The control grid
      spacing in pixels.
    deformations_magnitudes_um: 1-D Tensor with 3 elements. The magnitudes
      for the random deformations, the standard deviation (in micrometers) of a
      random normal distribution.
    rotation_angles: 1-D Tensor with 3 elements, (a0, a1, a2) with the 3
      rotation angles in radians, where a0 is the rotation around the x0 axis,
      etc.
    scale_factors: 1-D Tensor with 3 elements. Scale factors in x0, x1, and x2
      directions.
    mirror_factors: 1-D Tensor with 3 elements. Mirror factors in x0, x1, and
      x2 direction. Each factor should be 1 or -1.
    shearing_coefs: 1-D Tensor with 6 elements. The shearing coefficients
      (off-diagonal elements of the matrix: s01, s02, s10, s12, s20, s21) to
      create the shearing matrix
      [[ 1 , s01, s02],
       [s10,  1 , s12],
       [s20, s21,  1 ]]
    cropping_offset_pix: 1-D Tensor with 3 elements. Cropping position (center
      of the cropped patch in the raw image) in pixels relative to the image
      origin (the origin is specified above as raw_image_center_pos_pix).

  Returns:
    4-D Tensor (x0, x1, x2, comp) describing a 3D vector field.
  """
  # Set up the centered control grid for identity transform in real world
  # coordinates.
  control_grid = create_control_grid_for_cubic_interp(
      net_input_spatial_shape, net_input_element_size_um,
      control_grid_spacings_pix)

  # Add random deformation.
  control_grid += deformations_magnitudes_um * tf.random.normal(
      shape=control_grid.shape)

  # Apply affine transformation and transform units to raw image pixels.
  scale_to_pix = 1. / raw_image_element_size_um
  affine = tf.matmul(
      create_3x3_rotation_matrix(rotation_angles),
      tf.diag(scale_factors * mirror_factors * scale_to_pix))
  affine_shearing = tf.matmul(
      affine, create_3x3_shearing_matrix(shearing_coefs))

  control_grid = tf.reshape(
      tf.matmul(tf.reshape(control_grid, [-1, 3]), affine_shearing),
      control_grid.shape)

  # Translate to cropping position.
  control_grid += raw_image_center_pos_pix + cropping_offset_pix

  # Create the dense deformation field for the image.
  dense_deformation_field = augmentation_ops.cubic_interpolation3d(
      control_grid, control_grid_spacings_pix, net_input_spatial_shape)

  return dense_deformation_field
