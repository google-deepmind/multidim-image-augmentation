# Copyright 2023 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Setup for pip package."""

import pathlib

import setuptools
from setuptools import dist
from setuptools.command import install


__version__ = '0.0.4'
REQUIRED_PACKAGES = [
    'tensorflow >= 2.16.1',
    'six >= 1.10.0',
    'numpy >= 1.20.0',
    'absl-py == 2.1.0',
]
project_name = 'image_augmentation'


class InstallPlatlib(install.install):
  """This class is needed in order to install platform specific libraries.

  Attributes:
    install_lib: library to install
  """

  def finalize_options(self):
    install.install.finalize_options(self)
    self.install_lib = self.install_platlib


class BinaryDistribution(dist.Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return True

  def is_pure(self):
    return False

# read the contents of your README file
this_directory = pathlib.Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setuptools.setup(
    name=project_name,
    version=__version__,
    description=(
        'multidim_image_augmentation provides Tensorflow operations for 2D & 3D'
        ' image augmentation.'
    ),
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Google Inc.',
    author_email='opensource@google.com',
    url='https://github.com/google-deepmind/multidim-image-augmentation/',
    # Contained modules and scripts.
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES,
    # Add in any packaged data.
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={'install': InstallPlatlib},
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache 2.0',
    keywords='tensorflow custom op machine learning image augmentation 3d',
)
