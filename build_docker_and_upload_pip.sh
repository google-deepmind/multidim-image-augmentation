#!/bin/bash
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

for i in {10..12}; do
  ver=python3.$i
  echo Building for $ver

  docker build -f Dockerfile -t multidim-op-dev-$ver --build-arg PYTHON_VERSION=$ver . &&
  # Remove any existing containers
  docker rm multidim-op-dev
  # Print the functions within the new Op we just built
  docker run --name=multidim-op-dev multidim-op-dev-$ver /bin/bash -c "pip install /augmentation_src/dest/*manylinux* && python -c 'from multidim_image_augmentation import augmentation_ops; from inspect import getmembers, isfunction; print(getmembers(augmentation_ops, isfunction))'" &&
  # Copy the wheel and source tarball out of the container
  docker cp multidim-op-dev:/augmentation_src/dest/. dest/ &&
n  # Remove the container
  docker rm multidim-op-dev
done

echo Built files:
find dest*/

while true; do

read -p "Do you want to proceed with upload? (y/n) " yn

case $yn in
  [yY] ) echo Uploading...;
    break;;
  [nN] ) echo Exiting...;
    exit;;
  * ) echo Invalid Entry;;
esac

done

# Upload to PyPi
python3 -m twine upload dest/*manylinux* dest/*tar.gz
