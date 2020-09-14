# Multidimensional (2D and 3D) Image Augmentation for TensorFlow

This package provides TensorFlow Ops for multidimensional volumetric image
augmentation.

## Install prerequities

This project usings the [bazel build
system](https://docs.bazel.build/versions/master/install.html)

## Build and test

To fetch the code, build it, and run tests:

```shell
git clone https://github.com/deepmind/multidim-image-augmentation.git
cd multidim-image-augmentation/
bazel test --python_version=py3 --config=nativeopt //...
```

Note bazel 0.24.0 made a lot of backward incompatible changes to default flag
values, that have not yet been resolved in this project and its dependencies.
In the meantime, you can disable with a few simple extra flags:

```shell
# Bazel >= 0.24.0
bazel test \
    --incompatible_disable_genrule_cc_toolchain_dependency=false \
    --incompatible_disable_legacy_cc_provider=false \
    --incompatible_disable_third_party_license_checking=false \
    --incompatible_no_transitive_loads=false \
    --incompatible_bzl_disallow_load_after_statement=false \
    --incompatible_disallow_load_labels_to_cross_package_boundaries=false \
    --config=nativeopt //...
```
## Build and test with TensorFlow 2.3

Compile the library with TensorFlow 2.3
```
git clone -b r2.3 https://github.com/tensorflow/tensorflow
cd tensorflow/tensorflow
git clone https://github.com/deepmind/multidim-image-augmentation deepmind_mia
cd deepmind_mia
find . -name "BUILD" | xargs sed -i 's/@org_tensorflow//g'
find . -name "BUILD" | xargs sed -i 's/protobuf_archive/com_google_protobuf/g'
find . -name "*.h" -o -name "*.cc" | xargs sed -i 's/multidim\_image\_augmentation\//tensorflow\/deepmind\_mia\/multidim\_image\_augmentation\//g'
sed -i 's/uint64\_t/unsigned\ long\ long/g' multidim_image_augmentation/platform/types.h
sed -i 's/int64\_t/long\ long/g' multidim_image_augmentation/platform/types.h
cd ../../
./configure (all select `YES` for CPU-version)
bazel build --config=opt --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 //tensorflow/deepmind_mia/multidim_image_augmentation:python/ops/_augmentation_ops.so
bazel build --config=opt --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 //tensorflow/deepmind_mia/multidim_image_augmentation:gen_augmentation_ops_py
cp bazel-bin/tensorflow/deepmind_mia/multidim_image_augmentation/augmentation_ops.py tensorflow/deepmind_mia/multidim_image_augmentation/
cp -r bazel-bin/tensorflow/deepmind_mia/multidim_image_augmentation/python/ops/ tensorflow/deepmind_mia/multidim_image_augmentation/python/
```
   
Do tests, first add `tf.disable_eager_execution()` to `deformation_utils.py` and `deformation_utils_test.py` and then run
```
python multidim_image_augmentation/deformation_utils_test.py
```
the partial outputs should be like this:
```
Ran 12 tests in 3.022s

OK (skipped=2)
```

To learn more about image augmentation, see the [primer](doc/index.md)

For simple API usage examples, see the python test code.

