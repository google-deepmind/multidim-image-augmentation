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

To learn more about image augmentation, see the [primer](doc/index.md)

For simple API usage examples, see the python test code.
