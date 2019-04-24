workspace(name = "multidim_image_augmentation")

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "a38539c5b5c358548e75b44141b4ab637bba7c4dc02b46b1f62a96d6433f56ae",
    strip_prefix = "rules_closure-dbb96841cc0a5fb2664c37822803b06dab20c7d1",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",  # 2018-04-13
    ],
)

# Tensorflow. If your project already builds Tensorflow source you should
# replace this with an appropriate local_archice() call.
# To update this to a newer TF version:
# 1/ curl -L https://github.com/tensorflow/tensorflow/archive/vX.Y.Z.tar.gz | sha256sum
# 2/ Update `sha256`, `strip_prefix` and `urls` attributes appropriately.
# 3/ Update io_bazel_rules_closure above to match the version used by TF.
http_archive(
    name = "org_tensorflow",
    sha256 = "7cd19978e6bc7edc2c847bce19f95515a742b34ea5e28e4389dade35348f58ed",
    strip_prefix = "tensorflow-1.13.1",
    urls = [
        "https://mirror.bazel.build/github.com/tensorflow/tensorflow/archive/v1.13.1.tar.gz",
        "https://github.com/tensorflow/tensorflow/archive/v1.13.1.tar.gz",
    ],
)

# Please add all new multidim_image_augmentation dependencies in workspace.bzl.
load("//:workspace.bzl", "multidim_image_augmentation_workspace")

multidim_image_augmentation_workspace()

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("0.15.0")
