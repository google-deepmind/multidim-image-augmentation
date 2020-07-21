# multidim_image_augmentation provides Tensorflow operations for 2D & 3D
# image augmentation.

licenses(["notice"])

exports_files(["LICENSE"])

load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_custom_op_library",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
)
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

cc_library(
    name = "types",
    hdrs = ["platform/types.h"],
)

cc_library(
    name = "bspline",
    hdrs = ["kernels/bspline.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@protobuf_archive//:protobuf",
    ],
)

cc_test(
    name = "bspline_test",
    size = "small",
    srcs = ["kernels/bspline_test.cc"],
    deps = [
        ":bspline",
        "@org_tensorflow//tensorflow/core:test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cubic_interpolation",
    hdrs = ["kernels/cubic_interpolation.h"],
    deps = [
        ":bspline",
        "@eigen_archive//:eigen",
    ],
)

cc_test(
    name = "cubic_interpolation_test",
    size = "small",
    srcs = ["kernels/cubic_interpolation_test.cc"],
    deps = [
        ":cubic_interpolation",
        "@org_tensorflow//tensorflow/core:test",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen",
    ],
)

cc_library(
    name = "apply_deformation",
    hdrs = select({
        "//conditions:default": [],
    }) + ["kernels/apply_deformation.h"],
    deps = [
        ":types",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@eigen_archive//:eigen",
    ],
)

cc_test(
    name = "apply_deformation_test",
    size = "small",
    srcs = ["kernels/apply_deformation_test.cc"],
    deps = [
        ":apply_deformation",
        "@org_tensorflow//tensorflow/core:test",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen",
    ],
)

cc_library(
    name = "ops_hdrs",
    hdrs = ["ops/apply_deformation_ops.h"],
)

cc_library(
    name = "augmentation_ops_interfaces",
    srcs = [
        "ops/apply_deformation_ops.cc",
        "ops/apply_tabulated_functions_op.cc",
        "ops/cubic_interpolation1d_op.cc",
        "ops/cubic_interpolation2d_op.cc",
        "ops/cubic_interpolation3d_op.cc",
        "ops/random_lut_controlpoints_op.cc",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":ops_hdrs",
        ":types",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@eigen_archive//:eigen",
        "@protobuf_archive//:protobuf",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "apply_tabulated_functions_op_test",
    size = "small",
    srcs = ["kernels/apply_tabulated_functions_op_test.cc"],
    deps = [
        ":augmentation_ops_interfaces_and_kernels",
        "@org_tensorflow//tensorflow/cc:const_op",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
        "@org_tensorflow//tensorflow/core/kernels:ops_testutil",
        "@org_tensorflow//tensorflow/core/kernels:ops_util",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_kernel_library(
    name = "augmentation_ops_kernels",
    srcs = [
        "kernels/apply_deformation_ops.cc",
        "kernels/apply_tabulated_functions_op.cc",
        "kernels/cubic_interpolation_ops.cc",
        "kernels/random_lut_controlpoints_op.cc",
    ],
    hdrs = ["ops/apply_deformation_ops.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":apply_deformation",
        ":cubic_interpolation",
        ":ops_hdrs",
        ":types",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "augmentation_ops_interfaces_and_kernels",
    visibility = ["//visibility:public"],
    deps = [
        ":augmentation_ops_interfaces",
        ":augmentation_ops_kernels",
    ],
)

tf_custom_op_library(
    name = "python/ops/_augmentation_ops.so",
    srcs = [
        "ops/apply_deformation_ops.cc",
        "ops/apply_tabulated_functions_op.cc",
        "ops/cubic_interpolation1d_op.cc",
        "ops/cubic_interpolation2d_op.cc",
        "ops/cubic_interpolation3d_op.cc",
        "ops/random_lut_controlpoints_op.cc",
    ],
    deps = [
        ":augmentation_ops_kernels",
        ":types",
    ],
)

## Python API

tf_gen_op_wrapper_py(
    name = "gen_augmentation_ops_py",
    # 'out' has no gen_ prefix as we fully generate the python code.
    out = "augmentation_ops.py",
    deps = [
        ":augmentation_ops_interfaces",
    ],
)

tf_custom_op_py_library(
    name = "augmentation_ops",
    srcs = [
        "__init__.py",
        "augmentation_ops.py",
    ],
    dso = [":python/ops/_augmentation_ops.so"],
    kernels = [
        ":augmentation_ops_kernels",
        ":augmentation_ops_interfaces",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":gen_augmentation_ops_py",
        "@org_tensorflow//tensorflow/python:framework_for_generated_wrappers",
        "@org_tensorflow//tensorflow/python:platform",
    ],
)

py_test(
    name = "python/kernel_tests/apply_deformation_op_test",
    size = "medium",
    srcs = ["python/kernel_tests/apply_deformation_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":augmentation_ops",
        "@six_archive//:six",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//third_party/py/numpy",
    ],
)

py_test(
    name = "python/kernel_tests/cubic_interpolation1d_op_test",
    size = "medium",
    srcs = ["python/kernel_tests/cubic_interpolation1d_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":augmentation_ops",
        "@six_archive//:six",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//third_party/py/numpy",
    ],
)

py_test(
    name = "python/kernel_tests/cubic_interpolation2d_op_test",
    size = "medium",
    srcs = ["python/kernel_tests/cubic_interpolation2d_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":augmentation_ops",
        "@six_archive//:six",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//third_party/py/numpy",
    ],
)

py_test(
    name = "python/kernel_tests/cubic_interpolation3d_op_test",
    size = "medium",
    srcs = ["python/kernel_tests/cubic_interpolation3d_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":augmentation_ops",
        "@six_archive//:six",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//third_party/py/numpy",
    ],
)

py_test(
    name = "python/kernel_tests/random_lut_controlpoints_op_test",
    srcs = ["python/kernel_tests/random_lut_controlpoints_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":augmentation_ops",
        "@six_archive//:six",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "python/kernel_tests/apply_tabulated_functions_op_test",
    srcs = ["python/kernel_tests/apply_tabulated_functions_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":augmentation_ops",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//third_party/py/numpy",
    ],
)

py_library(
    name = "deformation_utils",
    srcs = [
        "deformation_utils.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":augmentation_ops",
        "@six_archive//:six",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//third_party/py/numpy",
    ],
)

py_test(
    name = "deformation_utils_test",
    size = "small",
    srcs = ["deformation_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":deformation_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
