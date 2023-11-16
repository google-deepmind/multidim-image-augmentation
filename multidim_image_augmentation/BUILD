# Description:
#   multidim_image_augmentation provides Tensorflow operations for 2D & 3D
#   image augmentation.

licenses(["notice"])

exports_files(["LICENSE"])

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

APPLY_DEFORMATION_HDRS_X86 = ["kernels/apply_deformation_avx.h"]

cc_library(
    name = "apply_deformation",
    hdrs = select({
        "//third_party/bazel_platforms/cpu:x86_64": APPLY_DEFORMATION_HDRS_X86,
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
        "@protobuf_archive//:protobuf",
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

## Python API
cc_library(
    name = "kernel_hdrs",
    hdrs = [
        "kernels/apply_deformation.h",
        "kernels/bspline.h",
        "kernels/cubic_interpolation.h",
        "platform/types.h",
    ],
)

cc_binary(
    name = 'python/ops/_augmentation_ops.so',
    srcs = [
        "kernels/apply_deformation_ops.cc",
        "kernels/apply_tabulated_functions_op.cc",
        "kernels/cubic_interpolation_ops.cc",
        "kernels/random_lut_controlpoints_op.cc",
        "ops/apply_deformation_ops.cc",
        "ops/apply_tabulated_functions_op.cc",
        "ops/cubic_interpolation1d_op.cc",
        "ops/cubic_interpolation2d_op.cc",
        "ops/cubic_interpolation3d_op.cc",
        "ops/random_lut_controlpoints_op.cc",
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        ":ops_hdrs",
        ":kernel_hdrs"
    ],
)

py_library(
    name = "augmentation_ops",
    srcs = ([
        "python/ops/augmentation_ops.py",
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    data = [
        ":python/ops/_augmentation_ops.so"
    ],
    visibility = ["//visibility:public"],
    srcs_version = "PY3ONLY",
)

py_test(
    name = "python/kernel_tests/apply_deformation_op_test",
    size = "medium",
    srcs = ["python/kernel_tests/apply_deformation_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":augmentation_ops",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
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
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
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
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
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
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
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
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
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
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
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
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
