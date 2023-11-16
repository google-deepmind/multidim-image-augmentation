# multidim_image_augmentation opensource project root BUILD file.

sh_binary(
    name = "build_pip_pkg",
    srcs = ["build_pip_pkg.sh"],
    data = [
        "LICENSE",
        "MANIFEST.in",
        "setup.py",
        "//multidim_image_augmentation:augmentation_ops",
        "//multidim_image_augmentation:deformation_utils",
        "//multidim_image_augmentation:python/ops/_augmentation_ops.so",
    ],
)
