"""Project external dependencies that can be loaded in WORKSPACE files."""

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

def multidim_image_augmentation_workspace():
    """All multidim_image_augmentation external dependencies."""

    tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")

    # ===== gRPC dependencies =====
    native.bind(
        name = "libssl",
        actual = "@boringssl//:ssl",
    )

    native.bind(
        name = "zlib",
        actual = "@zlib_archive//:zlib",
    )

    # gRPC wants the existence of a cares dependence but its contents are not
    # actually important since we have set GRPC_ARES=0 in tools/bazel.rc
    native.bind(
        name = "cares",
        actual = "@grpc//third_party/nanopb:nanopb",
    )
