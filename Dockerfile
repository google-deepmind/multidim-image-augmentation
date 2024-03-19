ARG PYTHON_VERSION="python3.10"

FROM tensorflow/build:2.16-$PYTHON_VERSION

# To use the default value of an ARG declared before the first FROM use
# an ARG instruction without a value inside.
# https://docs.docker.com/reference/dockerfile/#understand-how-arg-and-from-interact
ARG PYTHON_VERSION

# Allow statements and log messages to immediately appear
ENV PYTHONUNBUFFERED True

# Set the working directory
WORKDIR /augmentation_src

LABEL maintainer="no-reply@google.com"

# Copy the requirements file used for dependencies, place it under /app
COPY build_requirements.txt .

RUN $PYTHON_VERSION -mpip install -r build_requirements.txt

#Copy the rest of the source
COPY .  .

ENV TF_HEADER_DIR=/usr/local/lib/$PYTHON_VERSION/dist-packages/tensorflow/include
ENV TF_SHARED_LIBRARY_DIR=/usr/local/lib/$PYTHON_VERSION/dist-packages/tensorflow
RUN bazel build //:build_pip_pkg

RUN $PYTHON_VERSION -m build -o dest bazel-bin/build_pip_pkg.runfiles/_main/

ENV LD_LIBRARY_PATH=/usr/local/lib/$PYTHON_VERSION/dist-packages/tensorflow/
RUN auditwheel repair --plat manylinux_2_24_x86_64 --exclude libtensorflow_framework.so.2 -w dest/ dest/image_augmentation-*-cp3*-cp3*-linux_x86_64.whl

WORKDIR /

CMD ["/bin/bash"]
