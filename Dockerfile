FROM tensorflow/build:2.16-python3.10

# Allow statements and log messages to immediately appear
ENV PYTHONUNBUFFERED True

# Set the working directory
WORKDIR /augmentation_src

LABEL maintainer="no-reply@google.com"

# Re-declare args because the args declared before FROM can't be used in any
# instruction after a FROM.
ARG python_version="python3.10"

# Copy the requirements file used for dependencies, place it under /app
COPY build_requirements.txt .

RUN $python_version -mpip install -r build_requirements.txt

#Copy the rest of the source
COPY . .

RUN bazel build //:build_pip_pkg

RUN $python_version -m build -o dest bazel-bin/build_pip_pkg.runfiles/_main/

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/tensorflow/
RUN auditwheel repair --plat manylinux_2_24_x86_64 --exclude libtensorflow_framework.so.2 -w dest/ dest/image_augmentation-*-cp310-cp310-linux_x86_64.whl

WORKDIR /

CMD ["/bin/bash"]
