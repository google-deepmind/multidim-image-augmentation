FROM tensorflow/build:latest-python3.10

# Allow statements and log messages to immediately appear
ENV PYTHONUNBUFFERED True

# Set the working directory
WORKDIR /augmentation_src

LABEL maintainer="no-reply@google.com"

# Re-declare args because the args declared before FROM can't be used in any
# instruction after a FROM.
ARG python_version="python3.10"

# Update binutils to avoid linker(gold) issue. See b/227299577#comment9
RUN \
 wget http://old-releases.ubuntu.com/ubuntu/pool/main/b/binutils/binutils_2.35.1-1ubuntu1_amd64.deb \
 && wget http://old-releases.ubuntu.com/ubuntu/pool/main/b/binutils/binutils-x86-64-linux-gnu_2.35.1-1ubuntu1_amd64.deb \
 && wget http://old-releases.ubuntu.com/ubuntu/pool/main/b/binutils/binutils-common_2.35.1-1ubuntu1_amd64.deb \
 && wget http://old-releases.ubuntu.com/ubuntu/pool/main/b/binutils/libbinutils_2.35.1-1ubuntu1_amd64.deb

RUN \
  dpkg -i binutils_2.35.1-1ubuntu1_amd64.deb \
            binutils-x86-64-linux-gnu_2.35.1-1ubuntu1_amd64.deb \
            binutils-common_2.35.1-1ubuntu1_amd64.deb \
            libbinutils_2.35.1-1ubuntu1_amd64.deb

# Copy the requirements file used for dependencies, place it under /app
COPY build_requirements.txt .

RUN $python_version -mpip install -r build_requirements.txt

#Copy the rest of the source
COPY . .

RUN bazel build //:build_pip_pkg

RUN $python_version -m build -o dest bazel-bin/build_pip_pkg.runfiles/__main__/

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/tensorflow/
RUN auditwheel repair --plat manylinux2014_x86_64 --exclude libtensorflow_framework.so.2 -w dest/ dest/image_augmentation-*-cp310-cp310-linux_x86_64.whl

WORKDIR /

CMD ["/bin/bash"]
