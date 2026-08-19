# Linux CUDA CI build environment

Wrapper around the official NVIDIA CUDA 13.2.0 devel image. It is not a
product or release image.

- The base image is pinned by its linux/amd64 manifest digest.
- Every source revision is published under its commit SHA.
- Consumers must pin a verified GHCR digest, never a mutable tag. This pull
  request still lets release-binaries pin the official NVIDIA digest until a
  GHCR digest exists.
- The publish workflow verifies C/C++, CMake, nvcc, ALSA, and Python before a
  digest is eligible for consumers.
- The default user is a fixed non-root CI account.

To change the dependency contract, update the Dockerfile in the same pull
request. After the image workflow succeeds, copy its immutable digest into the
consumer workflow declarations and a future `.github/ci/linux-cuda.lock`.
