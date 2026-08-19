# Linux ROCm CI build environment

Wrapper around the official AMD ROCm 7.2.1 devel image. It is not a product
or release image.

- The base image is pinned by its manifest digest.
- Every source revision is published under its commit SHA.
- Consumers must pin a verified GHCR digest, never a mutable tag. This pull
  request still lets release-binaries pin the official ROCm digest until a
  GHCR digest exists.
- The publish workflow verifies C/C++, CMake, hipcc, ALSA, and Python before a
  digest is eligible for consumers.
- The default user is a fixed non-root CI account.

To change the dependency contract, update the Dockerfile in the same pull
request. After the image workflow succeeds, copy its immutable digest into the
consumer workflow declarations and a future `.github/ci/linux-rocm.lock`.
