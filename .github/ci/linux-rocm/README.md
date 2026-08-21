# Linux ROCm CI build environment

Wrapper around the official AMD ROCm 7.2.1 devel image plus hipBLAS. It is
not a product or release image.

- The base image is pinned by its manifest digest.
- Every source revision is published under its commit SHA.
- Consumers pin the verified GHCR digest in `.github/ci/linux-rocm.lock`,
  never a mutable tag.
- The publish workflow verifies C/C++, CMake, hipcc, hipBLAS, ALSA, and
  Python before a digest is eligible for consumers.
- The default user is a fixed non-root CI account.

To change the dependency contract, update the Dockerfile in the same pull
request. After the image workflow succeeds, copy its immutable digest into
`.github/ci/linux-rocm.lock` and `tooling/release-manifest/release_binaries_matrix.json`.
