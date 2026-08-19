# Linux CI build environment

This image removes networked `apt-get` work from routine Linux CI jobs. It is
not a product or release image.

- The base image is pinned by digest.
- Every source revision is published under its commit SHA.
- Consumer workflows pin the verified GHCR digest, never a mutable tag.
- The publish workflow verifies C/C++, CMake, ALSA, and Python/NumPy before a
  digest is eligible for consumers.
- The default user is a fixed non-root CI account so Unix permission tests keep
  the same semantics as GitHub-hosted runner jobs.
- Each consumer runs the image-owned `openasr-ci-verify` contract and marks the
  mounted checkout as a Git safe directory before invoking repository scripts.

To change the dependency contract, update the Dockerfile in the same pull
request. After the image workflow succeeds, copy its immutable digest into the
consumer workflow declarations and `.github/ci/linux-build-env.lock`, then let
their ordinary tests validate the new environment. The lint job rejects partial
digest updates and any reintroduced `apt-get` in routine consumers.
