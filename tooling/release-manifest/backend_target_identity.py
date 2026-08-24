"""Canonical target identities shared by Windows backend release gates."""

from __future__ import annotations

import re


CUDA_TARGET = re.compile(r"sm_[0-9]{2,3}")
HIP_TARGET = re.compile(r"gfx[0-9]{3,4}[a-z]?")


def is_cuda_qualification_target(value: object) -> bool:
    """Return whether value is one canonical CUDA compilation target."""

    return isinstance(value, str) and CUDA_TARGET.fullmatch(value) is not None


def is_hip_qualification_target(value: object) -> bool:
    """Return whether value is one canonical HIP/ROCm compilation target."""

    return isinstance(value, str) and HIP_TARGET.fullmatch(value) is not None


def is_vulkan_qualification_target(value: object) -> bool:
    """Return whether value is one exact reusable Vulkan capability class.

    The class combines PCI vendor/device ids with Vulkan's pipeline-cache
    compatibility UUID. Exact driver version is bound separately by every
    hardware/correctness receipt. A per-card physical UUID is intentionally
    excluded so evidence can cover another card in the same implementation
    class without broadening across GPU models or driver implementations.
    """

    if not isinstance(value, str) or not value.startswith("vk_caps_"):
        return False
    parts = value.removeprefix("vk_caps_").split("_")
    return (
        len(parts) == 3
        and len(parts[0]) == 8
        and len(parts[1]) == 8
        and len(parts[2]) == 32
        and all(char in "0123456789abcdef" for part in parts for char in part)
    )


def is_provider_qualification_target(provider: object, value: object) -> bool:
    """Validate an exact live qualification target for one provider."""

    return (
        (provider == "cuda" and is_cuda_qualification_target(value))
        or (provider == "hip" and is_hip_qualification_target(value))
        or (provider == "vulkan" and is_vulkan_qualification_target(value))
    )
