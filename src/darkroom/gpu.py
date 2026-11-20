"""The GPU's memory and how busy it is, read through NVML (the driver's own library): one call
into the driver, no process started, nothing taken from a render. Every GPU NVML lists is read,
and the one with the least memory free is told of, so what is packed beside it fits whichever
the renders land on. Plain Python."""

from __future__ import annotations

import ctypes
import sys
from typing import Any


class _Memory(ctypes.Structure):
    """NVML's `nvmlMemory_v2_t`: its "used" leaves out what the driver reserves, as
    nvidia-smi's does."""

    _fields_ = [("version", ctypes.c_uint), ("total", ctypes.c_ulonglong), ("reserved", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]  # fmt: skip


class _Busy(ctypes.Structure):
    """NVML's `nvmlUtilization_t`: the share of the last sample period the GPU (and its
    memory) was busy, percent."""

    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


#: NVML (the driver's own library) and every GPU's handle, opened once; None when not there.
_nvml: tuple[Any, list[ctypes.c_void_p]] | bool | None = False


def _opened() -> tuple[Any, list[ctypes.c_void_p]] | None:
    """NVML and every GPU's handle, opened when first asked; None when it is not there."""
    global _nvml
    if _nvml is False:
        _nvml = None
        try:
            lib = ctypes.CDLL("nvml.dll" if sys.platform == "win32" else "libnvidia-ml.so.1")
            count = ctypes.c_uint()
            if lib.nvmlInit_v2() == 0 and lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) == 0:
                handles = []
                for index in range(count.value):
                    handle = ctypes.c_void_p()
                    if lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) == 0:
                        handles.append(handle)
                _nvml = (lib, handles) if handles else None
        except OSError:
            pass
    return _nvml if isinstance(_nvml, tuple) else None


def _tightest() -> tuple[Any, ctypes.c_void_p, _Memory] | None:
    """NVML, the handle and the memory of the GPU with the least memory free."""
    opened = _opened()
    if opened is None:
        return None
    lib, handles = opened
    found = None
    for handle in handles:
        info = _Memory(version=ctypes.sizeof(_Memory) | 2 << 24)
        if lib.nvmlDeviceGetMemoryInfo_v2(handle, ctypes.byref(info)) != 0:
            continue
        if found is None or info.free < found[2].free:
            found = (lib, handle, info)
    return found


def memory() -> tuple[float, float] | None:
    """The GPU's memory in use and in all, GB (of the GPU with the least free); None unread."""
    found = _tightest()
    if found is None:
        return None
    info = found[2]
    return info.used / 2**30, info.total / 2**30


def busy() -> float | None:
    """How busy the GPU (with the least memory free) was over NVML's last sample period,
    percent (as nvtop and nvidia-smi say); None unread."""
    found = _tightest()
    if found is None:
        return None
    lib, handle, _ = found
    rates = _Busy()
    if lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(rates)) != 0:
        return None
    return float(rates.gpu)
