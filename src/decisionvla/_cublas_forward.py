"""Experimental exact-shape cuBLAS forward dispatch; not enabled in production.

Reuse Torch's handle once per expert group instead of preparing it per expert.
GEMM dimensions, BF16 storage, FP32 compute, reduction setting and algorithm match
the pinned Torch 2.9.1 cuBLAS fallback. Numerical equivalence still needs a gate.
"""
import ctypes
import functools
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile


@functools.lru_cache(maxsize=1)
def _dispatcher():
    import fcntl

    source = Path(__file__).with_suffix(".c")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home()/".cache"))/"decisionvla"/digest
    cache.mkdir(parents=True, exist_ok=True)
    target = cache/"cublas_forward.so"
    with (cache/"build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                built = Path(temporary)/"kernel.so"
                subprocess.run([os.environ.get("CC", "cc"), "-O3", "-std=c11", "-shared", "-fPIC",
                                str(source), "-o", str(built)], check=True, capture_output=True)
                built.replace(target)
    native = ctypes.CDLL(str(target))
    # Torch already loaded the pinned CUDA 13 library with this SONAME.
    cublas = ctypes.CDLL("libcublas.so.13")
    p, i = ctypes.c_void_p, ctypes.c_int
    native.decisionvla_cublas_forward.argtypes = [p]*8+[i]*5
    native.decisionvla_cublas_forward.restype = i
    cublas.cublasGetPointerMode_v2.argtypes = [p, ctypes.POINTER(i)]
    return native, cublas


def linear_forward(x, weight, offsets):
    import torch

    if (x.ndim != 2 or weight.ndim != 3 or offsets.ndim != 1
            or x.dtype != torch.bfloat16 or weight.dtype != x.dtype
            or offsets.dtype != torch.int32 or x.device.type != "cuda"
            or weight.device != x.device or offsets.device != x.device
            or not all(t.is_contiguous() for t in (x, weight, offsets))
            or weight.shape[2] != x.shape[1] or len(offsets) != weight.shape[0]
            or min(weight.shape) <= 0 or max(*x.shape, *weight.shape) >= 2**31):
        raise ValueError("cuBLAS diagnostic expects contiguous BF16 [T,K]/[E,N,K] and int32 offsets")
    if torch.version.cuda != "13.0" or torch.cuda.get_device_capability(x.device) != (12, 0):
        raise RuntimeError("This experimental dispatcher is pinned to CUDA13/SM120")
    native, cublas = _dispatcher()
    with torch.cuda.device(x.device):
        handle = torch.cuda.current_blas_handle()
        mode = ctypes.c_int()
        if cublas.cublasGetPointerMode_v2(handle, ctypes.byref(mode)) or mode.value != 0:
            raise RuntimeError("Expected host pointer mode on the Torch cuBLAS handle")
        cpu_offsets = offsets.cpu()
        output = torch.empty((len(x), weight.shape[1]), device=x.device, dtype=x.dtype)
        # DISALLOW_REDUCED_PRECISION_REDUCTION=16; do not change the process setting.
        math_mode = 0 if torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction else 16
        result = native.decisionvla_cublas_forward(
            ctypes.cast(cublas.cublasGemmEx, ctypes.c_void_p),
            ctypes.cast(cublas.cublasGetMathMode, ctypes.c_void_p),
            ctypes.cast(cublas.cublasSetMathMode, ctypes.c_void_p), handle,
            x.data_ptr(), weight.data_ptr(), output.data_ptr(), cpu_offsets.data_ptr(),
            weight.shape[0], len(x), weight.shape[1], weight.shape[2], math_mode)
        if result:
            raise RuntimeError(f"cuBLAS grouped dispatch failed: {result}")
        return output
