"""Rust FFTs with scipy.fft conventions (floating-point, not bitwise parity)."""
from contextlib import contextmanager
from contextvars import ContextVar
import operator
import os

import numpy as np
from . import _native

__version__ = "0.1.0"
__all__ = [
    "fft", "ifft", "rfft", "irfft", "hfft", "ihfft",
    "fftn", "ifftn", "rfftn", "irfftn", "fft2", "ifft2", "rfft2", "irfft2",
    "fftshift", "ifftshift", "fftfreq", "rfftfreq", "set_workers", "get_workers",
    "clear_cache",
]
_workers = ContextVar("fftlib_workers", default=1)


def _worker_count(workers):
    workers = _workers.get() if workers is None else operator.index(workers)
    if workers < 0:
        workers += (os.cpu_count() or 1) + 1
    if workers < 1:
        raise ValueError("workers must be nonzero and at least -os.cpu_count()")
    return workers


@contextmanager
def set_workers(workers):
    """Set the default batch worker count in this context."""
    token = _workers.set(_worker_count(workers))
    try:
        yield
    finally:
        _workers.reset(token)


def get_workers():
    return _workers.get()


def clear_cache():
    """Release cached FFT plans and real input buffers; active transforms remain valid."""
    _native.clear_cache()


def _normalization(norm):
    if norm is None:
        return "backward"
    if norm not in ("backward", "forward", "ortho"):
        raise ValueError("norm must be None, 'backward', 'forward', or 'ortho'")
    return norm


def _array(x, complex_output=False, real_input=False):
    x = np.asarray(x)
    if real_input and np.iscomplexobj(x):
        raise TypeError("x must be a real sequence")
    dtype = x.dtype
    if dtype.kind in "fc" and dtype.itemsize > (16 if dtype.kind == "c" else 8):
        raise NotImplementedError("extended precision is not supported; use SciPy for longdouble")
    single = dtype.kind == "f" and dtype.itemsize <= 4 or dtype.kind == "c" and dtype.itemsize <= 8
    target = (np.complex64 if single else np.complex128) if complex_output else (np.float32 if single else np.float64)
    return np.asarray(x, dtype=target)


def _axis(axis, ndim):
    axis = operator.index(axis)
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis % ndim


def _transform(x, n, axis, norm, workers, plan, kind):
    if plan is not None:
        raise NotImplementedError("precomputed plans are not supported")
    norm = _normalization(norm)
    workers = _worker_count(workers)
    x = _array(x, complex_output=kind != "r2c", real_input=kind == "r2c")
    axis = _axis(axis, x.ndim)
    moved = x if axis == x.ndim - 1 else np.moveaxis(x, axis, -1)
    current = moved.shape[-1]
    n = (2 * (current - 1) if kind == "c2r" else current) if n is None else operator.index(n)
    if n < 1:
        raise ValueError(f"invalid number of data points ({n}) specified")
    input_width = n // 2 + 1 if kind == "c2r" else n
    if input_width < current:
        moved = moved[..., :input_width]
    elif input_width > current:
        padded = np.zeros(moved.shape[:-1] + (input_width,), dtype=moved.dtype)
        padded[..., :current] = moved
        moved = padded
    data = np.ascontiguousarray(moved).reshape(-1)
    precision = "f32" if x.dtype in (np.dtype("float32"), np.dtype("complex64")) else "f64"
    if kind in ("fft", "ifft"):
        result = getattr(_native, "c2c_" + precision)(data, n, kind == "ifft", norm, workers)
        output_width = n
    else:
        result = getattr(_native, kind + "_" + precision)(data, n, norm, workers)
        output_width = n // 2 + 1 if kind == "r2c" else n
    result = result.reshape(moved.shape[:-1] + (output_width,))
    return result if axis == x.ndim - 1 else np.moveaxis(result, -1, axis)


def fft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None):
    return _transform(x, n, axis, norm, workers, plan, "fft")


def ifft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None):
    return _transform(x, n, axis, norm, workers, plan, "ifft")


def rfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None):
    return _transform(x, n, axis, norm, workers, plan, "r2c")


def irfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None):
    return _transform(x, n, axis, norm, workers, plan, "c2r")


def _swap_norm(norm):
    return {"backward": "forward", "forward": "backward", "ortho": "ortho"}[_normalization(norm)]


def hfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None):
    return irfft(np.conj(x), n, axis, _swap_norm(norm), overwrite_x, workers, plan=plan)


def ihfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None):
    return np.conj(rfft(x, n, axis, _swap_norm(norm), overwrite_x, workers, plan=plan))


def _shape_axes(x, s, axes):
    if s is not None:
        s = tuple(operator.index(n) for n in s)
    if axes is None:
        axes = tuple(range(x.ndim)) if s is None else tuple(range(x.ndim - len(s), x.ndim))
    else:
        axes = tuple(_axis(a, x.ndim) for a in axes)
    if s is not None and len(s) != len(axes):
        raise ValueError("shape and axes have different lengths")
    if len(axes) > x.ndim or any(a < 0 for a in axes):
        raise ValueError("shape requires more axes than are present")
    if len(set(axes)) != len(axes):
        raise ValueError("all axes must be unique")
    lengths = tuple(x.shape[a] for a in axes) if s is None else tuple(x.shape[a] if n == -1 else n for a, n in zip(axes, s))
    if any(n < 1 for n in lengths):
        raise ValueError("invalid number of data points specified")
    return lengths, axes


def _nd(x, s, axes, norm, workers, plan, kind):
    if plan is not None:
        raise NotImplementedError("precomputed plans are not supported")
    norm = _normalization(norm)
    workers = _worker_count(workers)
    x = np.asarray(x)
    lengths, axes = _shape_axes(x, s, axes)
    if not axes:
        if kind in ("rfft", "irfft"):
            raise ValueError("at least one axis is required for a real transform")
        return x.copy()
    x = _array(x, complex_output=kind != "rfft", real_input=kind == "rfft")
    if kind == "irfft" and s is None:
        lengths = lengths[:-1] + (2 * (x.shape[axes[-1]] - 1),)
    if any(n < 1 for n in lengths):
        raise ValueError("invalid number of data points specified")
    input_shape = list(x.shape)
    for axis, length in zip(axes, lengths):
        input_shape[axis] = length
    if kind == "irfft":
        input_shape[axes[-1]] = lengths[-1] // 2 + 1
    if tuple(input_shape) != x.shape:
        resized = np.zeros(input_shape, dtype=x.dtype)
        slices = tuple(slice(0, min(old, new)) for old, new in zip(x.shape, input_shape))
        resized[slices] = x[slices]
        x = resized
    data = np.ascontiguousarray(x).reshape(-1)
    precision = "f32" if x.dtype in (np.dtype("float32"), np.dtype("complex64")) else "f64"
    if kind in ("fft", "ifft"):
        result, shape, order = getattr(_native, "c2c_nd_" + precision)(
            data, x.shape, axes, kind == "ifft", norm, workers)
    elif kind == "rfft":
        result, shape, order = getattr(_native, "r2c_nd_" + precision)(
            data, x.shape, axes, norm, workers)
    else:
        result, shape, order = getattr(_native, "c2r_nd_" + precision)(
            data, x.shape, axes, lengths[-1], norm, workers)
    result = result.reshape(shape)
    inverse_order = tuple(order.index(axis) for axis in range(x.ndim))
    return result.transpose(inverse_order)


def fftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *, plan=None):
    return _nd(x, s, axes, norm, workers, plan, "fft")


def ifftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *, plan=None):
    return _nd(x, s, axes, norm, workers, plan, "ifft")


def rfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *, plan=None):
    return _nd(x, s, axes, norm, workers, plan, "rfft")


def irfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *, plan=None):
    return _nd(x, s, axes, norm, workers, plan, "irfft")


def fft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *, plan=None):
    return fftn(x, s, axes, norm, overwrite_x, workers, plan=plan)


def ifft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *, plan=None):
    return ifftn(x, s, axes, norm, overwrite_x, workers, plan=plan)


def rfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *, plan=None):
    return rfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)


def irfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *, plan=None):
    return irfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)


fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift
fftfreq = np.fft.fftfreq
rfftfreq = np.fft.rfftfreq

# Optional scipy.fft.set_backend(fftlib) integration. Unsupported functions can
# fall through to another SciPy backend; direct fftlib calls never import SciPy.
__ua_domain__ = "numpy.scipy.fft"
_backend_functions = {name: globals()[name] for name in __all__ if name not in ("set_workers", "get_workers", "clear_cache")}


def __ua_function__(method, args, kwargs):
    function = _backend_functions.get(method.__name__)
    if function is None:
        return NotImplemented
    if method.__name__ in {"fft", "ifft", "rfft", "irfft", "hfft", "ihfft", "fftn", "ifftn", "rfftn", "irfftn", "fft2", "ifft2", "rfft2", "irfft2"} and kwargs.get("workers") is None:
        import scipy.fft
        kwargs = dict(kwargs, workers=scipy.fft.get_workers())
    try:
        return function(*args, **kwargs)
    except NotImplementedError:
        return NotImplemented
