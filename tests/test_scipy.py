"""SciPy is an independent numerical oracle; never used by the implementation."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
import scipy.fft as sp
import fftlib as ft


def check(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    # Mixed absolute/relative tolerance accounts for bins close to zero.
    single = actual.dtype in (np.dtype("float32"), np.dtype("complex64"))
    atol = (3e-6 if single else 3e-13) * max(1, float(np.max(np.abs(expected), initial=0)))
    np.testing.assert_allclose(actual, expected, rtol=3e-6 if single else 3e-13, atol=atol)


@pytest.mark.parametrize("function", ["fft", "ifft", "rfft", "irfft", "hfft", "ihfft"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32])
@pytest.mark.parametrize("length", [1, 2, 3, 7, 16, 31, 1009])
@pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
def test_one_dimensional(function, dtype, length, norm):
    x = np.random.default_rng(24).normal(size=length).astype(dtype)
    # Explicit n also makes length-one inverse real transforms well-defined.
    check(getattr(ft, function)(x, n=length, norm=norm), getattr(sp, function)(x, n=length, norm=norm))


@pytest.mark.parametrize("function", ["fft", "ifft", "irfft", "hfft"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("norm", [None, "forward", "ortho"])
@pytest.mark.parametrize("n", [3, 6, 16, 31])
def test_complex_input(function, dtype, norm, n):
    rng = np.random.default_rng(4)
    x = (rng.normal(size=(5, 9)) + 1j * rng.normal(size=(5, 9))).astype(dtype)
    check(getattr(ft, function)(x, n=n, axis=1, norm=norm), getattr(sp, function)(x, n=n, axis=1, norm=norm))


@pytest.mark.parametrize("function", ["fft", "ifft", "rfft", "irfft", "hfft", "ihfft"])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("n", [None, 3, 19])
def test_strided_and_immutable(function, axis, n):
    x = np.random.default_rng(45).normal(size=(8, 9, 10))[::-2, :, ::-2]
    original = x.copy()
    x.flags.writeable = False
    check(getattr(ft, function)(x, n=n, axis=axis), getattr(sp, function)(x, n=n, axis=axis))
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("function", ["fftn", "ifftn", "rfftn", "irfftn"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("norm", [None, "forward", "ortho"])
@pytest.mark.parametrize("shape_axes", [(None, None), (None, (2, 0)), ((7, 3), (1, 0)), ((3, 6), None), ((-1, 7), (0, 2))])
def test_nd(function, dtype, norm, shape_axes):
    x = np.random.default_rng(78).normal(size=(4, 5, 6)).astype(dtype)
    s, axes = shape_axes
    check(getattr(ft, function)(x, s=s, axes=axes, norm=norm), getattr(sp, function)(x, s=s, axes=axes, norm=norm))


@pytest.mark.parametrize("function", ["fft2", "ifft2", "rfft2", "irfft2"])
def test_2d_aliases(function):
    x = np.random.default_rng(34).normal(size=(3, 7, 8))
    check(getattr(ft, function)(x), getattr(sp, function)(x))


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("function", ["fftn", "ifftn", "irfftn"])
def test_nd_complex(function, dtype):
    rng = np.random.default_rng(90)
    x = (rng.normal(size=(3, 5, 7)) + 1j * rng.normal(size=(3, 5, 7))).astype(dtype)
    check(getattr(ft, function)(x, s=(7, 6), axes=(2, 0)), getattr(sp, function)(x, s=(7, 6), axes=(2, 0)))


@pytest.mark.parametrize("function", ["fft", "ifft", "rfft", "irfft"])
@pytest.mark.parametrize("workers", [1, 2, -1])
def test_large_batch(function, workers):
    x = np.random.default_rng(9).normal(size=(64, 1024)).astype(np.float32)
    check(getattr(ft, function)(x, workers=workers), getattr(sp, function)(x, workers=workers))


def test_backend_and_workers():
    with ft.set_workers(2):
        assert ft.get_workers() == 2
        with ft.set_workers(1):
            assert ft.get_workers() == 1
        assert ft.get_workers() == 2
    assert ft.get_workers() == 1
    x = np.arange(31, dtype=np.float64)
    expected = sp.fft(x)
    with sp.set_backend(ft, only=True), sp.set_workers(2):
        check(sp.fft(x), expected)
    # Unsupported transforms fall through to SciPy when only=False.
    expected_dct = sp.dct(x)
    with sp.set_backend(ft):
        check(sp.dct(x), expected_dct)
    ft.clear_cache()
    check(ft.fft(x), expected)


def test_concurrent_calls():
    x = np.arange(32768, dtype=np.float64).reshape(32, 1024)
    with ThreadPoolExecutor(4) as executor:
        results = list(executor.map(lambda _: ft.rfft(x, workers=2), range(8)))
    for result in results:
        check(result, sp.rfft(x))


@pytest.mark.parametrize("function", ["fft", "rfft", "irfft"])
def test_empty_batches(function):
    check(getattr(ft, function)(np.empty((0, 7))), getattr(sp, function)(np.empty((0, 7))))
    check(getattr(ft, function)([], n=7), getattr(sp, function)([], n=7))


def test_byte_order_and_fortran_arrays():
    x = np.arange(48, dtype=">f8").reshape(6, 8)
    check(ft.fft(x, axis=0), sp.fft(x, axis=0))
    x = np.asfortranarray(x.astype(np.float32))
    check(ft.rfftn(x), sp.rfftn(x))


@pytest.mark.parametrize("kwargs", [{"n": 0}, {"n": -1}, {"axis": 3}, {"norm": "bad"}, {"workers": 0}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        ft.fft([1, 2, 3], **kwargs)


def test_invalid_nd_and_types():
    with pytest.raises(ValueError):
        ft.fftn(np.ones((3, 4)), axes=(0, 0))
    with pytest.raises(ValueError):
        ft.fftn(np.ones((3, 4)), s=(5,), axes=(0, 1))
    with pytest.raises(TypeError):
        ft.rfft([1j, 2j])
    with pytest.raises(NotImplementedError):
        ft.fft([1, 2], plan=object())
    if np.dtype(np.longdouble).itemsize > 8:
        with pytest.raises(NotImplementedError):
            ft.fft(np.ones(4, dtype=np.longdouble))
    check(ft.fftn([1, 2], axes=()), sp.fftn([1, 2], axes=()))


def test_roundtrip_and_parseval():
    rng = np.random.default_rng(88)
    x = rng.normal(size=(7, 31))
    for norm in [None, "forward", "ortho"]:
        np.testing.assert_allclose(ft.irfftn(ft.rfftn(x, norm=norm), s=x.shape, norm=norm), x, atol=2e-14)
    spectrum = ft.fftn(x, norm="ortho")
    np.testing.assert_allclose(np.sum(np.abs(spectrum) ** 2), np.sum(x ** 2), rtol=1e-13)


@pytest.mark.parametrize("function", ["fftn", "ifftn"])
@pytest.mark.parametrize("dtype", [np.int32, np.float16, np.float32, np.float64, np.complex64, np.complex128])
def test_no_axes_preserves_dtype(function, dtype):
    x = np.arange(6).astype(dtype).reshape(2, 3)
    check(getattr(ft, function)(x, axes=()), getattr(sp, function)(x, axes=()))


@pytest.mark.parametrize("function", ["fftn", "ifftn", "rfftn", "irfftn"])
@pytest.mark.parametrize("axes", [(0, 1, 2), (2, 0), (1, 0, 2)])
@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nd_native_strided_parallel(function, axes, workers, dtype):
    x = np.random.default_rng(53).normal(size=(34, 36, 130)).astype(dtype)[::-1, ::2, ::-2]
    original = x.copy()
    check(getattr(ft, function)(x, axes=axes, workers=workers), getattr(sp, function)(x, axes=axes, workers=workers))
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("function", ["fftn", "ifftn", "rfftn", "irfftn"])
def test_nd_empty_batch_native(function):
    x = np.empty((0, 7, 9))
    check(getattr(ft, function)(x, axes=(2, 1), s=(11, 5)), getattr(sp, function)(x, axes=(2, 1), s=(11, 5)))


@pytest.mark.parametrize("function", ["fftn", "ifftn", "rfftn", "irfftn"])
@pytest.mark.parametrize("shape", [(1, 1, 7), (7, 1, 1), (1, 7, 1), (2, 1, 7)])
def test_nd_singleton_dimensions(function, shape):
    x = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    check(getattr(ft, function)(x, s=shape, norm="ortho"), getattr(sp, function)(x, s=shape, norm="ortho"))


@pytest.mark.parametrize("function", ["fft", "ifft", "rfft", "irfft"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("workers", [2, 4, 7])
def test_large_threaded_copy_and_outputs(function, dtype, workers):
    # Irregular row count exercises partial worker chunks. Input exceeds the
    # parallel-copy threshold, including in single precision.
    x = np.random.default_rng(49).normal(size=(137, 4096)).astype(dtype)
    original = x.copy()
    for norm in (None, "ortho"):
        check(getattr(ft, function)(x, norm=norm, workers=workers),
              getattr(sp, function)(x, norm=norm, workers=workers))
    np.testing.assert_array_equal(x, original)


def test_real_workspace_reuse_preserves_previous_output():
    x = np.random.default_rng(6).normal(size=(137, 4096))
    first = ft.rfft(x, workers=4)
    saved = first.copy()
    second = ft.rfft(x * 3, workers=4)
    np.testing.assert_array_equal(first, saved)
    check(second, sp.rfft(x * 3))
    ft.clear_cache()
    check(ft.rfft(x, workers=4), sp.rfft(x))


def test_concurrent_real_workspace_reuse():
    x = np.random.default_rng(29).normal(size=(137, 4096))
    with ThreadPoolExecutor(4) as executor:
        results = list(executor.map(lambda scale: ft.rfft(x * scale, workers=4), range(1, 7)))
    for scale, result in enumerate(results, start=1):
        check(result, sp.rfft(x * scale))
