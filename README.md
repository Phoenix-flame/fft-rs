# fftlib

A Rust FFT library with a Python interface following `scipy.fft` conventions.
The FFT computation uses RustFFT and RealFFT; Python handles shapes and NumPy
conversion. SciPy is used only in tests, benchmarks, and optional backend dispatch.

**Numerical equivalence, not bit-for-bit equality.** Different FFT algorithms,
SIMD instructions, and operation order produce different rounding. Single and
double precision are supported. Validation uses SciPy as an independent oracle.
No universal speedup over SciPy is claimed: SciPy already uses compiled PocketFFT.

## Build

Install a stable Rust toolchain and Python 3.10+, then from this directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install maturin numpy
maturin develop --release
```

For a distributable wheel, use `maturin build --release` and install the wheel
from `target/wheels`. Default builds select SIMD instructions at runtime. For a
build restricted to your own CPU, optionally set `RUSTFLAGS="-C target-cpu=native"`.
Do not distribute that CPU-specific wheel to machines with fewer CPU features.

## Python

```python
import numpy as np
import fftlib as fft

x = np.random.default_rng(0).normal(size=(64, 4096)).astype(np.float32)
spectrum = fft.rfft(x, axis=-1, workers=4)
recovered = fft.irfft(spectrum, n=x.shape[-1], workers=4)
np.testing.assert_allclose(recovered, x, rtol=1e-5, atol=1e-6)

with fft.set_workers(4):
    image_spectrum = fft.rfft2(x, norm="ortho")
```

Supported transforms: `fft`, `ifft`, `rfft`, `irfft`, `hfft`, `ihfft`, `fftn`,
`ifftn`, `rfftn`, `irfftn`, and the corresponding `fft2`/`ifft2`/`rfft2`/`irfft2`.
Also exposes NumPy's frequency and shift helpers. Supports positive/negative axes,
cropping, zero padding, arbitrary lengths, strided inputs, batch dimensions, and
`backward`, `forward`, and `ortho` normalization. `s=-1` entries retain input lengths.
Pass the original odd length explicitly to inverse real transforms.

Float16 promotes to float32; float32/complex64 retain single precision;
integers and float64/complex128 use double precision. Extended precision raises
`NotImplementedError`. Complex input to forward real transforms raises `TypeError`.
Inputs remain untouched, including when `overwrite_x=True` (permission to overwrite
is accepted but not used). Non-None `plan` arguments are unsupported.

Optional SciPy backend:

```python
import scipy.fft
import fftlib

with scipy.fft.set_backend(fftlib):
    y = scipy.fft.fft(x)
```

Unsupported functions fall through to SciPy unless `only=True` is used. This
package does not implement DCT/DST, multidimensional Hermitian transforms,
`next_fast_len`, GPU arrays, or the entire SciPy FFT namespace.

## Rust

The native library target is named `_native` so that it also serves the Python
extension; Rust callers import that target directly:

```rust
use _native::{Complex, Engine, Norm};

fn main() -> Result<(), String> {
    let engine = Engine::<f64>::default();
    let mut signal = vec![Complex::new(1.0, 0.0); 1024];
    engine.c2c(&mut signal, 1024, false, Norm::Backward, 1)?;
    Ok(())
}
```

`Engine::c2c` transforms adjacent rows in place; `r2c` and `c2r` use mutable input
workspace and return output vectors. Reuse an engine to amortize plan creation.
Positive `workers` controls batch parallelism. A single row runs on one thread.
Python releases the GIL during FFT work and transfers owned output to NumPy.
`c2c_nd`, `r2c_nd`, and `c2r_nd` retain ownership across N-D axis passes. They
return an `Array` containing `data`, physical `shape`, and an `order` mapping each
physical axis to its original logical axis. Python restores logical order with
a NumPy view. Rust callers can use this metadata to index the returned workspace.

## Validation and performance

```sh
cargo test
python -m pip install '.[test]'
python -m pytest -q
python benchmarks/compare_scipy.py --workers 1
python benchmarks/compare_scipy.py --workers 4 --json benchmark-workers4.json
```

Tests cover normalization, dtype, prime and odd sizes, shapes, strides, real
endpoint handling, concurrent calls, batch workers, roundtrips, and Parseval's
identity. SciPy comparisons use mixed absolute and relative tolerances scaled by
the largest output bin: `3e-6` for single and `3e-13` for double precision. These
are test tolerances, not a formal error bound for every possible input.

The benchmark measures the full Python call, including conversion, copying,
allocation, and shape handling. Plans and worker pools are warmed for both
implementations. It prints median timings and records versions and numerical
errors in JSON. A SciPy/Rust ratio greater than 1 means fftlib is faster.

Optimizations include runtime SIMD dispatch, cached plans and worker pools,
specialized real FFTs, per-job scratch reuse, and a threshold that avoids batch
threading overhead for small inputs. N-D transforms now run in Rust without
intermediate Python-to-Rust copies. Axis gathering and complex FFT execution are
combined using small column blocks and reusable scratch space; singleton axis
rearrangements avoid copying. Array conversion and axis rearrangement can still
dominate some transforms, and PocketFFT can remain faster for threaded workloads.

FFT plans are cached per length and precision without a fixed bound. Call
`fftlib.clear_cache()` or `Engine::clear_cache()` to release them when processing
many distinct lengths. Worker pools are cached per requested positive count;
reuse a small set of worker counts. Scratch space is reused within each batch,
but allocated again for subsequent calls.

## Measured results

Validated on an Intel Core i5-11400H with Python 3.12.14, NumPy 2.3.5, and SciPy
1.15.3: **1,057 Python tests and two Rust tests passed**. The packaged Linux CPython
3.12 wheel also passed an independent import and 2-D roundtrip check.

Representative median end-to-end timings with one worker:

| Transform | Rust | SciPy | SciPy/Rust |
| --- | ---: | ---: | ---: |
| `fft`, 4,096 complex64 samples | 7.62 µs | 25.12 µs | 3.30× |
| `fft`, 1,048,576 complex64 samples | 6.74 ms | 19.65 ms | 2.92× |
| `rfft`, 65,536 float32 samples | 94.98 µs | 293.47 µs | 3.09× |
| `fft2`, 512×512 complex128 samples | 2.14 ms | 2.22 ms | 1.04× |

Rust was faster in 21 of 24 one-worker cases. Maximum observed relative L2
disagreement across these benchmark inputs was `3.10e-7` in single precision
and `8.46e-16` in double precision. Four-worker batched and 2-D transforms often
favored SciPy, whose integrated implementation has less copying and dispatch
overhead. These measurements describe this machine and these inputs only.

Full measurements: [one worker](benchmarks/results-workers1.json) and
[four workers](benchmarks/results-workers4.json). Re-run the benchmark on your
own workload before choosing the engine and worker count.

The N-D optimization was also measured against a snapshot of the earlier
implementation, using seven interleaved timing repetitions with warmed plans:

| Transform | Workers | Before | After | Improvement |
| --- | ---: | ---: | ---: | ---: |
| `fft2`, 512×512 complex128 | 1 | 3.20 ms | 2.03 ms | 1.58× |
| `fft2`, 512×512 complex64 | 4 | 1.12 ms | 0.50 ms | 2.25× |
| `fftn`, 64×64×64 complex128 | 4 | 3.98 ms | 1.40 ms | 2.84× |
| `rfftn`, 64×64×64 float64 | 4 | 1.14 ms | 0.66 ms | 1.73× |

One-dimensional kernels are unchanged; small timing differences for those cases
are measurement variation. Full results are in
[optimization-results.json](benchmarks/optimization-results.json). A local
baseline package snapshot is saved in `target/benchmark-baseline`; compare it
again with `python benchmarks/compare_versions.py --baseline target/benchmark-baseline`.

## Hamming-windowed harmonic comparison

```sh
python -m pip install '.[plot]'
python examples/harmonic_comparison.py
```

This example generates a 123.4 Hz fundamental and its second, third, fifth, and
seventh harmonics, applies a periodic Hamming window, and compares both `fft` and
`rfft` with SciPy in float32 and float64. It also checks inverse reconstruction,
measures warmed FFT timings, and saves Matplotlib PNG/SVG plots and JSON metrics
in `artifacts/`. The one-sided spectrum corrects for the window's coherent gain.
The frequencies intentionally fall between FFT bins, so measured peak amplitudes
include scalloping loss. Use `--help` to customize the signal and output directory.
The saved [plot](artifacts/harmonic_comparison.png) shows overlapping spectra;
[metrics](artifacts/harmonic_comparison.json) record relative L2 differences of
`4.50e-16` for float64 and `1.96e-7` for float32. On this signal, warmed `rfft`
calls were 1.89× and 2.91× faster than SciPy, respectively.

Implementation references: [SciPy FFT conventions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html),
[RustFFT runtime SIMD planning](https://docs.rs/rustfft/latest/rustfft/), and
[RealFFT algorithms and scaling](https://docs.rs/realfft/latest/realfft/).
