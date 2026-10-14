"""Compare fftlib and SciPy on a Hamming-windowed harmonic sine signal.

Run with the project's Python environment after installing .[plot].
The FFT timings include conversion and allocation, but exclude signal generation,
windowing, plotting, and first-call plan creation.
"""
import argparse
import json
from pathlib import Path
import platform
import statistics
import timeit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.fft as scipy_fft
from scipy.signal.windows import hamming

import fftlib


HARMONICS = (1, 2, 3, 5, 7)
AMPLITUDES = (1.0, 0.5, 0.25, 0.15, 0.08)
PHASES = (0.0, 0.4, -0.7, 1.1, -0.3)


def timed_rfft(function, x):
    function(x, workers=1)
    timer = timeit.Timer(lambda: function(x, workers=1))
    initial = timer.timeit(number=1)
    number = max(5, min(1000, int(0.05 / max(initial, 1e-9))))
    return statistics.median(timer.repeat(repeat=7, number=number)) / number


def amplitude(spectrum, window, n):
    """One-sided amplitude, with coherent-gain correction and endpoint weights."""
    result = 2 * np.abs(spectrum).astype(np.float64) / np.sum(window, dtype=np.float64)
    result[0] *= 0.5
    if n % 2 == 0:
        result[-1] *= 0.5
    return result


def validate(signal, window, dtype, frequencies, fundamental):
    window = window.astype(dtype)
    windowed = signal.astype(dtype) * window
    actual = fftlib.rfft(windowed, workers=1)
    expected = scipy_fft.rfft(windowed, workers=1)
    tolerance = 3e-6 if dtype is np.float32 else 3e-13
    scale = max(1.0, float(np.max(np.abs(expected))))
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance * scale)

    full_actual = fftlib.fft(windowed, workers=1)
    full_expected = scipy_fft.fft(windowed, workers=1)
    np.testing.assert_allclose(full_actual, full_expected, rtol=tolerance, atol=tolerance * scale)
    reconstructed = fftlib.irfft(actual, n=len(windowed), workers=1)
    np.testing.assert_allclose(reconstructed, windowed, rtol=tolerance,
                               atol=tolerance * max(1.0, float(np.max(np.abs(windowed)))))

    # Widen before computing differences and norms to avoid rounding the metrics
    # themselves to float32 when validating the single-precision transforms.
    difference = actual.astype(np.complex128) - expected.astype(np.complex128)
    normalized_error = amplitude(difference, window, len(windowed))
    rust_amplitude = amplitude(actual, window, len(windowed))
    scipy_amplitude = amplitude(expected, window, len(windowed))
    spacing = frequencies[1] - frequencies[0]
    peaks = []
    for harmonic, target_amplitude in zip(HARMONICS, AMPLITUDES):
        target = fundamental * harmonic
        candidates = np.flatnonzero(np.abs(frequencies - target) <= 2 * spacing)
        index = int(candidates[np.argmax(scipy_amplitude[candidates])])
        assert abs(frequencies[index] - target) <= spacing / 2 + 1e-12
        peaks.append(dict(harmonic=harmonic, input_frequency_hz=target,
                          peak_bin_hz=float(frequencies[index]), input_amplitude=target_amplitude,
                          rust_peak_amplitude=float(rust_amplitude[index]),
                          scipy_peak_amplitude=float(scipy_amplitude[index])))

    rust_time = timed_rfft(fftlib.rfft, windowed)
    scipy_time = timed_rfft(scipy_fft.rfft, windowed)
    metrics = dict(dtype=np.dtype(dtype).name, spectrum_dtype=str(actual.dtype),
                   max_absolute_fft_error=float(np.max(np.abs(difference))),
                   relative_l2_fft_error=float(np.linalg.norm(difference) / np.linalg.norm(expected.astype(np.complex128))),
                   max_amplitude_scaled_complex_error=float(np.max(normalized_error)),
                   max_full_fft_error=float(np.max(np.abs(full_actual.astype(np.complex128) - full_expected.astype(np.complex128)))),
                   max_roundtrip_error=float(np.max(np.abs(reconstructed.astype(np.float64) - windowed.astype(np.float64)))),
                   rust_rfft_seconds=rust_time, scipy_rfft_seconds=scipy_time,
                   speedup=scipy_time / rust_time, peaks=peaks, validation="passed")
    return metrics, dict(windowed=windowed, rust_amplitude=rust_amplitude,
                         scipy_amplitude=scipy_amplitude, error=normalized_error)


def plot_comparison(t, signal, window, frequencies, data, metrics, fundamental, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.grid": True,
                         "grid.alpha": 0.20, "figure.facecolor": "white"})
    fig = plt.figure(figsize=(13.5, 10), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(1, 1.5, 1))
    ax_signal = fig.add_subplot(grid[0, 0])
    ax_window = fig.add_subplot(grid[0, 1])
    ax_spectrum = fig.add_subplot(grid[1, :])
    ax_error = fig.add_subplot(grid[2, :])
    colors = dict(rust="#c45513", scipy="#176487", single="#a54984", double="#24775b")

    # A start-of-record excerpt makes the Hamming taper visible alongside the
    # waveform. The separate window plot shows the taper over the whole record.
    excerpt = t <= 5 / fundamental
    ax_signal.plot(t[excerpt] * 1000, signal[excerpt], color="#42536b", lw=1.4, label="Original signal")
    ax_signal.plot(t[excerpt] * 1000, data["float64"]["windowed"][excerpt], color=colors["rust"], lw=1.5, label="After Hamming window")
    ax_signal.set(title="Five harmonics · first five cycles", xlabel="Time (ms)", ylabel="Amplitude")
    ax_signal.legend(loc="upper right", fontsize=9, frameon=False)
    ax_window.plot(t, window, color=colors["double"], lw=2)
    ax_window.fill_between(t, window, color=colors["double"], alpha=0.10)
    ax_window.set(title="Periodic Hamming window", xlabel="Time (s)", ylabel="Window weight", ylim=(0, 1.07))

    double = data["float64"]
    display = frequencies <= min(frequencies[-1], fundamental * 8)
    floor = 10 ** (-180 / 20)
    ax_spectrum.plot(frequencies[display], 20 * np.log10(np.maximum(double["scipy_amplitude"][display], floor)),
                     color=colors["scipy"], lw=2.2, label="SciPy rfft · float64")
    ax_spectrum.plot(frequencies[display], 20 * np.log10(np.maximum(double["rust_amplitude"][display], floor)),
                     color=colors["rust"], lw=1.3, ls="--", label="Rust fftlib.rfft · float64")
    for harmonic in HARMONICS:
        frequency = fundamental * harmonic
        ax_spectrum.axvline(frequency, color="#748393", alpha=0.35, lw=0.8, ls=":")
        ax_spectrum.text(frequency, 1.5, f"{harmonic}f₀\n{frequency:g} Hz", ha="center", va="bottom", fontsize=9)
    ax_spectrum.set(title="The spectra overlap at every harmonic", xlabel="Frequency (Hz)",
                    ylabel="One-sided amplitude (dB re 1)", xlim=(0, fundamental * 8), ylim=(-140, 11))
    ax_spectrum.legend(loc="lower left", fontsize=10, frameon=False)

    for dtype, color in [("float32", colors["single"]), ("float64", colors["double"])]:
        ax_error.semilogy(frequencies, np.maximum(data[dtype]["error"], 1e-20), color=color,
                          lw=0.8, alpha=0.9, label=f"{dtype}: relative L2 error {metrics[dtype]['relative_l2_fft_error']:.2e}")
    ax_error.set(title="Numerical difference · all FFT bins", xlabel="Frequency (Hz)",
                 ylabel="Amplitude-scaled |Rust − SciPy|", xlim=(0, frequencies[-1]))
    ax_error.legend(loc="upper right", fontsize=10, frameon=False)
    ax_error.text(0.015, 0.035, "DC/Nyquist weighted once; exact zero differences drawn at 10⁻²⁰.",
                  transform=ax_error.transAxes, fontsize=9, color="#546575")
    fig.suptitle("Hamming-windowed harmonic signal: Rust vs SciPy\n"
                 f"f₀ = {fundamental:g} Hz  |  N = {len(t):,}  |  Δf = {frequencies[1]:g} Hz  |  double-precision spectrum",
                 fontsize=17, fontweight="bold")
    fig.savefig(output / "harmonic_comparison.png", dpi=180, facecolor="white")
    fig.savefig(output / "harmonic_comparison.svg", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=65536)
    parser.add_argument("--sample-rate", type=float, default=8192)
    parser.add_argument("--fundamental", type=float, default=123.4)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.samples < 16 or not np.isfinite(args.sample_rate) or args.sample_rate <= 0:
        parser.error("use at least 16 samples and a finite positive sample rate")
    if not np.isfinite(args.fundamental) or args.fundamental <= 0 or max(HARMONICS) * args.fundamental >= args.sample_rate / 2:
        parser.error("all harmonics must be positive and below the Nyquist frequency")
    spacing = args.sample_rate / args.samples
    if args.fundamental < 8 * spacing:
        parser.error("increase sample count to resolve the individual harmonic peaks")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(args.samples, dtype=np.float64) / args.sample_rate
    signal = sum(a * np.sin(2 * np.pi * args.fundamental * h * t + phase)
                 for h, a, phase in zip(HARMONICS, AMPLITUDES, PHASES))
    window = hamming(args.samples, sym=False)
    frequencies = scipy_fft.rfftfreq(args.samples, d=1 / args.sample_rate)
    metrics, data = {}, {}
    for dtype in (np.float64, np.float32):
        name = np.dtype(dtype).name
        metrics[name], data[name] = validate(signal, window, dtype, frequencies, args.fundamental)
    plot_comparison(t, signal, window, frequencies, data, metrics, args.fundamental, args.output_dir)
    report = dict(samples=args.samples, sample_rate_hz=args.sample_rate, duration_seconds=args.samples / args.sample_rate,
                  frequency_spacing_hz=spacing, fundamental_hz=args.fundamental,
                  harmonic_numbers=HARMONICS, amplitudes=AMPLITUDES, phases_radians=PHASES,
                  window="periodic Hamming (sym=False)", coherent_gain=float(np.mean(window)),
                  amplitude_normalization="2*abs(rfft)/sum(window); DC and Nyquist weighted once",
                  note="Off-bin harmonics exhibit spectral leakage and scalloping; nearest-bin amplitudes can differ from input amplitudes.",
                  versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                matplotlib=matplotlib.__version__, fftlib=fftlib.__version__),
                  timing="Median of 7 warmed repetitions; end-to-end rfft call, one worker, no windowing or plotting.",
                  results=metrics)
    report_path = args.output_dir / "harmonic_comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for name, result in metrics.items():
        print(f"{name}: PASS | max FFT difference {result['max_absolute_fft_error']:.3e} | "
              f"relative L2 {result['relative_l2_fft_error']:.3e} | "
              f"Rust {result['rust_rfft_seconds']*1e6:.1f} µs / SciPy {result['scipy_rfft_seconds']*1e6:.1f} µs | "
              f"{result['speedup']:.2f}x")
    print(f"Saved plots and metrics in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
