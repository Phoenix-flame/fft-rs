"""Interleave timings of a previous package snapshot, current fftlib, and SciPy."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
import timeit

import numpy as np
import scipy
import scipy.fft
import fftlib


def load_previous(directory):
    spec = importlib.util.spec_from_file_location("fftlib_previous", directory / "__init__.py",
                                                submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="snapshot of the previous fftlib package, including its native extension")
    parser.add_argument("--json", type=Path, default=Path("benchmarks/optimization-results.json"))
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--baseline-commit", default=None)
    args = parser.parse_args()
    if args.repeats < 1 or any(worker < 1 for worker in args.workers):
        parser.error("repeats and worker counts must be positive")
    previous = load_previous(args.baseline.resolve())
    rng = np.random.default_rng(62)
    cases = [("fft", (65536,)), ("rfft", (65536,)), ("fft", (64, 1024)),
             ("rfft", (256, 4096)), ("fft2", (512, 512)), ("rfft2", (512, 512)),
             ("fftn", (64, 64, 64)), ("rfftn", (64, 64, 64))]
    rows = []
    print(f"{'case':45} {'workers':>7} {'before µs':>11} {'after µs':>11} {'improvement':>11} {'vs SciPy':>10}")
    for workers in args.workers:
        for dtype in (np.float32, np.float64):
            for name, shape in cases:
                x = rng.normal(size=shape).astype(dtype)
                if name in ("fft", "fft2", "fftn"):
                    x = (x + 1j * rng.normal(size=shape).astype(dtype)).astype(np.complex64 if dtype is np.float32 else np.complex128)
                functions = [getattr(m, name) for m in (previous, fftlib, scipy.fft)]
                outputs = [fn(x, workers=workers) for fn in functions]
                tolerance = 3e-6 if dtype is np.float32 else 3e-13
                for output in outputs[:2]:
                    np.testing.assert_allclose(output, outputs[2], rtol=tolerance,
                                               atol=tolerance * max(1.0, float(np.max(np.abs(outputs[2])))))
                timers = [timeit.Timer(lambda fn=fn: fn(x, workers=workers)) for fn in functions]
                initial = [timer.timeit(1) for timer in timers]
                number = max(3, min(500, int(0.03 / max(initial))))
                timings = [[], [], []]
                # Rotate measurement order so thermal/scheduling drift doesn't
                # consistently favor one implementation.
                for repetition in range(args.repeats):
                    for offset in range(3):
                        index = (repetition + offset) % 3
                        timings[index].append(timers[index].timeit(number) / number)
                before, after, scipy_time = (statistics.median(values) for values in timings)
                row = dict(transform=name, shape=list(shape), dtype=str(x.dtype), workers=workers,
                           before_seconds=before, after_seconds=after, scipy_seconds=scipy_time,
                           improvement=before / after, scipy_speedup=scipy_time / after)
                rows.append(row)
                print(f"{name + ' ' + str(shape) + ' ' + str(x.dtype):45} {workers:7} {before*1e6:11.2f} {after*1e6:11.2f} {row['improvement']:10.2f}x {row['scipy_speedup']:9.2f}x", flush=True)
    args.json.write_text(json.dumps(dict(python=platform.python_version(), scipy=scipy.__version__,
                                        numpy=np.__version__, fftlib=fftlib.__version__,
                                        baseline_commit=args.baseline_commit,
                                        timing=f"{args.repeats} interleaved repeats, median, warmed plans, end-to-end calls",
                                        records=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
