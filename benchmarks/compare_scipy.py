"""End-to-end timings, including array conversion and output allocation."""
import argparse
import json
import os
import platform
from pathlib import Path
import statistics
import timeit
import numpy as np
import scipy
import scipy.fft as sp
import fftlib as ft


def timing(function, x, workers, repeats):
    function(x, workers=workers)  # warm plan and thread-pool caches
    timer = timeit.Timer(lambda: function(x, workers=workers))
    first = timer.timeit(number=1)
    number = max(3, min(1000, int(0.03 / max(first, 1e-9))))
    return statistics.median(timer.repeat(repeat=repeats, number=number)) / number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", dest="json_path", default="benchmark-results.json")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    rng = np.random.default_rng(2026)
    cases = [("fft", (n,)) for n in (32, 1009, 4096, 65536, 1048576)]
    cases += [("rfft", (n,)) for n in (1024, 65536, 1048576)]
    cases += [("fft", (64, 1024)), ("rfft", (256, 4096)), ("fft2", (512, 512)), ("rfft2", (512, 512))]
    records = []
    print(f"{'transform/shape/dtype':44} {'Rust µs':>11} {'SciPy µs':>11} {'SciPy/Rust':>11}")
    for dtype in (np.float32, np.float64):
        for name, shape in cases:
            x = rng.standard_normal(shape).astype(dtype)
            if name in ("fft", "fft2"):
                x = (x + 1j * rng.standard_normal(shape).astype(dtype)).astype(np.complex64 if dtype is np.float32 else np.complex128)
            rust_function, scipy_function = getattr(ft, name), getattr(sp, name)
            actual, expected = rust_function(x, workers=args.workers), scipy_function(x, workers=args.workers)
            difference = np.abs(actual - expected)
            max_error = float(np.max(difference))
            relative_l2 = float(np.linalg.norm((actual - expected).ravel()) / np.linalg.norm(expected.ravel()))
            tolerance = 3e-6 if dtype is np.float32 else 3e-13
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance * max(1, float(np.max(np.abs(expected)))))
            rust_time = timing(rust_function, x, args.workers, args.repeats)
            scipy_time = timing(scipy_function, x, args.workers, args.repeats)
            record = dict(transform=name, shape=list(shape), dtype=str(x.dtype), workers=args.workers,
                          rust_seconds=rust_time, scipy_seconds=scipy_time, speedup=scipy_time/rust_time,
                          max_absolute_error=max_error, relative_l2_error=relative_l2)
            records.append(record)
            label = f"{name} {shape} {x.dtype}"
            print(f"{label:44} {rust_time*1e6:11.2f} {scipy_time*1e6:11.2f} {record['speedup']:10.2f}x")
    cpu = platform.processor()
    try:
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), cpu)
    except OSError:
        pass
    result = dict(python=platform.python_version(), platform=platform.platform(), cpu=cpu, cpu_count=os.cpu_count(),
                  numpy=np.__version__, scipy=scipy.__version__, fftlib=ft.__version__, records=records)
    with open(args.json_path, "w") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(f"Results: {args.json_path}. Values above 1x indicate Rust is faster.")


if __name__ == "__main__":
    main()
