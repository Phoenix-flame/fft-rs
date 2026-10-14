//! Local diagnostic: isolate copying, zeroing, and real FFT execution.
use _native::{copy_input, Complex, Engine, Norm};
use std::hint::black_box;
use std::time::Instant;

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() {
    let n = 4096;
    let rows = 256;
    let original: Vec<f64> = (0..n * rows).map(|i| (i as f64).sin()).collect();
    let engine = Engine::<f64>::default();
    for workers in [1, 2, 4, 8] {
        let mut data = original.clone();
        black_box(engine.r2c(&mut data, n, Norm::Backward, workers).unwrap());
        let mut copies = Vec::new();
        let mut zeros = Vec::new();
        let mut ffts = Vec::new();
        for _ in 0..51 {
            let start = Instant::now();
            let input = black_box(copy_input(&original, workers).unwrap());
            copies.push(start.elapsed().as_secs_f64());
            drop(input);
            let start = Instant::now();
            let output = black_box(vec![Complex::<f64>::new(0., 0.); rows * (n / 2 + 1)]);
            zeros.push(start.elapsed().as_secs_f64());
            drop(output);
            data.copy_from_slice(&original);
            let start = Instant::now();
            let spectrum = black_box(engine.r2c(&mut data, n, Norm::Backward, workers).unwrap());
            ffts.push(start.elapsed().as_secs_f64());
            drop(spectrum);
        }
        println!(
            "workers {workers}: copy {:.1} µs, zero {:.1} µs, FFT including output {:.1} µs",
            median(copies) * 1e6,
            median(zeros) * 1e6,
            median(ffts) * 1e6
        );
    }
}
