//! Cached SIMD FFT plans for contiguous batches. Python support is optional.
//! Forward transforms use exp(-2πikn/N), matching scipy.fft.
use rayon::prelude::*;
use realfft::RealFftPlanner;
pub use rustfft::num_complex::Complex;
use rustfft::{FftDirection, FftNum, FftPlanner};
use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Clone, Copy, Debug)]
pub enum Norm {
    Backward,
    Forward,
    Ortho,
}

impl Norm {
    fn scale<T: FftNum>(self, n: usize, inverse: bool) -> T {
        T::from_f64(match self {
            Self::Ortho => 1.0 / (n as f64).sqrt(),
            Self::Backward if inverse => 1.0 / n as f64,
            Self::Forward if !inverse => 1.0 / n as f64,
            _ => 1.0,
        })
        .unwrap()
    }
}

type Pools = Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>;
static POOLS: OnceLock<Pools> = OnceLock::new();

fn pool(workers: usize) -> Result<Arc<rayon::ThreadPool>, String> {
    let mut pools = POOLS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    if let Some(pool) = pools.get(&workers) {
        return Ok(Arc::clone(pool));
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .map_err(|e| e.to_string())?,
    );
    pools.insert(workers, Arc::clone(&pool));
    Ok(pool)
}

// Workers initialize their own output pages before the FFT, instead of forcing
// the calling thread to touch the entire allocation first.
fn initialize<T: Copy>(slots: &mut [MaybeUninit<T>], value: T) -> &mut [T] {
    for slot in slots.iter_mut() {
        slot.write(value);
    }
    // SAFETY: every slot above contains a valid T; MaybeUninit has T's layout.
    // The returned exclusive borrow is tied to the same slots' lifetime.
    unsafe { std::slice::from_raw_parts_mut(slots.as_mut_ptr().cast(), slots.len()) }
}

/// Copy large Python inputs in parallel while the caller still holds the GIL.
/// All FFT work subsequently operates on owned Rust memory.
pub fn copy_input<T: Copy + Send + Sync>(input: &[T], workers: usize) -> Result<Vec<T>, String> {
    if workers == 0 {
        return Err("workers must be positive".into());
    }
    if workers == 1 || std::mem::size_of_val(input) < 2 * 1024 * 1024 {
        return Ok(input.to_vec());
    }
    let mut output = Vec::with_capacity(input.len());
    let chunk = input.len().div_ceil(workers).div_ceil(64) * 64;
    pool(workers)?.install(|| {
        output.spare_capacity_mut()[..input.len()]
            .par_chunks_mut(chunk)
            .zip(input.par_chunks(chunk))
            .for_each(|(dest, source)| {
                for (dest, source) in dest.iter_mut().zip(source) {
                    dest.write(*source);
                }
            });
    });
    // SAFETY: chunks cover exactly input.len() slots; every slot was written.
    unsafe {
        output.set_len(input.len());
    }
    Ok(output)
}

/// Reuse an engine to amortize FFT planning. Methods are safe to call concurrently.
pub struct Engine<T: FftNum> {
    complex: Mutex<FftPlanner<T>>,
    real: Mutex<RealFftPlanner<T>>,
    real_workspaces: Mutex<Vec<Vec<T>>>,
}

/// Owned row-major workspace. `order[i]` identifies the original axis now at i.
/// A consumer can restore logical axis order using a view, without copying data.
pub struct Array<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    pub order: Vec<usize>,
}

const TRANSPOSE_COLUMNS: usize = 8;

fn transpose_block<T: Copy>(
    input: &[T],
    output: &mut [T],
    block_index: usize,
    n: usize,
    inner: usize,
) {
    let first = block_index * TRANSPOSE_COLUMNS;
    let columns = output.len() / n;
    let mut offsets = [0; TRANSPOSE_COLUMNS];
    for (column, offset) in offsets[..columns].iter_mut().enumerate() {
        let lane = first + column;
        *offset = (lane / inner) * n * inner + lane % inner;
    }
    for row in 0..n {
        for column in 0..columns {
            output[column * n + row] = input[offsets[column] + row * inner];
        }
    }
}

impl<T: Copy + Send + Sync> Array<T> {
    fn new(data: Vec<T>, shape: &[usize], axes: &[usize], workers: usize) -> Result<Self, String> {
        let size = shape
            .iter()
            .try_fold(1usize, |size, &n| size.checked_mul(n))
            .ok_or("array shape is too large")?;
        if size != data.len() || workers == 0 {
            return Err("array shape or worker count is invalid".into());
        }
        let mut seen = vec![false; shape.len()];
        for &axis in axes {
            if axis >= shape.len() || seen[axis] || shape[axis] == 0 {
                return Err("FFT axes must be unique, in bounds, and nonempty".into());
            }
            seen[axis] = true;
        }
        Ok(Self {
            data,
            shape: shape.to_vec(),
            order: (0..shape.len()).collect(),
        })
    }

    // Transpose [outer, axis, inner] to [outer, inner, axis]. Small column blocks
    // keep adjacent source reads together, avoiding a full Python axis copy.
    fn move_last(&mut self, original_axis: usize, workers: usize, zero: T) -> Result<(), String> {
        let axis = self
            .order
            .iter()
            .position(|&a| a == original_axis)
            .ok_or("invalid axis")?;
        if axis + 1 == self.shape.len() {
            return Ok(());
        }
        let n = self.shape[axis];
        let inner = self.shape[axis + 1..].iter().product::<usize>();
        if !self.data.is_empty() && inner != 1 && n != 1 {
            let mut output = vec![zero; self.data.len()];
            let block_size = n * TRANSPOSE_COLUMNS.min(self.data.len() / n);
            let transpose = |block_index: usize, block: &mut [T]| {
                transpose_block(&self.data, block, block_index, n, inner);
            };
            if workers > 1 && self.data.len() >= 32768 {
                pool(workers)?.install(|| {
                    output
                        .par_chunks_mut(block_size)
                        .enumerate()
                        .for_each(|(index, block)| transpose(index, block));
                });
            } else {
                for (index, block) in output.chunks_mut(block_size).enumerate() {
                    transpose(index, block);
                }
            }
            self.data = output;
        }
        self.rotate_metadata(axis);
        Ok(())
    }

    fn rotate_metadata(&mut self, axis: usize) {
        let n = self.shape.remove(axis);
        self.shape.push(n);
        let axis = self.order.remove(axis);
        self.order.push(axis);
    }
}

impl<T: FftNum> Default for Engine<T> {
    fn default() -> Self {
        Self {
            complex: Mutex::new(FftPlanner::new()),
            real: Mutex::new(RealFftPlanner::new()),
            real_workspaces: Mutex::new(Vec::new()),
        }
    }
}

fn validate(len: usize, width: usize, workers: usize) -> Result<(), String> {
    if width == 0 || !len.is_multiple_of(width) {
        return Err("expected a positive transform length and complete rows".into());
    }
    if workers == 0 {
        return Err("workers must be positive".into());
    }
    Ok(())
}

fn parallel(len: usize, width: usize, workers: usize) -> bool {
    workers > 1 && len / width > 1 && len >= 32768
}

impl<T: FftNum> Engine<T> {
    pub fn clear_cache(&self) {
        *self.complex.lock().unwrap() = FftPlanner::new();
        *self.real.lock().unwrap() = RealFftPlanner::new();
        self.real_workspaces.lock().unwrap().clear();
    }

    /// Reuse a bounded real input buffer; returned memory is exclusively owned.
    pub fn copy_real_input(&self, input: &[T], workers: usize) -> Result<Vec<T>, String> {
        if workers == 0 {
            return Err("workers must be positive".into());
        }
        if !(2 * 1024 * 1024..=16 * 1024 * 1024).contains(&std::mem::size_of_val(input)) {
            return copy_input(input, workers);
        }
        let cached = {
            let mut buffers = self.real_workspaces.lock().unwrap();
            buffers
                .iter()
                .position(|buffer| buffer.len() == input.len())
                .map(|index| buffers.swap_remove(index))
        };
        let Some(mut output) = cached else {
            return copy_input(input, workers);
        };
        if workers > 1 && std::mem::size_of_val(input) >= 2 * 1024 * 1024 {
            let chunk = input.len().div_ceil(workers).div_ceil(64) * 64;
            pool(workers)?.install(|| {
                output
                    .par_chunks_mut(chunk)
                    .zip(input.par_chunks(chunk))
                    .for_each(|(dest, source)| dest.copy_from_slice(source));
            });
        } else {
            output.copy_from_slice(input);
        }
        Ok(output)
    }

    /// Store at most two 2–16 MiB buffers. Small and oversized buffers are freed.
    pub fn recycle_real_input(&self, data: Vec<T>) {
        let bytes = data.capacity().saturating_mul(std::mem::size_of::<T>());
        if (2 * 1024 * 1024..=16 * 1024 * 1024).contains(&bytes) {
            let mut buffers = self.real_workspaces.lock().unwrap();
            if buffers.len() < 2 {
                buffers.push(data);
            }
        }
    }

    fn complex_axes(
        &self,
        array: &mut Array<Complex<T>>,
        axes: &[usize],
        inverse: bool,
        norm: Norm,
        workers: usize,
    ) -> Result<(), String> {
        for &original_axis in axes {
            let axis = array
                .order
                .iter()
                .position(|&a| a == original_axis)
                .ok_or("invalid FFT axis")?;
            let n = array.shape[axis];
            if axis + 1 == array.shape.len() {
                self.c2c(&mut array.data, n, inverse, norm, workers)?;
                continue;
            }
            let inner = array.shape[axis + 1..].iter().product::<usize>();
            if inner == 1 || n == 1 {
                self.c2c(&mut array.data, n, inverse, norm, workers)?;
                array.rotate_metadata(axis);
                continue;
            }
            if !array.data.is_empty() {
                let direction = if inverse {
                    FftDirection::Inverse
                } else {
                    FftDirection::Forward
                };
                let fft = self.complex.lock().unwrap().plan_fft(n, direction);
                let scale = norm.scale::<T>(n, inverse);
                let zero = Complex::new(T::zero(), T::zero());
                let threaded = workers > 1 && array.data.len() >= 32768;
                let output_len = array.data.len();
                let mut output = if threaded {
                    Vec::with_capacity(output_len)
                } else {
                    vec![zero; output_len]
                };
                let block_size = n * TRANSPOSE_COLUMNS.min(array.data.len() / n);
                let workspace = || {
                    (
                        vec![zero; block_size],
                        vec![zero; fft.get_outofplace_scratch_len()],
                    )
                };
                let process = |buffers: &mut (Vec<Complex<T>>, Vec<Complex<T>>),
                               index: usize,
                               block: &mut [Complex<T>]| {
                    let (input, scratch) = buffers;
                    let input = &mut input[..block.len()];
                    transpose_block(&array.data, input, index, n, inner);
                    // Gather into a small reusable workspace and transform directly
                    // into the final axis layout, eliminating an intermediate write.
                    fft.process_outofplace_with_scratch(input, block, scratch);
                    if scale != T::one() {
                        block.iter_mut().for_each(|v| *v = *v * scale);
                    }
                };
                if threaded {
                    let min_blocks = (array.data.len() / block_size).div_ceil(workers).max(1);
                    pool(workers)?.install(|| {
                        output.spare_capacity_mut()[..output_len]
                            .par_chunks_mut(block_size)
                            .enumerate()
                            .with_min_len(min_blocks)
                            .for_each_init(workspace, |buffers, (index, block)| {
                                process(buffers, index, initialize(block, zero))
                            });
                    });
                    // SAFETY: each output block was initialized before processing.
                    unsafe {
                        output.set_len(output_len);
                    }
                } else {
                    let mut buffers = workspace();
                    for (index, block) in output.chunks_mut(block_size).enumerate() {
                        process(&mut buffers, index, block);
                    }
                }
                array.data = output;
            }
            array.rotate_metadata(axis);
        }
        Ok(())
    }

    /// N-D complex FFT with ownership retained across all axis passes.
    /// Shape changes (cropping/padding) are applied by the caller before entry.
    pub fn c2c_nd(
        &self,
        data: Vec<Complex<T>>,
        shape: &[usize],
        axes: &[usize],
        inverse: bool,
        norm: Norm,
        workers: usize,
    ) -> Result<Array<Complex<T>>, String> {
        let mut array = Array::new(data, shape, axes, workers)?;
        let sequence: Vec<_> = axes.iter().rev().copied().collect();
        self.complex_axes(&mut array, &sequence, inverse, norm, workers)?;
        Ok(array)
    }

    /// N-D real forward FFT. The last requested axis is reduced to n/2+1 bins.
    pub fn r2c_nd(
        &self,
        data: Vec<T>,
        shape: &[usize],
        axes: &[usize],
        norm: Norm,
        workers: usize,
    ) -> Result<Array<Complex<T>>, String> {
        let &real_axis = axes.last().ok_or("a real FFT axis is required")?;
        let mut input = Array::new(data, shape, axes, workers)?;
        input.move_last(real_axis, workers, T::zero())?;
        let n = *input.shape.last().unwrap();
        let data = self.r2c(&mut input.data, n, norm, workers)?;
        self.recycle_real_input(input.data);
        *input.shape.last_mut().unwrap() = n / 2 + 1;
        let mut output = Array {
            data,
            shape: input.shape,
            order: input.order,
        };
        let sequence: Vec<_> = axes[..axes.len() - 1].iter().rev().copied().collect();
        self.complex_axes(&mut output, &sequence, false, norm, workers)?;
        Ok(output)
    }

    /// N-D real inverse FFT. n is the full output length of the last FFT axis.
    pub fn c2r_nd(
        &self,
        data: Vec<Complex<T>>,
        shape: &[usize],
        axes: &[usize],
        n: usize,
        norm: Norm,
        workers: usize,
    ) -> Result<Array<T>, String> {
        let &real_axis = axes.last().ok_or("a real FFT axis is required")?;
        let mut input = Array::new(data, shape, axes, workers)?;
        if n == 0 || shape[real_axis] != n / 2 + 1 {
            return Err("invalid inverse real FFT length".into());
        }
        self.complex_axes(&mut input, &axes[..axes.len() - 1], true, norm, workers)?;
        input.move_last(real_axis, workers, Complex::new(T::zero(), T::zero()))?;
        let data = self.c2r(&mut input.data, n, norm, workers)?;
        *input.shape.last_mut().unwrap() = n;
        Ok(Array {
            data,
            shape: input.shape,
            order: input.order,
        })
    }

    /// In-place FFT of adjacent rows of length n. No per-row allocation.
    pub fn c2c(
        &self,
        data: &mut [Complex<T>],
        n: usize,
        inverse: bool,
        norm: Norm,
        workers: usize,
    ) -> Result<(), String> {
        validate(data.len(), n, workers)?;
        if data.is_empty() {
            return Ok(());
        }
        let direction = if inverse {
            FftDirection::Inverse
        } else {
            FftDirection::Forward
        };
        let fft = self.complex.lock().unwrap().plan_fft(n, direction);
        let scale = norm.scale::<T>(n, inverse);
        let process = |block: &mut [Complex<T>], scratch: &mut Vec<Complex<T>>| {
            fft.process_with_scratch(block, scratch);
            if scale != T::one() {
                block.iter_mut().for_each(|v| *v = *v * scale);
            }
        };
        if parallel(data.len(), n, workers) {
            let rows_per_block = (data.len() / n).div_ceil(workers);
            pool(workers)?.install(|| {
                data.par_chunks_mut(rows_per_block * n).for_each_init(
                    || vec![Complex::new(T::zero(), T::zero()); fft.get_inplace_scratch_len()],
                    |scratch, block| process(block, scratch),
                );
            });
        } else {
            let mut scratch =
                vec![Complex::new(T::zero(), T::zero()); fft.get_inplace_scratch_len()];
            process(data, &mut scratch);
        }
        Ok(())
    }

    /// Real FFT of adjacent rows. Input is workspace and may be overwritten.
    pub fn r2c(
        &self,
        data: &mut [T],
        n: usize,
        norm: Norm,
        workers: usize,
    ) -> Result<Vec<Complex<T>>, String> {
        validate(data.len(), n, workers)?;
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let fft = self.real.lock().unwrap().plan_fft_forward(n);
        let width = n / 2 + 1;
        let zero = Complex::new(T::zero(), T::zero());
        let threaded = parallel(data.len(), n, workers);
        let output_len = (data.len() / n) * width;
        let mut output = if threaded {
            Vec::with_capacity(output_len)
        } else {
            vec![zero; output_len]
        };
        let scale = norm.scale::<T>(n, false);
        let process =
            |input: &mut [T], output: &mut [Complex<T>], scratch: &mut Vec<Complex<T>>| {
                fft.process_with_scratch(input, output, scratch)
                    .expect("validated row lengths");
                if scale != T::one() {
                    output.iter_mut().for_each(|v| *v = *v * scale);
                }
            };
        if threaded {
            let rows_per_block = (data.len() / n).div_ceil(workers);
            pool(workers)?.install(|| {
                data.par_chunks_mut(n * rows_per_block)
                    .zip(
                        output.spare_capacity_mut()[..output_len]
                            .par_chunks_mut(width * rows_per_block),
                    )
                    .for_each_init(
                        || fft.make_scratch_vec(),
                        |scratch, (input, output)| {
                            let output = initialize(output, zero);
                            for (input, output) in input.chunks_mut(n).zip(output.chunks_mut(width))
                            {
                                process(input, output, scratch);
                            }
                        },
                    );
            });
            // SAFETY: each output block was initialized before FFT execution.
            unsafe {
                output.set_len(output_len);
            }
        } else {
            let mut scratch = fft.make_scratch_vec();
            for (input, output) in data.chunks_mut(n).zip(output.chunks_mut(width)) {
                process(input, output, &mut scratch);
            }
        }
        Ok(output)
    }

    /// Inverse real FFT; imaginary DC and Nyquist parts are ignored, as in SciPy.
    pub fn c2r(
        &self,
        data: &mut [Complex<T>],
        n: usize,
        norm: Norm,
        workers: usize,
    ) -> Result<Vec<T>, String> {
        if n == 0 {
            return Err("transform length must be positive".into());
        }
        let width = n / 2 + 1;
        validate(data.len(), width, workers)?;
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let fft = self.real.lock().unwrap().plan_fft_inverse(n);
        let threaded = parallel(data.len(), width, workers);
        let output_len = (data.len() / width) * n;
        let mut output = if threaded {
            Vec::with_capacity(output_len)
        } else {
            vec![T::zero(); output_len]
        };
        let scale = norm.scale::<T>(n, true);
        let process =
            |input: &mut [Complex<T>], output: &mut [T], scratch: &mut Vec<Complex<T>>| {
                input[0].im = T::zero();
                if n.is_multiple_of(2) {
                    input[width - 1].im = T::zero();
                }
                fft.process_with_scratch(input, output, scratch)
                    .expect("validated Hermitian row");
                if scale != T::one() {
                    output.iter_mut().for_each(|v| *v = *v * scale);
                }
            };
        if threaded {
            let rows_per_block = (data.len() / width).div_ceil(workers);
            pool(workers)?.install(|| {
                data.par_chunks_mut(width * rows_per_block)
                    .zip(
                        output.spare_capacity_mut()[..output_len]
                            .par_chunks_mut(n * rows_per_block),
                    )
                    .for_each_init(
                        || fft.make_scratch_vec(),
                        |scratch, (input, output)| {
                            let output = initialize(output, T::zero());
                            for (input, output) in input.chunks_mut(width).zip(output.chunks_mut(n))
                            {
                                process(input, output, scratch);
                            }
                        },
                    );
            });
            // SAFETY: all complete output rows were initialized by their workers.
            unsafe {
                output.set_len(output_len);
            }
        } else {
            let mut scratch = fft.make_scratch_vec();
            for (input, output) in data.chunks_mut(width).zip(output.chunks_mut(n)) {
                process(input, output, &mut scratch);
            }
        }
        Ok(output)
    }
}

#[cfg(feature = "python")]
mod python;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_copy_initializes_every_element() {
        let data: Vec<_> = (0..300_007)
            .map(|i| Complex::new(i as f64, -(i as f64)))
            .collect();
        for workers in [1, 2, 4, 7] {
            assert_eq!(copy_input(&data, workers).unwrap(), data);
        }
        assert!(copy_input(&data, 0).is_err());
        assert!(copy_input::<f64>(&[], 4).unwrap().is_empty());
    }

    #[test]
    fn real_workspace_cache_is_bounded_and_refilled() {
        let engine = Engine::<f64>::default();
        let input: Vec<_> = (0..300_007).map(|i| i as f64).collect();
        for _ in 0..4 {
            engine.recycle_real_input(vec![-777.0; input.len()]);
        }
        assert_eq!(engine.real_workspaces.lock().unwrap().len(), 2);
        for workers in [1, 4, 7] {
            let data = engine.copy_real_input(&input, workers).unwrap();
            assert_eq!(data, input);
            engine.recycle_real_input(data);
        }
        engine.clear_cache();
        assert!(engine.real_workspaces.lock().unwrap().is_empty());
    }

    #[test]
    fn complex_matches_direct_dft() {
        let engine = Engine::<f64>::default();
        for n in [1, 2, 3, 7, 16, 31, 128] {
            let input: Vec<_> = (0..n)
                .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
                .collect();
            let expected: Vec<_> = (0..n)
                .map(|k| {
                    input
                        .iter()
                        .enumerate()
                        .map(|(j, v)| {
                            *v * Complex::from_polar(
                                1.0,
                                -std::f64::consts::TAU * (j * k) as f64 / n as f64,
                            )
                        })
                        .sum::<Complex<f64>>()
                })
                .collect();
            let mut actual = input.clone();
            engine
                .c2c(&mut actual, n, false, Norm::Backward, 1)
                .unwrap();
            for (a, e) in actual.iter().zip(expected) {
                assert!((*a - e).norm() < 1e-10);
            }
            engine.c2c(&mut actual, n, true, Norm::Backward, 1).unwrap();
            for (a, e) in actual.iter().zip(input) {
                assert!((*a - e).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn real_roundtrip_and_parallel_batches() {
        let engine = Engine::<f64>::default();
        for n in [1, 2, 7, 16, 257] {
            let input: Vec<_> = (0..n * 256).map(|i| (i as f64).sin()).collect();
            for norm in [Norm::Backward, Norm::Forward, Norm::Ortho] {
                let mut data = input.clone();
                let mut spectrum = engine.r2c(&mut data, n, norm, 2).unwrap();
                let result = engine.c2r(&mut spectrum, n, norm, 2).unwrap();
                assert!(input.iter().zip(result).all(|(a, b)| (a - b).abs() < 1e-12));
            }
        }
        assert!(engine.c2c(&mut [], 0, false, Norm::Backward, 1).is_err());
        assert!(engine.r2c(&mut [1.0], 2, Norm::Backward, 1).is_err());
        assert!(engine.c2r(&mut [], 1, Norm::Backward, 0).is_err());
    }
}
