use crate::{Complex, Engine, Norm};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::OnceLock;

static F32: OnceLock<Engine<f32>> = OnceLock::new();
static F64: OnceLock<Engine<f64>> = OnceLock::new();

fn norm(value: &str) -> PyResult<Norm> {
    match value {
        "backward" => Ok(Norm::Backward),
        "forward" => Ok(Norm::Forward),
        "ortho" => Ok(Norm::Ortho),
        _ => Err(PyValueError::new_err("invalid normalization")),
    }
}

// Copy NumPy input while holding the GIL; computation then owns its workspace.
// Returning a Vec transfers output ownership to NumPy without another copy.
macro_rules! bindings {
    ($c2c:ident, $r2c:ident, $c2r:ident, $ndc:ident, $ndr:ident, $ndi:ident, $t:ty, $engine:ident) => {
        #[pyfunction]
        fn $c2c<'py>(
            py: Python<'py>,
            x: PyReadonlyArray1<'py, Complex<$t>>,
            n: usize,
            inverse: bool,
            normalization: &str,
            workers: usize,
        ) -> PyResult<Bound<'py, PyArray1<Complex<$t>>>> {
            let mut data = x.as_slice()?.to_vec();
            let normalization = norm(normalization)?;
            py.detach(|| {
                $engine.get_or_init(Engine::default).c2c(
                    &mut data,
                    n,
                    inverse,
                    normalization,
                    workers,
                )
            })
            .map_err(PyValueError::new_err)?;
            Ok(data.into_pyarray(py))
        }
        #[pyfunction]
        fn $r2c<'py>(
            py: Python<'py>,
            x: PyReadonlyArray1<'py, $t>,
            n: usize,
            normalization: &str,
            workers: usize,
        ) -> PyResult<Bound<'py, PyArray1<Complex<$t>>>> {
            let mut data = x.as_slice()?.to_vec();
            let normalization = norm(normalization)?;
            let output = py
                .detach(|| {
                    $engine
                        .get_or_init(Engine::default)
                        .r2c(&mut data, n, normalization, workers)
                })
                .map_err(PyValueError::new_err)?;
            Ok(output.into_pyarray(py))
        }
        #[pyfunction]
        fn $c2r<'py>(
            py: Python<'py>,
            x: PyReadonlyArray1<'py, Complex<$t>>,
            n: usize,
            normalization: &str,
            workers: usize,
        ) -> PyResult<Bound<'py, PyArray1<$t>>> {
            let mut data = x.as_slice()?.to_vec();
            let normalization = norm(normalization)?;
            let output = py
                .detach(|| {
                    $engine
                        .get_or_init(Engine::default)
                        .c2r(&mut data, n, normalization, workers)
                })
                .map_err(PyValueError::new_err)?;
            Ok(output.into_pyarray(py))
        }
        #[pyfunction]
        fn $ndc<'py>(
            py: Python<'py>,
            x: PyReadonlyArray1<'py, Complex<$t>>,
            shape: Vec<usize>,
            axes: Vec<usize>,
            inverse: bool,
            normalization: &str,
            workers: usize,
        ) -> PyResult<(Bound<'py, PyArray1<Complex<$t>>>, Vec<usize>, Vec<usize>)> {
            let data = x.as_slice()?.to_vec();
            let normalization = norm(normalization)?;
            let output = py
                .detach(|| {
                    $engine.get_or_init(Engine::default).c2c_nd(
                        data,
                        &shape,
                        &axes,
                        inverse,
                        normalization,
                        workers,
                    )
                })
                .map_err(PyValueError::new_err)?;
            Ok((output.data.into_pyarray(py), output.shape, output.order))
        }
        #[pyfunction]
        fn $ndr<'py>(
            py: Python<'py>,
            x: PyReadonlyArray1<'py, $t>,
            shape: Vec<usize>,
            axes: Vec<usize>,
            normalization: &str,
            workers: usize,
        ) -> PyResult<(Bound<'py, PyArray1<Complex<$t>>>, Vec<usize>, Vec<usize>)> {
            let data = x.as_slice()?.to_vec();
            let normalization = norm(normalization)?;
            let output = py
                .detach(|| {
                    $engine.get_or_init(Engine::default).r2c_nd(
                        data,
                        &shape,
                        &axes,
                        normalization,
                        workers,
                    )
                })
                .map_err(PyValueError::new_err)?;
            Ok((output.data.into_pyarray(py), output.shape, output.order))
        }
        #[pyfunction]
        fn $ndi<'py>(
            py: Python<'py>,
            x: PyReadonlyArray1<'py, Complex<$t>>,
            shape: Vec<usize>,
            axes: Vec<usize>,
            n: usize,
            normalization: &str,
            workers: usize,
        ) -> PyResult<(Bound<'py, PyArray1<$t>>, Vec<usize>, Vec<usize>)> {
            let data = x.as_slice()?.to_vec();
            let normalization = norm(normalization)?;
            let output = py
                .detach(|| {
                    $engine.get_or_init(Engine::default).c2r_nd(
                        data,
                        &shape,
                        &axes,
                        n,
                        normalization,
                        workers,
                    )
                })
                .map_err(PyValueError::new_err)?;
            Ok((output.data.into_pyarray(py), output.shape, output.order))
        }
    };
}
bindings!(c2c_f32, r2c_f32, c2r_f32, c2c_nd_f32, r2c_nd_f32, c2r_nd_f32, f32, F32);
bindings!(c2c_f64, r2c_f64, c2r_f64, c2c_nd_f64, r2c_nd_f64, c2r_nd_f64, f64, F64);

#[pyfunction]
fn clear_cache() {
    if let Some(engine) = F32.get() {
        engine.clear_cache();
    }
    if let Some(engine) = F64.get() {
        engine.clear_cache();
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(c2c_f32, m)?)?;
    m.add_function(wrap_pyfunction!(r2c_f32, m)?)?;
    m.add_function(wrap_pyfunction!(c2r_f32, m)?)?;
    m.add_function(wrap_pyfunction!(c2c_f64, m)?)?;
    m.add_function(wrap_pyfunction!(r2c_f64, m)?)?;
    m.add_function(wrap_pyfunction!(c2r_f64, m)?)?;
    m.add_function(wrap_pyfunction!(c2c_nd_f32, m)?)?;
    m.add_function(wrap_pyfunction!(r2c_nd_f32, m)?)?;
    m.add_function(wrap_pyfunction!(c2r_nd_f32, m)?)?;
    m.add_function(wrap_pyfunction!(c2c_nd_f64, m)?)?;
    m.add_function(wrap_pyfunction!(r2c_nd_f64, m)?)?;
    m.add_function(wrap_pyfunction!(c2r_nd_f64, m)?)?;
    m.add_function(wrap_pyfunction!(clear_cache, m)?)?;
    Ok(())
}
