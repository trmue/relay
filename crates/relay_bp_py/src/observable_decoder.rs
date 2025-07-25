// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::sync::Arc;

use crate::decoder::{get_sprs_bit_matrix_from_python, DecodeResult, DynDecoder};
use relay_bp::decoder::{Bit, Decoder as DecoderInner, DecoderRunner};
use relay_bp::observable_decoder::{
    ObservableDecodeResult as ObservableDecodeResultInner, ObservableDecoder,
    ObservableDecoderRunner as ObservableDecoderRunnerInner,
};

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::{Bound, PyResult};
use std::mem;

#[pyclass(module = "observable_decoder")]
pub struct ObservableDecodeResult {
    inner: ObservableDecodeResultInner,
}

impl ObservableDecodeResult {
    pub fn new(inner: ObservableDecodeResultInner) -> Self {
        ObservableDecodeResult { inner }
    }
}
#[pymethods]
impl ObservableDecodeResult {
    #[getter]
    pub fn observables<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Bit>> {
        PyArray1::from_array(py, &self.inner.observables)
    }

    #[getter]
    pub fn error_detected(&self) -> Option<bool> {
        let true_decoding = self.inner.true_decoding.as_ref()?;
        Some(true_decoding.error_detected)
    }

    #[getter]
    pub fn error_mismatch_detected(&self) -> Option<bool> {
        let true_decoding = self.inner.true_decoding.as_ref()?;
        Some(true_decoding.error_mismatch_detected)
    }

    #[getter]
    pub fn converged(&self) -> bool {
        self.inner.converged
    }

    #[getter]
    pub fn iterations(&self) -> usize {
        self.inner.iterations
    }

    #[getter]
    pub fn unconverged_no_error(&self) -> Option<bool> {
        let true_decoding = self.inner.true_decoding.as_ref()?;
        Some(true_decoding.unconverged_no_error)
    }

    #[getter]
    pub fn better_decoding_quality_error(&self) -> Option<bool> {
        let true_decoding = self.inner.true_decoding.as_ref()?;
        Some(true_decoding.better_decoding_quality_error)
    }

    #[getter]
    pub fn worse_decoding_quality_error(&self) -> Option<bool> {
        let true_decoding = self.inner.true_decoding.as_ref()?;
        Some(true_decoding.worse_decoding_quality_error)
    }

    #[getter]
    pub fn physical_decode_result(&self) -> Option<DecodeResult> {
        if let Some(result) = &self.inner.physical_decode_result {
            return Some(DecodeResult::new(result.clone()));
        }
        None
    }
}

#[pyclass(module = "observable_decoder")]
#[allow(dead_code)]
pub struct ObservableDecoderRunner {
    // Static lifetime to workaround lifetime issue referenced above.
    inner: ObservableDecoderRunnerInner<'static>,
}

#[pymethods]
impl ObservableDecoderRunner {
    #[new]
    #[pyo3(signature = (decoder, observable_error_matrix, include_decode_result=false))]
    pub fn new(
        py: Python<'_>,
        decoder: DynDecoder,
        observable_error_matrix: &Bound<'_, PyAny>,
        include_decode_result: bool,
    ) -> PyResult<Self> {
        let inner: relay_bp::observable_decoder::ObservableDecoderRunner<'_> = unsafe {
            mem::transmute(ObservableDecoderRunnerInner::new(
                decoder.0,
                Arc::new(get_sprs_bit_matrix_from_python(
                    py,
                    observable_error_matrix,
                )?),
                include_decode_result,
            ))
        };
        Ok(Self { inner })
    }

    pub fn decode<'py>(
        &mut self,
        py: Python<'py>,
        detectors: PyReadonlyArray1<'_, Bit>,
    ) -> Bound<'py, PyArray1<Bit>> {
        self.inner.decode(detectors.as_array()).into_pyarray(py)
    }

    pub fn decode_detailed(&mut self, detectors: PyReadonlyArray1<'_, Bit>) -> DecodeResult {
        DecodeResult::new(self.inner.decode_detailed(detectors.as_array()))
    }

    pub fn compute_observables<'py>(
        &mut self,
        py: Python<'py>,
        errors: PyReadonlyArray1<'_, Bit>,
    ) -> Bound<'py, PyArray1<Bit>> {
        PyArray1::from_array(py, &self.inner.compute_observables(errors.as_array()))
    }

    pub fn decode_observables<'py>(
        &mut self,
        py: Python<'py>,
        detectors: PyReadonlyArray1<'_, Bit>,
    ) -> Bound<'py, PyArray1<Bit>> {
        PyArray1::from_array(py, &self.inner.decode_observables(detectors.as_array()))
    }

    pub fn from_errors_decode_observables_detailed(
        &mut self,
        errors: PyReadonlyArray1<'_, Bit>,
    ) -> ObservableDecodeResult {
        ObservableDecodeResult::new(
            self.inner
                .from_errors_decode_observables_detailed(errors.as_array()),
        )
    }

    #[pyo3(signature = (detectors, parallel=false, progress_bar=true, leave_progress_bar_on_finish=false))]
    pub fn decode_batch<'py>(
        &mut self,
        py: Python<'py>,
        detectors: PyReadonlyArray2<'_, Bit>,
        parallel: bool,
        progress_bar: bool,
        leave_progress_bar_on_finish: bool,
    ) -> Bound<'py, PyArray2<Bit>> {
        let results = match (parallel, progress_bar) {
            (false, false) => self.inner.decode_batch(detectors.as_array()),
            (true, false) => self.inner.par_decode_batch(detectors.as_array()),
            (false, true) => self
                .inner
                .decode_batch_progress_bar(detectors.as_array(), leave_progress_bar_on_finish),
            (true, true) => self
                .inner
                .par_decode_batch_progress_bar(detectors.as_array(), leave_progress_bar_on_finish),
        };
        results.into_pyarray(py)
    }

    #[pyo3(signature = (detectors, parallel=false, progress_bar=true, leave_progress_bar_on_finish=false))]
    pub fn decode_detailed_batch(
        &mut self,
        detectors: PyReadonlyArray2<'_, Bit>,
        parallel: bool,
        progress_bar: bool,
        leave_progress_bar_on_finish: bool,
    ) -> Vec<DecodeResult> {
        let results = match (parallel, progress_bar) {
            (false, false) => self.inner.decode_detailed_batch(detectors.as_array()),
            (true, false) => self.inner.par_decode_detailed_batch(detectors.as_array()),
            (false, true) => self.inner.decode_detailed_batch_progress_bar(
                detectors.as_array(),
                leave_progress_bar_on_finish,
            ),
            (true, true) => self.inner.par_decode_detailed_batch_progress_bar(
                detectors.as_array(),
                leave_progress_bar_on_finish,
            ),
        };
        results.into_iter().map(DecodeResult::new).collect()
    }

    #[pyo3(signature = (detectors, parallel=false, progress_bar=true, leave_progress_bar_on_finish=false))]
    pub fn decode_observables_batch<'py>(
        &mut self,
        py: Python<'py>,
        detectors: PyReadonlyArray2<'_, Bit>,
        parallel: bool,
        progress_bar: bool,
        leave_progress_bar_on_finish: bool,
    ) -> Bound<'py, PyArray2<Bit>> {
        let results = match (parallel, progress_bar) {
            (false, false) => self.inner.decode_observables_batch(detectors.as_array()),
            (true, false) => self
                .inner
                .par_decode_observables_batch(detectors.as_array()),
            (false, true) => self.inner.decode_observables_batch_progress_bar(
                detectors.as_array(),
                leave_progress_bar_on_finish,
            ),
            (true, true) => self.inner.par_decode_observables_batch_progress_bar(
                detectors.as_array(),
                leave_progress_bar_on_finish,
            ),
        };
        results.into_pyarray(py)
    }

    #[pyo3(signature = (errors, parallel=false, progress_bar=true, leave_progress_bar_on_finish=false))]
    pub fn from_errors_decode_observables_batch<'py>(
        &mut self,
        py: Python<'py>,
        errors: PyReadonlyArray2<'_, Bit>,
        parallel: bool,
        progress_bar: bool,
        leave_progress_bar_on_finish: bool,
    ) -> Bound<'py, PyArray2<Bit>> {
        let results = match (parallel, progress_bar) {
            (false, false) => self
                .inner
                .from_errors_decode_observables_batch(errors.as_array()),
            (true, false) => self
                .inner
                .par_from_errors_decode_observables_batch(errors.as_array()),
            (false, true) => self
                .inner
                .from_errors_decode_observables_batch_progress_bar(
                    errors.as_array(),
                    leave_progress_bar_on_finish,
                ),
            (true, true) => self
                .inner
                .par_from_errors_decode_observables_batch_progress_bar(
                    errors.as_array(),
                    leave_progress_bar_on_finish,
                ),
        };

        results.into_pyarray(py)
    }

    #[pyo3(signature = (detectors, parallel=false, progress_bar=true, leave_progress_bar_on_finish=false))]
    pub fn decode_observables_detailed_batch(
        &mut self,
        detectors: PyReadonlyArray2<'_, Bit>,
        parallel: bool,
        progress_bar: bool,
        leave_progress_bar_on_finish: bool,
    ) -> Vec<ObservableDecodeResult> {
        let results = match (parallel, progress_bar) {
            (false, false) => self
                .inner
                .decode_observables_detailed_batch(detectors.as_array()),
            (true, false) => self
                .inner
                .par_decode_observables_detailed_batch(detectors.as_array()),
            (false, true) => self.inner.decode_observables_detailed_batch_progress_bar(
                detectors.as_array(),
                leave_progress_bar_on_finish,
            ),
            (true, true) => self
                .inner
                .par_decode_observables_detailed_batch_progress_bar(
                    detectors.as_array(),
                    leave_progress_bar_on_finish,
                ),
        };
        results
            .into_iter()
            .map(ObservableDecodeResult::new)
            .collect()
    }

    #[pyo3(signature = (errors, parallel=false, progress_bar=true, leave_progress_bar_on_finish=false))]
    pub fn from_errors_decode_observables_detailed_batch(
        &mut self,
        errors: PyReadonlyArray2<'_, Bit>,
        parallel: bool,
        progress_bar: bool,
        leave_progress_bar_on_finish: bool,
    ) -> Vec<ObservableDecodeResult> {
        let results = match (parallel, progress_bar) {
            (false, false) => self
                .inner
                .from_errors_decode_observables_detailed_batch(errors.as_array()),
            (true, false) => self
                .inner
                .par_from_errors_decode_observables_detailed_batch(errors.as_array()),
            (false, true) => self
                .inner
                .from_errors_decode_observables_detailed_batch_progress_bar(
                    errors.as_array(),
                    leave_progress_bar_on_finish,
                ),
            (true, true) => self
                .inner
                .par_from_errors_decode_observables_detailed_batch_progress_bar(
                    errors.as_array(),
                    leave_progress_bar_on_finish,
                ),
        };

        results
            .into_iter()
            .map(ObservableDecodeResult::new)
            .collect()
    }
}

/// A Python module implemented in Rust.
#[pymodule]
pub fn _observable_decoder<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<ObservableDecoderRunner>()?;
    m.add_class::<ObservableDecodeResult>()?;
    Ok(())
}

pub fn init_observable_decoder<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    // Workaround for https://github.com/PyO3/pyo3/issues/759
    let decoder_module = PyModule::new(_py, "_relay_bp._observable_decoder")?;

    _observable_decoder(_py, &decoder_module)?;

    m.add("_observable_decoder", &decoder_module)?;
    decoder_module.setattr("__name__", "_observable_decoder")?;
    _py.import("sys")?
        .getattr("modules")?
        .set_item("_relay_bp._observable_decoder", &decoder_module)?;
    Ok(())
}
