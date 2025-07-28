// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;

use std::sync::Arc;

use crate::decoder::{get_sprs_bit_matrix_from_python, DecodeResult, DynDecoder};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use relay_bp::bp::min_sum::MinSumDecoderConfig;
use relay_bp::bp::min_sum_fixed::MinSumBPDecoderFixed as MinSumBPDecoderFixedInner;
use relay_bp::decoder::Bit;

#[pyclass(extends=DynDecoder, subclass, module = "bp")]
#[allow(dead_code)]
pub struct MinSumBPDecoderFixed {}

#[pymethods]
impl MinSumBPDecoderFixed {
    #[new]
    #[pyo3(signature = (check_matrix, error_priors, max_iter=200, alpha=None, alpha_iteration_scaling_factor=1.0, gamma0=None, data_scale_value=None, max_data_value=None, int_bits=None, frac_bits=None))]
    #[allow(clippy::missing_transmute_annotations, clippy::too_many_arguments)]
    pub fn new(
        py: Python<'_>,
        check_matrix: &Bound<'_, PyAny>,
        error_priors: &Bound<'_, PyArray1<f64>>,
        max_iter: usize,
        alpha: Option<f64>,
        alpha_iteration_scaling_factor: f64,
        gamma0: Option<f64>,
        data_scale_value: Option<f64>,
        max_data_value: Option<f64>,
        int_bits: Option<isize>,
        frac_bits: Option<isize>,
    ) -> PyResult<(Self, DynDecoder)> {
        let min_sum_decoder = Self {};

        let config = MinSumDecoderConfig {
            error_priors: unsafe { error_priors.as_array() }.to_owned(),
            max_iter,
            alpha,
            alpha_iteration_scaling_factor,
            gamma0,
            data_scale_value,
            max_data_value,
            int_bits,
            frac_bits,
        };

        let inner_decoder = MinSumBPDecoderFixedInner::new(
            Arc::new(get_sprs_bit_matrix_from_python(py, check_matrix)?),
            Arc::new(config),
        );

        let dyn_decoder = DynDecoder(Box::new(inner_decoder));
        Ok((min_sum_decoder, dyn_decoder))
    }

    pub fn decode<'py>(
        mut self_: PyRefMut<'_, Self>,
        py: Python<'py>,
        detectors: PyReadonlyArray1<'_, Bit>,
    ) -> Bound<'py, PyArray1<Bit>> {
        self_
            .as_super()
            .inner()
            .decode(detectors.as_array())
            .into_pyarray(py)
    }

    pub fn decode_detailed(
        mut self_: PyRefMut<'_, Self>,
        detectors: PyReadonlyArray1<'_, Bit>,
    ) -> DecodeResult {
        DecodeResult::new(
            self_
                .as_super()
                .inner()
                .decode_detailed(detectors.as_array()),
        )
    }

    pub fn decode_batch<'py>(
        mut self_: PyRefMut<'_, Self>,
        py: Python<'py>,
        detectors: PyReadonlyArray2<'_, Bit>,
    ) -> Bound<'py, PyArray2<Bit>> {
        self_
            .as_super()
            .inner()
            .decode_batch(detectors.as_array())
            .into_pyarray(py)
    }

    pub fn decode_detailed_batch(
        mut self_: PyRefMut<'_, Self>,
        detectors: PyReadonlyArray2<'_, Bit>,
    ) -> Vec<DecodeResult> {
        self_
            .as_super()
            .inner()
            .decode_detailed_batch(detectors.as_array())
            .into_iter()
            .map(DecodeResult::new)
            .collect()
    }
}
