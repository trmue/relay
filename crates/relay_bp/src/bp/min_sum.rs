// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::bipartite_graph::SparseBipartiteGraph;
use crate::decoder::{BPExtraResult, DecodeResult, Decoder, DecoderRunner};
use crate::decoder::{Bit, SparseBitMatrix};
use itertools::izip;
use log::debug;
use ndarray::{Array1, ArrayView1};
use num_traits::FromPrimitive;
use num_traits::{Bounded, Signed, ToPrimitive};
use sprs::CsMatView;
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct MinSumDecoderConfig {
    pub error_priors: Array1<f64>,
    pub max_iter: usize,
    pub alpha: Option<f64>,
    pub alpha_iteration_scaling_factor: f64,
    pub gamma0: Option<f64>,
    pub data_scale_value: Option<f64>,
    pub max_data_value: Option<f64>,
    pub int_bits: Option<isize>,
    pub frac_bits: Option<isize>,
}

impl Default for MinSumDecoderConfig {
    fn default() -> Self {
        Self {
            error_priors: Default::default(),
            max_iter: 200,
            alpha: None,
            alpha_iteration_scaling_factor: 1.,
            gamma0: None,
            data_scale_value: None,
            max_data_value: None,
            int_bits: None,
            frac_bits: None,
        }
    }
}

impl MinSumDecoderConfig {
    pub fn prior_ratios(&self) -> Array1<f64> {
        // A funky way of making (1-p)/p handle left to right type inference for arithmetic
        (1.0 - &self.error_priors) / &self.error_priors
    }

    pub fn log_prior_ratios(&self) -> Array1<f64> {
        self.prior_ratios().ln()
    }

    pub fn set_max_iter(&mut self, iterations: usize) {
        self.max_iter = iterations;
    }

    pub fn set_fixed(&mut self, int_bits: isize, frac_bits: isize) {
        self.int_bits = Some(int_bits);
        self.frac_bits = Some(frac_bits);
        self.max_data_value = Some((1 << (int_bits - 1)) as f64);
    }
}

/// A fast min-sum implementation of BP implemented internally
/// using a sparse bipartite graph.
#[derive(Clone)]
pub struct MinSumBPDecoder<N: PartialEq + Default + Clone + Copy> {
    check_matrix: Arc<SparseBitMatrix>,
    pub config: Arc<MinSumDecoderConfig>,
    log_prior_ratios: Array1<N>,
    check_to_variable: SparseBipartiteGraph<N>,
    variable_to_check: SparseBipartiteGraph<N>,
    // A cache of check to variable data mappings.
    check_to_variable_nnz_map: Vec<usize>,
    // A cache of variable to check data mappings.
    variable_to_check_nnz_map: Vec<usize>,
    posterior_ratios: Array1<N>,
    memory_strengths: Array1<N>,
    decoding: Array1<Bit>,
    max_data_value: Option<N>,
    data_scale_value: Option<N>,
    pub current_iteration: usize,
}

impl<N> MinSumBPDecoder<N>
where
    N: PartialEq
        + Debug
        + Default
        + Clone
        + Copy
        + Signed
        + Bounded
        + FromPrimitive
        + ToPrimitive
        + std::cmp::PartialOrd
        + std::ops::Add
        + std::ops::AddAssign
        + std::ops::DivAssign
        + std::ops::Mul<N>
        + std::ops::MulAssign
        + Send
        + Sync
        + std::fmt::Display
        + 'static,
{
    pub fn new(
        check_matrix: Arc<SparseBitMatrix>,
        config: Arc<MinSumDecoderConfig>,
    ) -> MinSumBPDecoder<N> {
        let check_to_variable = MinSumBPDecoder::build_check_to_variable(check_matrix.clone());
        let variable_to_check = MinSumBPDecoder::build_variable_to_check(check_matrix.clone());
        let (check_to_variable_nnz_map, variable_to_check_nnz_map) =
            MinSumBPDecoder::build_nnz_maps(check_to_variable.view(), variable_to_check.view());

        let max_data_value = match config.max_data_value {
            Some(val) => N::from_f64(val),
            None => None,
        };

        let data_scale_value = match config.data_scale_value {
            Some(val) => N::from_f64(val),
            None => None,
        };

        let log_prior_ratios = config.log_prior_ratios().mapv_into_any(|val| {
            let updated_val = match val {
                f64::INFINITY => N::max_value(),
                _ => {
                    // Apply optional data scaling
                    let prior = match config.data_scale_value {
                        Some(scale_value) => scale_value * val,
                        None => val,
                    };
                    N::from_f64(prior).unwrap()
                }
            };

            // Bound prior values if necessary
            match max_data_value {
                Some(max_val) => Self::bound_value_magnitude(updated_val, max_val),
                None => updated_val,
            }
        });

        // Initialize EWA and posteriors
        let ewa_factor_float = config.gamma0.unwrap_or(0.);
        let ewa_factor = N::from_f64(match config.data_scale_value {
            Some(scale_value) => scale_value * ewa_factor_float,
            None => ewa_factor_float,
        })
        .unwrap();

        let memory_strengths = Array1::from_elem(check_matrix.cols(), ewa_factor);

        let posterior_ratios = if config.gamma0.is_some() {
            log_prior_ratios.clone()
        } else {
            Array1::zeros(check_matrix.cols())
        };

        let decoding = Array1::zeros(check_matrix.cols());

        MinSumBPDecoder::<N> {
            check_matrix,
            config,
            log_prior_ratios,
            check_to_variable,
            variable_to_check,
            check_to_variable_nnz_map,
            variable_to_check_nnz_map,
            posterior_ratios,
            memory_strengths,
            decoding,
            max_data_value,
            data_scale_value,
            current_iteration: 0,
        }
    }

    pub fn set_log_prior_ratio(&mut self, mut log_prior_ratios: Array1<N>) {
        self.log_prior_ratios = match self.data_scale_value {
            Some(scale_val) => {
                log_prior_ratios.iter_mut().for_each(|v| *v *= scale_val);
                log_prior_ratios
            }
            None => log_prior_ratios,
        };
    }

    pub fn set_log_prior_ratio_f64(&mut self, log_prior_ratios: Array1<f64>) {
        self.log_prior_ratios = match self.config.data_scale_value {
            Some(scale_val) => {
                log_prior_ratios.mapv_into_any(|v| N::from_f64(scale_val * v).unwrap())
            }
            None => log_prior_ratios.mapv_into_any(|v| N::from_f64(v).unwrap()),
        };
    }

    /// Set external memory strengths from f64. Applies scaling if needed.
    pub fn set_memory_strengths_f64(&mut self, memory_strengths: Array1<f64>) {
        self.memory_strengths = match self.config.data_scale_value {
            Some(scale_val) => {
                memory_strengths.mapv_into_any(|v| N::from_f64(scale_val * v).unwrap())
            }
            None => memory_strengths.mapv_into_any(|v| N::from_f64(v).unwrap()),
        };
    }

    /// Set external memory strengths from N. Applies scaling if needed.
    pub fn set_memory_strengths(&mut self, mut memory_strengths: Array1<N>) {
        self.memory_strengths = match self.data_scale_value {
            Some(scale_val) => {
                memory_strengths.iter_mut().for_each(|v| *v *= scale_val);
                memory_strengths
            }
            None => memory_strengths,
        };
    }

    // Construct a new check message graph
    fn build_check_to_variable(check_matrix: Arc<SparseBitMatrix>) -> SparseBipartiteGraph<N> {
        let check_matrix_csc = check_matrix.to_csc();

        let default_messages: Vec<_> = vec![N::default(); check_matrix_csc.nnz()];

        SparseBipartiteGraph::new_csc(
            check_matrix_csc.shape(),
            check_matrix_csc.indptr().raw_storage().to_vec(),
            check_matrix_csc.indices().to_vec(),
            default_messages,
        )
    }

    // Construct a new variable message graph
    fn build_variable_to_check(check_matrix: Arc<SparseBitMatrix>) -> SparseBipartiteGraph<N> {
        let check_matrix_csr = check_matrix.to_csr();

        let default_messages: Vec<_> = vec![N::default(); check_matrix_csr.nnz()];

        SparseBipartiteGraph::new(
            check_matrix_csr.shape(),
            check_matrix_csr.indptr().raw_storage().to_vec(),
            check_matrix_csr.indices().to_vec(),
            default_messages,
        )
    }

    fn build_nnz_maps(
        check_to_variable: CsMatView<N>,
        variable_to_check: CsMatView<N>,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut check_to_variable_nnz_map: Vec<usize> = Vec::with_capacity(check_to_variable.nnz());
        for (_, (row, col)) in check_to_variable.view().iter() {
            check_to_variable_nnz_map.push(variable_to_check.nnz_index(row, col).unwrap().0);
        }
        let mut variable_to_check_nnz_map: Vec<usize> = Vec::with_capacity(variable_to_check.nnz());
        for (_, (row, col)) in variable_to_check.view().iter() {
            variable_to_check_nnz_map.push(check_to_variable.nnz_index(row, col).unwrap().0);
        }
        (check_to_variable_nnz_map, variable_to_check_nnz_map)
    }

    /// Initialize variable message state to the prior
    pub fn initialize_variable_to_check(&mut self) -> &mut SparseBipartiteGraph<N> {
        for mut row_vec in self.variable_to_check.outer_iterator_mut() {
            row_vec
                .iter_mut()
                .for_each(|(col_ind, val)| *val = self.log_prior_ratios[col_ind]);
        }
        &mut self.variable_to_check
    }

    fn alpha(&self) -> N {
        let mut alpha = match self.config.alpha {
            Some(0.) => {
                let iteration = (self.current_iteration + 1) as f64;
                1.0 - (2_f64).powf(-(iteration / self.config.alpha_iteration_scaling_factor))
            }
            Some(val) => val,
            None => 1.0,
        };

        // Handle case of alpha < 0 defaulting to 1.
        // This aligns with integer case of ldpc-simulation
        if alpha < 0. {
            alpha = 1.
        }
        // Scale if needed
        alpha = match self.config.data_scale_value {
            Some(scale_val) => scale_val * alpha,
            None => alpha,
        };

        N::from_f64(alpha).unwrap()
    }

    /// Compute check to bit message iteration
    fn compute_check_to_variable(
        &mut self,
        detectors: ArrayView1<Bit>,
    ) -> &mut SparseBipartiteGraph<N> {
        let alpha = self.alpha();

        for (var_check_row_ind, var_check_row_vec) in
            self.variable_to_check.outer_iterator().enumerate()
        {
            let row_sign = if detectors[var_check_row_ind] == 1 {
                N::one().neg()
            } else {
                N::one()
            };
            let mut accumulated_sign = row_sign.is_negative();
            let mut min_ind: usize = 0;
            // True min message
            let mut min_message = N::max_value();
            // Next lowest min message to be used for self-exlusive min value
            let mut second_min_message = N::max_value();

            for (var_check_col_ind, var_check_col_val) in var_check_row_vec.iter() {
                accumulated_sign ^= var_check_col_val.is_negative();
                let abs_msg = var_check_col_val.abs();
                if abs_msg <= min_message {
                    second_min_message = min_message;
                    min_message = abs_msg;
                    min_ind = var_check_col_ind;
                } else if abs_msg <= second_min_message {
                    second_min_message = abs_msg
                }
            }

            debug!("Variable messages for row {var_check_row_ind:?}: {var_check_row_vec:?}");

            // Iterate over the row's storage indices
            let data_range = self
                .variable_to_check
                .indptr()
                .outer_inds(var_check_row_ind);

            for (ind, var_check_col_ind, var_check_col_val) in izip!(
                data_range.clone(),
                &self.variable_to_check.indices()[data_range.clone()],
                &self.variable_to_check.data()[data_range.clone()]
            ) {
                // Extract the sign from the accumulated sign.
                let check_to_variable_sign = accumulated_sign ^ var_check_col_val.is_negative();
                let check_to_variable_min: N = if *var_check_col_ind != min_ind {
                    min_message
                } else {
                    second_min_message
                };
                // Copy the sign to the variable. check_to_variable_min is guranteed to be positive.
                let mut check_to_variable = alpha * check_to_variable_min;
                if check_to_variable_sign {
                    check_to_variable = check_to_variable.neg();
                }

                // We directly manipulate the indicies of the check_to_variable_matrix using
                // the cached value map to avoid the need for a logarithmic insert
                self.check_to_variable.data_mut()[self.variable_to_check_nnz_map[ind]] =
                    check_to_variable;
            }
        }

        if let Some(scale_val) = self.data_scale_value {
            self.check_to_variable /= scale_val
        }

        &mut self.check_to_variable
    }

    fn compute_variable_prior(&self, variable: usize) -> N {
        // Apply membp
        if self.config.gamma0.is_some() {
            if self.log_prior_ratios[variable] == N::max_value() {
                return self.log_prior_ratios[variable];
            }
            let scaled_one = self.data_scale_value.unwrap_or(N::one());
            // First divide through denominator before numerator to avoid overflow
            let prior_component = (self.log_prior_ratios[variable] / scaled_one)
                * (scaled_one - self.memory_strengths[variable]);
            let posterior_component =
                (self.posterior_ratios[variable] / scaled_one) * self.memory_strengths[variable];
            return prior_component + posterior_component;
        }
        self.log_prior_ratios[variable]
    }

    /// Compute bit to check message iteration
    fn compute_variable_to_check(&mut self) -> &mut SparseBipartiteGraph<N> {
        for (check_var_col_ind, check_var_col_vec) in
            self.check_to_variable.outer_iterator().enumerate()
        {
            // Accumulate messages
            let mut check_to_var_row_sum = self.compute_variable_prior(check_var_col_ind);

            debug!("Check messages for col {check_var_col_ind:?}: {check_var_col_vec:?}");

            let data_range = self
                .check_to_variable
                .indptr()
                .outer_inds(check_var_col_ind);

            // Perform iteration in the forward direction to accumulate left to right
            for (ind, check_var_row_val) in izip!(
                data_range.clone(),
                &self.check_to_variable.data()[data_range.clone()]
            ) {
                self.variable_to_check.data_mut()[self.check_to_variable_nnz_map[ind]] =
                    check_to_var_row_sum;
                check_to_var_row_sum += *check_var_row_val;
            }

            self.posterior_ratios[check_var_col_ind] = check_to_var_row_sum;

            // Now perform iteration in the reverse direction to accumulate right to left
            check_to_var_row_sum = N::zero();
            // Remove each messages contribution
            for (ind, check_var_row_val) in izip!(
                data_range.clone(),
                &self.check_to_variable.data()[data_range.clone()]
            )
            .rev()
            {
                let map_ind = self.check_to_variable_nnz_map[ind];
                self.variable_to_check.data_mut()[map_ind] += check_to_var_row_sum;
                check_to_var_row_sum += *check_var_row_val;

                // We directly manipulate the indicies of the variable_to_check matrix using
                // the cached value map to avoid the need for a logarithmic insert
                debug!(
                    "location ({:?}, {:?}), variable_to_check: {:.32}",
                    self.check_to_variable.indices()[ind],
                    check_var_col_ind,
                    self.variable_to_check.data_mut()[self.check_to_variable_nnz_map[ind]]
                );
            }
        }

        self.bound_magnitudes();

        &mut self.variable_to_check
    }

    pub fn run_iteration(&mut self, detectors: ArrayView1<Bit>) {
        debug!("Iteration {:?} start", self.current_iteration);
        self.compute_check_to_variable(detectors);
        // Now compute variable to check messages
        self.compute_variable_to_check();

        self.compute_hard_decision();
        debug!("Iteration {:?} end", self.current_iteration);
    }

    pub fn build_result(
        &mut self,
        success: bool,
        decoded_detectors: Array1<Bit>,
        max_iter: usize,
    ) -> DecodeResult {
        DecodeResult {
            decoding: self.decoding.clone(),
            decoded_detectors,
            posterior_ratios: self.posterior_ratios.clone().mapv_into_any(|val| {
                let posterior = N::to_f64(&val).unwrap();
                match self.config.data_scale_value {
                    Some(scale_val) => posterior / scale_val,
                    None => posterior,
                }
            }),
            success,
            decoding_quality: if success {
                self.get_decoding_quality(self.decoding.clone().view())
            } else {
                f64::MAX
            },
            iterations: self.current_iteration,
            max_iter,
            extra: BPExtraResult::None,
        }
    }
    fn bound_magnitudes(&mut self) {
        // Bound magnitudes
        if self.max_data_value.is_some() {
            let max_val = self.max_data_value.unwrap();
            self.variable_to_check
                .data_mut()
                .iter_mut()
                .for_each(|v| *v = Self::bound_value_magnitude(*v, max_val));
            self.posterior_ratios
                .iter_mut()
                .for_each(|v| *v = Self::bound_value_magnitude(*v, max_val));
        }
    }

    fn bound_value_magnitude(value: N, max_val: N) -> N
    where
        N: std::ops::Add,
    {
        if value < max_val.neg() {
            max_val.neg()
        } else if value > max_val {
            max_val
        } else {
            value
        }
    }

    fn compute_hard_decision(&mut self) {
        for (idx, posterior) in self.posterior_ratios.iter().enumerate() {
            self.decoding[idx] = Bit::from((*posterior) <= N::zero());
        }
        debug!("Posteriors: {:?}", self.posterior_ratios);
        debug!("Hard decision: {:?}", self.decoding);
    }

    pub fn compute_decoded_detectors(&self) -> Array1<Bit> {
        self.get_detectors(self.decoding.view())
    }

    // Check the convergence of the problem instance
    pub fn check_convergence(
        &self,
        detectors: ArrayView1<Bit>,
        decoded_detectors: ArrayView1<Bit>,
    ) -> bool {
        detectors == decoded_detectors
    }
}

impl<N> Decoder for MinSumBPDecoder<N>
where
    N: PartialEq
        + Debug
        + Default
        + Clone
        + Copy
        + FromPrimitive
        + ToPrimitive
        + Signed
        + Bounded
        + std::cmp::PartialOrd
        + std::ops::Add
        + std::ops::AddAssign
        + std::ops::Mul<N>
        + std::ops::MulAssign
        + std::ops::DivAssign
        + Send
        + Sync
        + std::fmt::Display
        + 'static,
{
    fn check_matrix(&self) -> Arc<SparseBitMatrix> {
        self.check_matrix.clone()
    }

    fn log_prior_ratios(&mut self) -> Array1<f64> {
        self.config.log_prior_ratios()
    }

    fn decode_detailed(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult {
        // Initialize probability ratios
        self.current_iteration = 0;
        self.initialize_variable_to_check();
        // Initialize posteriors if needed for mem-BP
        if self.config.gamma0.is_some() {
            self.posterior_ratios = self.log_prior_ratios.clone()
        };
        let mut success: bool = false;
        let mut decoded_detectors = Array1::default(detectors.dim());

        for _ in 0..self.config.max_iter {
            self.run_iteration(detectors);
            self.current_iteration += 1;
            decoded_detectors = self.compute_decoded_detectors();
            success = self.check_convergence(detectors, decoded_detectors.view());

            // If we have converged may now exit
            if success {
                debug!("Succeeded on iteration {:?}", self.current_iteration);
                break;
            }
        }

        self.build_result(success, decoded_detectors, self.config.max_iter)
    }
}

impl<N> DecoderRunner for MinSumBPDecoder<N> where
    N: PartialEq
        + Debug
        + Default
        + Clone
        + Copy
        + FromPrimitive
        + ToPrimitive
        + Signed
        + Bounded
        + std::cmp::PartialOrd
        + std::ops::Add
        + std::ops::AddAssign
        + std::ops::Mul<N>
        + std::ops::MulAssign
        + std::ops::DivAssign
        + Send
        + Sync
        + std::fmt::Display
        + 'static
{
}

#[cfg(test)]
mod tests {
    use crate::bipartite_graph::BipartiteGraph;

    use super::*;
    use env_logger;
    use ndarray::prelude::*;

    use crate::dem::DetectorErrorModel;
    use crate::utilities::test::get_test_data_path;
    use ndarray::Array2;
    use ndarray_npy::read_npy;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn decode_detailed_repetition_code() {
        init();

        // Build 3, 2 qubit repetition code with weight 2 checks
        let check_matrix = array![[1, 1, 0], [0, 1, 1],];

        let check_matrix: SparseBipartiteGraph<_> = SparseBipartiteGraph::from_dense(check_matrix);
        let arc_check_matrix = Arc::new(check_matrix);

        let iterations = 10;
        let bp_config = MinSumDecoderConfig {
            error_priors: array![0.003, 0.003, 0.003],
            max_iter: iterations,
            ..Default::default()
        };
        let arc_bp_config = Arc::new(bp_config);

        let mut decoder: MinSumBPDecoder<f64> =
            MinSumBPDecoder::new(arc_check_matrix, arc_bp_config);

        let error = array![0, 0, 0];
        let detectors: Array1<Bit> = array![0, 0];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);

        let error = array![1, 0, 0];
        let detectors: Array1<Bit> = array![1, 0];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);

        let error = array![0, 1, 0];
        let detectors: Array1<Bit> = array![1, 1];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);

        let error = array![0, 0, 1];
        let detectors: Array1<Bit> = array![0, 1];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);
    }

    #[test]
    fn decode_detailed_repetition_code_int() {
        init();

        // Build 3, 2 qubit repetition code with weight 2 checks
        let check_matrix = array![[1, 1, 0], [0, 1, 1],];

        let check_matrix: SparseBipartiteGraph<_> = SparseBipartiteGraph::from_dense(check_matrix);
        let arc_check_matrix = Arc::new(check_matrix);

        let iterations = 10;

        let bits = 7;
        let scale = 4.0;

        let bp_config = MinSumDecoderConfig {
            error_priors: array![0.003, 0.003, 0.003],
            max_iter: iterations,
            max_data_value: Some(((1 << bits) - 1) as f64),
            data_scale_value: Some(scale),
            ..Default::default()
        };
        let arc_bp_config = Arc::new(bp_config);

        let mut decoder: MinSumBPDecoder<isize> =
            MinSumBPDecoder::new(arc_check_matrix, arc_bp_config);

        let error = array![0, 0, 0];
        let detectors: Array1<Bit> = array![0, 0];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);

        let error = array![1, 0, 0];
        let detectors: Array1<Bit> = array![1, 0];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);

        let error = array![0, 1, 0];
        let detectors: Array1<Bit> = array![1, 1];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);

        let error = array![0, 0, 1];
        let detectors: Array1<Bit> = array![0, 1];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);
    }

    #[test]
    fn decode_detailed_144_12_12() {
        let resources = get_test_data_path();
        let code_144_12_12 =
            DetectorErrorModel::load(resources.join("144_12_12")).expect("Unable to load the code");
        let detectors_144_12_12: Array2<Bit> =
            read_npy(resources.join("144_12_12_detectors.npy")).expect("Unable to open file");
        let check_matrix = Arc::new(code_144_12_12.detector_error_matrix);
        let bp_config_144_12_12 = MinSumDecoderConfig {
            error_priors: code_144_12_12.error_priors,
            alpha: Some(0.),
            ..Default::default()
        };
        let config = Arc::new(bp_config_144_12_12);

        let mut decoder_144_12_12: MinSumBPDecoder<f64> =
            MinSumBPDecoder::new(check_matrix, config);
        let num_errors = 100;
        let detectors_slice = detectors_144_12_12.slice(s![..num_errors, ..]);
        let results = decoder_144_12_12.par_decode_detailed_batch(detectors_slice);

        assert!(
            results.iter().map(|x| x.success as usize).sum::<usize>() as f64
                >= (detectors_slice.shape()[0] as f64) * 0.93
        );

        assert_eq!(results[0].decoding.len(), 8785);
    }

    #[test]
    fn decode_144_12_12() {
        let resources = get_test_data_path();
        let code_144_12_12 =
            DetectorErrorModel::load(resources.join("144_12_12")).expect("Unable to load the code");
        let detectors_144_12_12: Array2<Bit> =
            read_npy(resources.join("144_12_12_detectors.npy")).expect("Unable to open file");
        let check_matrix = Arc::new(code_144_12_12.detector_error_matrix);
        let bp_config_144_12_12 = MinSumDecoderConfig {
            error_priors: code_144_12_12.error_priors,
            ..Default::default()
        };
        let config = Arc::new(bp_config_144_12_12);

        let mut decoder_144_12_12: MinSumBPDecoder<f64> =
            MinSumBPDecoder::new(check_matrix, config);
        let num_errors = 100;
        let detectors_slice = detectors_144_12_12.slice(s![..num_errors, ..]);

        let results = decoder_144_12_12.par_decode_batch(detectors_slice);

        let results_detailed = decoder_144_12_12.par_decode_detailed_batch(detectors_slice);

        for i in 0..results.shape()[0] {
            assert!(results.row(i) == results_detailed[i].decoding)
        }
    }

    #[test]
    fn decode_detailed_144_12_12_membp() {
        let resources = get_test_data_path();
        let code_144_12_12 =
            DetectorErrorModel::load(resources.join("144_12_12")).expect("Unable to load the code");
        let detectors_144_12_12: Array2<Bit> =
            read_npy(resources.join("144_12_12_detectors.npy")).expect("Unable to open file");
        let check_matrix = Arc::new(code_144_12_12.detector_error_matrix);
        let bp_config_144_12_12 = MinSumDecoderConfig {
            error_priors: code_144_12_12.error_priors,
            gamma0: Some(0.15),
            ..Default::default()
        };
        let config = Arc::new(bp_config_144_12_12);

        let mut decoder_144_12_12: MinSumBPDecoder<f64> =
            MinSumBPDecoder::new(check_matrix, config);
        let num_errors = 100;
        let detectors_slice = detectors_144_12_12.slice(s![..num_errors, ..]);
        let par_results = decoder_144_12_12.par_decode_detailed_batch(detectors_slice);
        let results = decoder_144_12_12.decode_detailed_batch(detectors_slice);
        assert!(
            results.iter().map(|x| x.success as usize).sum::<usize>() as f64
                == par_results
                    .iter()
                    .map(|x| x.success as usize)
                    .sum::<usize>() as f64
        );
        assert!(
            results.iter().map(|x| x.success as usize).sum::<usize>() as f64
                >= (detectors_slice.shape()[0] as f64) * 0.93
        );

        assert_eq!(results[0].decoding.len(), 8785);
    }

    #[test]
    fn decode_detailed_144_12_12_int() {
        let resources = get_test_data_path();
        let code_144_12_12 =
            DetectorErrorModel::load(resources.join("144_12_12")).expect("Unable to load the code");
        let detectors_144_12_12: Array2<Bit> =
            read_npy(resources.join("144_12_12_detectors.npy")).expect("Unable to open file");
        let check_matrix = Arc::new(code_144_12_12.detector_error_matrix);

        let bits = 16;
        let scale = 8.0;

        let bp_config_144_12_12 = MinSumDecoderConfig {
            error_priors: code_144_12_12.error_priors,
            max_data_value: Some(((1 << bits) - 1) as f64),
            data_scale_value: Some(scale),
            alpha: Some(0.),
            ..Default::default()
        };
        let config = Arc::new(bp_config_144_12_12);

        let mut decoder_144_12_12: MinSumBPDecoder<isize> =
            MinSumBPDecoder::new(check_matrix, config);
        let num_errors = 100;
        let detectors_slice = detectors_144_12_12.slice(s![..num_errors, ..]);
        let results = decoder_144_12_12.par_decode_detailed_batch(detectors_slice);

        assert!(
            results.iter().map(|x| x.success as usize).sum::<usize>() as f64
                >= (detectors_slice.shape()[0] as f64) * 0.93
        );

        assert_eq!(results[0].decoding.len(), 8785);
    }
}
