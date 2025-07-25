// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::decoder::{Bit, DecodeResult, Decoder, DecoderRunner, Mod2Mul, SparseBitMatrix};
use serde::{Deserialize, Serialize};

use indicatif::{ParallelProgressIterator, ProgressFinish, ProgressIterator, ProgressStyle};
use ndarray::{stack, Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

use std::sync::Arc;

pub trait ObservableDecoder: Decoder {
    /// The logical action matrix of the underlying trait.
    fn observable_error_matrix(&self) -> Arc<SparseBitMatrix>;

    /// Compute the logical errors from the logical action matrix.
    fn compute_observables(&self, errors: ArrayView1<Bit>) -> Array1<Bit> {
        self.observable_error_matrix().mul_mod2(&errors.to_owned())
    }

    fn decode_observables(&mut self, detectors: ArrayView1<Bit>) -> Array1<Bit> {
        let errors = self.decode(detectors);
        self.compute_observables(errors.view())
    }
}

#[derive(Clone)]
pub struct ObservableDecoderRunner<'a> {
    decoder: Box<dyn Decoder + Send + 'a>,
    observable_error_matrix: Arc<SparseBitMatrix>,
    include_decode_result: bool,
}

impl<'a> ObservableDecoderRunner<'a> {
    pub fn new(
        decoder: Box<dyn Decoder + Send + 'a>,
        observable_error_matrix: Arc<SparseBitMatrix>,
        include_decode_result: bool,
    ) -> Self {
        ObservableDecoderRunner {
            decoder,
            observable_error_matrix,
            include_decode_result,
        }
    }

    pub fn get_decoder(&self) -> &dyn Decoder {
        self.decoder.as_ref()
    }

    pub fn get_decoder_mut(&mut self) -> &mut dyn Decoder {
        self.decoder.as_mut()
    }

    pub fn decode_observables(&mut self, detectors: ArrayView1<Bit>) -> Array1<Bit> {
        let decode_result = self.decode(detectors.view());
        self.compute_observables(decode_result.view())
    }

    pub fn decode_observables_detailed(
        &mut self,
        detectors: ArrayView1<Bit>,
    ) -> ObservableDecodeResult {
        let decode_result = self.decode_detailed(detectors.view());
        let observables = self.compute_observables(decode_result.decoding.view());

        ObservableDecodeResult {
            observables,
            converged: decode_result.success,
            iterations: decode_result.iterations,
            true_decoding: None,
            physical_decode_result: if self.include_decode_result {
                Some(decode_result)
            } else {
                None
            },
        }
    }

    pub fn decode_observables_batch(&mut self, detectors: ArrayView2<Bit>) -> Array2<Bit> {
        let arrs: Vec<Array1<Bit>> = detectors
            .axis_iter(Axis(0))
            .map(|row| self.decode_observables(row))
            .collect();
        stack(Axis(0), &arrs.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap()
    }

    pub fn decode_observables_batch_progress_bar(
        &mut self,
        detectors: ArrayView2<Bit>,
        leave_progress_bar_on_finish: bool,
    ) -> Array2<Bit> {
        let finish_mode = match leave_progress_bar_on_finish {
            true => ProgressFinish::AndLeave,
            false => ProgressFinish::AndClear,
        };

        let arrs: Vec<Array1<Bit>> = detectors
            .axis_iter(Axis(0))
            .progress_with_style(self.get_progress_bar_style())
            .with_finish(finish_mode)
            .map(|row| self.decode_observables(row))
            .collect();
        stack(Axis(0), &arrs.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap()
    }

    pub fn decode_observables_detailed_batch(
        &mut self,
        detectors: ArrayView2<Bit>,
    ) -> Vec<ObservableDecodeResult> {
        detectors
            .axis_iter(Axis(0))
            .map(|row| self.decode_observables_detailed(row))
            .collect()
    }

    pub fn par_decode_observables_batch(&mut self, detectors: ArrayView2<Bit>) -> Array2<Bit> {
        let arrs: Vec<Array1<Bit>> = detectors
            .axis_iter(Axis(0))
            .into_par_iter()
            .map_with(
                || self.clone(),
                |decoder, row| decoder().decode_observables(row),
            )
            .collect();
        stack(Axis(0), &arrs.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap()
    }

    pub fn par_decode_observables_batch_progress_bar(
        &mut self,
        detectors: ArrayView2<Bit>,
        leave_progress_bar_on_finish: bool,
    ) -> Array2<Bit> {
        let finish_mode = match leave_progress_bar_on_finish {
            true => ProgressFinish::AndLeave,
            false => ProgressFinish::AndClear,
        };

        let arrs: Vec<Array1<Bit>> = detectors
            .axis_iter(Axis(0))
            .into_par_iter()
            .progress_with_style(self.get_progress_bar_style())
            .with_finish(finish_mode)
            .map_with(
                || self.clone(),
                |decoder, row| decoder().decode_observables(row),
            )
            .collect();
        stack(Axis(0), &arrs.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap()
    }

    pub fn par_decode_observables_detailed_batch(
        &mut self,
        detectors: ArrayView2<Bit>,
    ) -> Vec<ObservableDecodeResult> {
        detectors
            .axis_iter(Axis(0))
            .into_par_iter()
            .map_with(
                || self.clone(),
                |decoder, row| decoder().decode_observables_detailed(row),
            )
            .collect()
    }

    /// Decode a batch displaying a progress bar
    pub fn decode_observables_detailed_batch_progress_bar(
        &mut self,
        detectors: ArrayView2<Bit>,
        leave_progress_bar_on_finish: bool,
    ) -> Vec<ObservableDecodeResult> {
        let finish_mode = match leave_progress_bar_on_finish {
            true => ProgressFinish::AndLeave,
            false => ProgressFinish::AndClear,
        };

        detectors
            .axis_iter(Axis(0))
            .progress_with_style(self.get_progress_bar_style())
            .with_finish(finish_mode)
            .map(|row| self.decode_observables_detailed(row))
            .collect()
    }

    pub fn par_decode_observables_detailed_batch_progress_bar(
        &mut self,
        detectors: ArrayView2<Bit>,
        leave_progress_bar_on_finish: bool,
    ) -> Vec<ObservableDecodeResult> {
        let finish_mode = match leave_progress_bar_on_finish {
            true => ProgressFinish::AndLeave,
            false => ProgressFinish::AndClear,
        };

        detectors
            .axis_iter(Axis(0))
            .into_par_iter()
            .progress_with_style(self.get_progress_bar_style())
            .with_finish(finish_mode)
            .map_with(
                || self.clone(),
                |decoder, row| decoder().decode_observables_detailed(row),
            )
            .collect()
    }

    pub fn from_errors_decode_observables(&mut self, errors: ArrayView1<Bit>) -> Array1<Bit> {
        let detectors = self.get_detectors(errors);
        let decode_result = self.decode(detectors.view());
        self.compute_observables(decode_result.view())
    }

    pub fn from_errors_decode_observables_detailed(
        &mut self,
        errors: ArrayView1<Bit>,
    ) -> ObservableDecodeResult {
        let detectors = self.get_detectors(errors);
        let decode_result = self.decode_detailed(detectors.view());
        let observables = self.compute_observables(errors);
        let decoded_observables = self.compute_observables(decode_result.decoding.view());

        let error_detected: bool = observables != decoded_observables;
        let error_mismatch_detected: bool = errors != decode_result.decoding;
        let better_decoding_quality: bool =
            decode_result.decoding_quality < self.get_decoding_quality(errors.clone().view());

        let unconverged_no_error: bool = !error_detected && !decode_result.success;
        let better_decoding_quality_error: bool = error_detected && better_decoding_quality;
        let worse_decoding_quality_error: bool = error_detected && !better_decoding_quality;

        ObservableDecodeResult {
            observables,
            converged: decode_result.success,
            iterations: decode_result.iterations,
            true_decoding: Some(TrueDecodingResults {
                error_detected,
                error_mismatch_detected,
                better_decoding_quality_error,
                worse_decoding_quality_error,
                unconverged_no_error,
            }),
            physical_decode_result: if self.include_decode_result {
                Some(decode_result)
            } else {
                None
            },
        }
    }

    pub fn from_errors_decode_observables_batch(&mut self, errors: ArrayView2<Bit>) -> Array2<Bit> {
        let arrs: Vec<Array1<Bit>> = errors
            .axis_iter(Axis(0))
            .map(|row| self.from_errors_decode_observables(row))
            .collect();
        stack(Axis(0), &arrs.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap()
    }

    pub fn par_from_errors_decode_observables_batch(
        &mut self,
        errors: ArrayView2<Bit>,
    ) -> Array2<Bit> {
        let arrs: Vec<Array1<Bit>> = errors
            .axis_iter(Axis(0))
            .into_par_iter()
            .map_with(
                || self.clone(),
                |decoder, row| decoder().from_errors_decode_observables(row),
            )
            .collect();
        stack(Axis(0), &arrs.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap()
    }

    pub fn from_errors_decode_observables_batch_progress_bar(
        &mut self,
        errors: ArrayView2<Bit>,
        leave_progress_bar_on_finish: bool,
    ) -> Array2<Bit> {
        let finish_mode = match leave_progress_bar_on_finish {
            true => ProgressFinish::AndLeave,
            false => ProgressFinish::AndClear,
        };

        let arrs: Vec<Array1<Bit>> = errors
            .axis_iter(Axis(0))
            .progress_with_style(self.get_progress_bar_style())
            .with_finish(finish_mode)
            .map(|row| self.from_errors_decode_observables(row))
            .collect();
        stack(Axis(0), &arrs.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap()
    }

    pub fn par_from_errors_decode_observables_batch_progress_bar(
        &mut self,
        errors: ArrayView2<Bit>,
        leave_progress_bar_on_finish: bool,
    ) -> Array2<Bit> {
        let finish_mode = match leave_progress_bar_on_finish {
            true => ProgressFinish::AndLeave,
            false => ProgressFinish::AndClear,
        };

        let arrs: Vec<Array1<Bit>> = errors
            .axis_iter(Axis(0))
            .into_par_iter()
            .progress_with_style(self.get_progress_bar_style())
            .with_finish(finish_mode)
            .map_with(
                || self.clone(),
                |decoder, row| decoder().from_errors_decode_observables(row),
            )
            .collect();
        stack(Axis(0), &arrs.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap()
    }

    pub fn from_errors_decode_observables_detailed_batch(
        &mut self,
        errors: ArrayView2<Bit>,
    ) -> Vec<ObservableDecodeResult> {
        errors
            .axis_iter(Axis(0))
            .map(|row| self.from_errors_decode_observables_detailed(row))
            .collect()
    }

    pub fn par_from_errors_decode_observables_detailed_batch(
        &mut self,
        errors: ArrayView2<Bit>,
    ) -> Vec<ObservableDecodeResult> {
        errors
            .axis_iter(Axis(0))
            .into_par_iter()
            .map_with(
                || self.clone(),
                |decoder, row| decoder().from_errors_decode_observables_detailed(row),
            )
            .collect()
    }

    /// Decode a batch displaying a progress bar
    pub fn from_errors_decode_observables_detailed_batch_progress_bar(
        &mut self,
        errors: ArrayView2<Bit>,
        leave_progress_bar_on_finish: bool,
    ) -> Vec<ObservableDecodeResult> {
        let finish_mode = match leave_progress_bar_on_finish {
            true => ProgressFinish::AndLeave,
            false => ProgressFinish::AndClear,
        };

        errors
            .axis_iter(Axis(0))
            .progress_with_style(self.get_progress_bar_style())
            .with_finish(finish_mode)
            .map(|row| self.from_errors_decode_observables_detailed(row))
            .collect()
    }

    pub fn par_from_errors_decode_observables_detailed_batch_progress_bar(
        &mut self,
        errors: ArrayView2<Bit>,
        leave_progress_bar_on_finish: bool,
    ) -> Vec<ObservableDecodeResult> {
        let finish_mode = match leave_progress_bar_on_finish {
            true => ProgressFinish::AndLeave,
            false => ProgressFinish::AndClear,
        };

        errors
            .axis_iter(Axis(0))
            .into_par_iter()
            .progress_with_style(self.get_progress_bar_style())
            .with_finish(finish_mode)
            .map_with(
                || self.clone(),
                |decoder, row| decoder().from_errors_decode_observables_detailed(row),
            )
            .collect()
    }

    fn get_progress_bar_style(&self) -> ProgressStyle {
        ProgressStyle::default_bar().template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({per_sec}, {eta})").unwrap()
    }
}

impl DecoderRunner for ObservableDecoderRunner<'_> {}

impl Decoder for ObservableDecoderRunner<'_> {
    fn check_matrix(&self) -> Arc<SparseBitMatrix> {
        self.get_decoder().check_matrix()
    }
    fn log_prior_ratios(&mut self) -> Array1<f64> {
        self.get_decoder_mut().log_prior_ratios()
    }
    fn decode_detailed(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult {
        self.get_decoder_mut().decode_detailed(detectors)
    }
    fn get_decoding_quality(&mut self, errors: ArrayView1<u8>) -> f64 {
        self.get_decoder_mut().get_decoding_quality(errors)
    }
}

impl ObservableDecoder for ObservableDecoderRunner<'_> {
    fn observable_error_matrix(&self) -> Arc<SparseBitMatrix> {
        self.observable_error_matrix.clone()
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ObservableDecodeResult {
    pub observables: Array1<Bit>,
    pub converged: bool,
    pub iterations: usize,
    pub true_decoding: Option<TrueDecodingResults>,
    pub physical_decode_result: Option<DecodeResult>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TrueDecodingResults {
    pub error_detected: bool,
    pub error_mismatch_detected: bool,
    pub unconverged_no_error: bool,
    pub better_decoding_quality_error: bool,
    pub worse_decoding_quality_error: bool,
}

#[cfg(test)]
mod tests {

    use super::*;
    use ndarray::prelude::*;

    use crate::bp::min_sum::{MinSumBPDecoder, MinSumDecoderConfig};
    use crate::dem::DetectorErrorModel;
    use crate::utilities::test::get_test_data_path;
    use ndarray::Array2;
    use ndarray_npy::read_npy;

    #[test]
    fn min_sum_decode_144_12_12() {
        let resources = get_test_data_path();
        let code_144_12_12 =
            DetectorErrorModel::load(resources.join("144_12_12")).expect("Unable to load the code");
        let errors_144_12_12: Array2<Bit> =
            read_npy(resources.join("144_12_12_errors.npy")).expect("Unable to open file");
        let bp_config_144_12_12 = MinSumDecoderConfig {
            error_priors: code_144_12_12.error_priors,
            alpha: Some(0.),
            ..Default::default()
        };

        let check_matrix = Arc::new(code_144_12_12.detector_error_matrix);
        let bp_config = Arc::new(bp_config_144_12_12);
        let decoder_144_12_12: Box<MinSumBPDecoder<f64>> =
            Box::new(MinSumBPDecoder::new(check_matrix, bp_config));

        let obs_matrix = Arc::new(code_144_12_12.observable_error_matrix);
        let mut observable_decoder =
            ObservableDecoderRunner::new(decoder_144_12_12, obs_matrix, true);

        let num_errors = 100;
        let errors_slice = errors_144_12_12.slice(s![..num_errors, ..]);
        let results =
            observable_decoder.par_from_errors_decode_observables_detailed_batch(errors_slice);

        // Assert 90% correct.
        assert!(
            results
                .iter()
                .map(|x| x.physical_decode_result.as_ref().unwrap().success as usize)
                .sum::<usize>() as f64
                >= (errors_slice.shape()[0] as f64) * 0.93
        );

        assert!(
            results
                .iter()
                .map(|x| x.true_decoding.as_ref().unwrap().error_mismatch_detected as usize)
                .sum::<usize>() as f64
                >= (errors_slice.shape()[0] as f64) * 0.96
        );

        assert_eq!(
            results[0].observables,
            array![1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1]
        );

        assert!(
            results
                .iter()
                .map(|x| x.true_decoding.as_ref().unwrap().error_detected as usize)
                .sum::<usize>() as f64
                <= (errors_slice.shape()[0] as f64) * 0.07
        );

        let direct_results = observable_decoder.decode_observables(
            observable_decoder
                .get_detectors(errors_144_12_12.row(0))
                .view(),
        );
        assert_eq!(results[0].observables, direct_results);

        let results2 = observable_decoder.par_decode_observables_batch(
            observable_decoder
                .get_detectors_batch(errors_144_12_12.view())
                .view(),
        );

        assert_eq!(results2.row(0), direct_results);
    }
}
