// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use fixed::{types::extra as fbits, FixedI64};

use crate::bp::min_sum::{MinSumBPDecoder, MinSumDecoderConfig};
use crate::decoder::{Bit, DecodeResult, Decoder, DecoderRunner, SparseBitMatrix};
use ndarray::{Array1, ArrayView1};

use std::sync::Arc;

/// Runtime implemetnation of fixed point precision which routes to the appropriate compile time
/// implementation.
#[derive(Clone)]
pub struct MinSumBPDecoderFixed {
    decoder: Box<dyn Decoder + Send + 'static>,
}

impl MinSumBPDecoderFixed {
    pub fn new(
        check_matrix: Arc<SparseBitMatrix>,
        min_sum_config: Arc<MinSumDecoderConfig>,
    ) -> Self {
        MinSumBPDecoderFixed {
            decoder: MinSumBPDecoderFixed::build_min_sum(check_matrix, min_sum_config),
        }
    }

    fn build_min_sum(
        check_matrix: Arc<SparseBitMatrix>,
        min_sum_config: Arc<MinSumDecoderConfig>,
    ) -> Box<dyn Decoder + Send + 'static> {
        macro_rules! genmatch {
            ($Variant:path) => {
                return Box::new(MinSumBPDecoder::<FixedI64<$Variant>>::new(
                    check_matrix,
                    min_sum_config,
                ))
            };
        }

        match min_sum_config.frac_bits {
            Some(0) => genmatch!(fbits::U0),
            Some(1) => genmatch!(fbits::U1),
            Some(2) => genmatch!(fbits::U2),
            Some(3) => genmatch!(fbits::U3),
            Some(4) => genmatch!(fbits::U4),
            Some(5) => genmatch!(fbits::U5),
            Some(6) => genmatch!(fbits::U6),
            Some(7) => genmatch!(fbits::U7),
            Some(8) => genmatch!(fbits::U8),
            Some(9) => genmatch!(fbits::U9),
            Some(10) => genmatch!(fbits::U10),
            Some(11) => genmatch!(fbits::U11),
            Some(12) => genmatch!(fbits::U12),
            Some(13) => genmatch!(fbits::U13),
            Some(14) => genmatch!(fbits::U14),
            Some(15) => genmatch!(fbits::U15),
            Some(16) => genmatch!(fbits::U16),
            Some(17) => genmatch!(fbits::U17),
            Some(18) => genmatch!(fbits::U18),
            Some(19) => genmatch!(fbits::U19),
            Some(20) => genmatch!(fbits::U20),
            Some(21) => genmatch!(fbits::U21),
            Some(22) => genmatch!(fbits::U22),
            Some(23) => genmatch!(fbits::U23),
            Some(24) => genmatch!(fbits::U24),
            Some(25) => genmatch!(fbits::U25),
            Some(26) => genmatch!(fbits::U26),
            Some(27) => genmatch!(fbits::U27),
            Some(28) => genmatch!(fbits::U28),
            Some(29) => genmatch!(fbits::U29),
            Some(30) => genmatch!(fbits::U30),
            Some(31) => genmatch!(fbits::U31),
            _ => panic!(
                "Decoder does not support {:?} fractional bits",
                min_sum_config.frac_bits
            ),
        }
    }
}

impl Decoder for MinSumBPDecoderFixed {
    fn check_matrix(&self) -> Arc<SparseBitMatrix> {
        self.decoder.check_matrix()
    }

    fn decode_detailed(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult {
        self.decoder.decode_detailed(detectors)
    }

    fn log_prior_ratios(&mut self) -> Array1<f64> {
        self.decoder.log_prior_ratios()
    }
}

impl DecoderRunner for MinSumBPDecoderFixed {}

#[cfg(test)]
mod tests {
    use crate::bipartite_graph::BipartiteGraph;
    use crate::bipartite_graph::SparseBipartiteGraph;
    use crate::decoder::{Bit, Decoder, DecoderRunner};

    use super::*;
    use env_logger;
    use ndarray::prelude::*;

    use crate::dem::DetectorErrorModel;
    use crate::utilities::test::get_test_data_path;
    use ndarray::Array2;
    use ndarray_npy::read_npy;

    use std::sync::Arc;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn decode_repetition_code_int() {
        init();

        // Build 3, 2 qubit repetition code with weight 2 checks
        let check_matrix = array![[1, 1, 0], [0, 1, 1],];

        let check_matrix: SparseBipartiteGraph<_> = SparseBipartiteGraph::from_dense(check_matrix);

        let iterations = 10;
        let mut bp_config = MinSumDecoderConfig {
            error_priors: array![0.003, 0.003, 0.003],
            max_iter: iterations,
            ..Default::default()
        };
        bp_config.set_fixed(5, 2);

        let check_matrix_arc = Arc::new(check_matrix);
        let config = Arc::new(bp_config);
        let mut decoder: MinSumBPDecoderFixed = MinSumBPDecoderFixed::new(check_matrix_arc, config);

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
    fn decode_144_12_12() {
        let resources = get_test_data_path();
        let code_144_12_12 =
            DetectorErrorModel::load(resources.join("144_12_12")).expect("Unable to load the code");
        let detectors_144_12_12: Array2<Bit> =
            read_npy(resources.join("144_12_12_detectors.npy")).expect("Unable to open file");
        let mut bp_config_144_12_12 = MinSumDecoderConfig {
            error_priors: code_144_12_12.error_priors,
            ..Default::default()
        };
        bp_config_144_12_12.set_fixed(5, 2);

        let check_matrix = Arc::new(code_144_12_12.detector_error_matrix);
        let bp_config = Arc::new(bp_config_144_12_12);
        let mut decoder_144_12_12: MinSumBPDecoderFixed =
            MinSumBPDecoderFixed::new(check_matrix, bp_config);
        let num_errors = 1;
        let detectors_slice = detectors_144_12_12.slice(s![..num_errors, ..]);
        let results = decoder_144_12_12.par_decode_detailed_batch(detectors_slice);

        assert!(
            results.iter().map(|x| x.success as usize).sum::<usize>() as f64
                >= (detectors_slice.shape()[0] as f64) * 0.93
        );

        assert_eq!(results[0].decoding.len(), 8785);
    }
}
