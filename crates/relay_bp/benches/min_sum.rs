// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use ndarray_npy::read_npy;
use relay_bp::bp::min_sum::{MinSumBPDecoder, MinSumDecoderConfig};
use relay_bp::decoder::{Bit, Decoder, DecoderRunner};
use relay_bp::dem::DetectorErrorModel;
use relay_bp::utilities::test::get_test_data_path;
use std::sync::Arc;

fn gross_code_benchmark(c: &mut Criterion) {
    let resources = get_test_data_path();
    let code_144_12_12 =
        DetectorErrorModel::load(resources.join("144_12_12")).expect("Unable to load the code");
    let detectorss_144_12_12: Array2<Bit> =
        read_npy(resources.join("144_12_12_detectors.npy")).expect("Unable to open file");

    let bp_config_144_12_12 = MinSumDecoderConfig {
        error_priors: code_144_12_12.error_priors,
        max_iter: 200,
        alpha: None,
        alpha_iteration_scaling_factor: 1.,
        ..Default::default()
    };

    let check_matrix = Arc::new(code_144_12_12.detector_error_matrix);
    let config = Arc::new(bp_config_144_12_12);
    let mut decoder_144_12_12: MinSumBPDecoder<f64> = MinSumBPDecoder::new(check_matrix, config);

    let mut group = c.benchmark_group("min_sum_144_12_12");
    group.sample_size(10);
    group.bench_function("100_samples", |b| {
        b.iter(|| decoder_144_12_12.decode_batch(black_box(detectorss_144_12_12.view())))
    });
    group.bench_function("100_samples_par", |b| {
        b.iter(|| decoder_144_12_12.par_decode_batch(black_box(detectorss_144_12_12.view())))
    });
}

criterion_group!(benches, gross_code_benchmark);
criterion_main!(benches);
