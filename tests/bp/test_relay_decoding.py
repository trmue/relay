# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
import numpy as np

import relay_bp


def test_decode_detailed(repetition_code_config):
    repetition_code_config.pop("max_iter", None)
    decoder = relay_bp.RelayDecoderF32(
        **repetition_code_config,
        pre_iter=120,
        num_sets=40,
        set_max_iter=60,
        gamma_dist_interval=(-0.24, 0.66),
        explicit_gammas=None,
        stop_nconv=3,
    )

    detectors = np.array([1, 1], dtype=np.uint8)

    result = decoder.decode_detailed(detectors)
    assert result.success
    assert np.all(result.decoding == np.array([0, 1, 0]))
    assert result.iterations <= 120
    assert result.max_iter == 120


def test_decode_detailed_batch(repetition_code_config):
    repetition_code_config.pop("max_iter", None)
    decoder = relay_bp.RelayDecoderF32(
        **repetition_code_config,
        pre_iter=120,
        num_sets=40,
        set_max_iter=60,
        gamma_dist_interval=(-0.24, 0.66),
        explicit_gammas=None,
        stop_nconv=3,
    )

    detectors = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.uint8)

    results = decoder.decode_detailed_batch(detectors)

    result0 = results[0]
    assert result0.success
    assert np.all(result0.decoding == np.array([1, 0, 0]))

    result1 = results[1]
    assert result1.success
    assert np.all(result1.decoding == np.array([0, 1, 0]))

    result2 = results[2]
    assert result2.success
    assert np.all(result2.decoding == np.array([0, 0, 1]))


def test_decode_priors_of_one_half():
    """Priors of exactly 1/2 forming a stopping set decode with min_log_prior_ratio."""
    check_matrix = np.ones((1, 3), dtype=np.uint8)
    config = dict(
        error_priors=np.array([0.5, 0.5, 0.5]),
        gamma0=0.1,
        pre_iter=80,
        num_sets=60,
        set_max_iter=60,
        stop_nconv=3,
    )

    # A log prior ratio of 0 carries no sign, so no syndrome information reaches the columns
    decoder = relay_bp.RelayDecoderF32(check_matrix, **config)
    result = decoder.decode_detailed(np.array([0], dtype=np.uint8))
    assert not result.success

    decoder = relay_bp.RelayDecoderF32(check_matrix, **config, min_log_prior_ratio=4e-5)
    result = decoder.decode_detailed(np.array([0], dtype=np.uint8))
    assert result.success
    assert np.all(result.decoding == np.array([0, 0, 0]))
