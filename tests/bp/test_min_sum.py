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
    decoder = relay_bp.MinSumBPDecoderF32(**repetition_code_config)

    detectors = np.array([1, 1], dtype=np.uint8)

    result = decoder.decode(detectors)
    assert np.all(result == np.array([0, 1, 0]))


def test_decode_detailed(repetition_code_config):
    decoder = relay_bp.MinSumBPDecoderF32(**repetition_code_config)

    detectors = np.array([1, 1], dtype=np.uint8)

    result = decoder.decode_detailed(detectors)
    assert result.success
    assert np.all(result.decoding == np.array([0, 1, 0]))
    assert result.iterations <= 100
    assert result.max_iter == 100


def test_decode_detailed_batch(repetition_code_config):
    decoder = relay_bp.MinSumBPDecoderF32(**repetition_code_config)

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


def test_decode_batch(repetition_code_config):
    decoder = relay_bp.MinSumBPDecoderF32(**repetition_code_config)

    detectors = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.uint8)

    results = decoder.decode_batch(detectors)

    assert np.all(results[0] == np.array([1, 0, 0]))

    assert np.all(results[1] == np.array([0, 1, 0]))

    assert np.all(results[2] == np.array([0, 0, 1]))


def test_decode_int(repetition_code_config_int):
    decoder = relay_bp.MinSumBPDecoderI64(**repetition_code_config_int)

    detectors = np.array([1, 1], dtype=np.uint8)

    result = decoder.decode_detailed(detectors)
    assert result.success
    assert np.all(result.decoding == np.array([0, 1, 0]))
    assert result.iterations <= 100
    assert result.max_iter == 100


def test_decode_detailed_batch_int(repetition_code_config_int):
    decoder = relay_bp.MinSumBPDecoderI64(**repetition_code_config_int)

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
