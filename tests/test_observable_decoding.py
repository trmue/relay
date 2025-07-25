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
import pytest

import relay_bp


@pytest.fixture
def repetition_code_logical_evaluator(repetition_code_config):
    physical_decoder = relay_bp.MinSumBPDecoderF32(**repetition_code_config)
    observable_error_matrix = np.array([[1, 1, 1]])

    return relay_bp.ObservableDecoderRunner(physical_decoder, observable_error_matrix)


def test_observable_decoder_evaluator(repetition_code_logical_evaluator):
    """Test logical decoding"""

    errors = np.array([[0, 1, 0], [1, 1, 1]], dtype=np.uint8)
    logical_decode_results_detailed = (
        repetition_code_logical_evaluator.from_errors_decode_observables_detailed_batch(
            errors
        )
    )
    assert not logical_decode_results_detailed[0].error_detected
    assert not logical_decode_results_detailed[0].error_mismatch_detected
    assert logical_decode_results_detailed[0].converged
    assert np.all(logical_decode_results_detailed[0].observables)

    logical_decode_results = (
        repetition_code_logical_evaluator.from_errors_decode_observables_batch(
            errors,
            parallel=True,
        )
    )
    for i in range(len(logical_decode_results_detailed)):
        np.all(
            logical_decode_results_detailed[i].observables == logical_decode_results[i]
        )
