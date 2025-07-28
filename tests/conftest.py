# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import pytest
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

pytest.importorskip("relay_bp", reason="relay_bp is required for these tests")


@pytest.fixture
def repetition_code_error_priors():
    return np.array([0.003, 0.003, 0.003], dtype=np.float64)


@pytest.fixture
def repetition_code_dense():
    return np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=np.uint8,
    )


@pytest.fixture
def repetition_code_sparse_csr(repetition_code_dense):
    return csr_matrix(repetition_code_dense)


@pytest.fixture
def repetition_code_sparse_csc(repetition_code_dense):
    return csc_matrix(repetition_code_dense)


@pytest.fixture
def repetition_code_config(repetition_code_sparse_csr, repetition_code_error_priors):
    return {
        "check_matrix": repetition_code_sparse_csr,
        "error_priors": repetition_code_error_priors,
        "max_iter": 100,
        "alpha": 1.0,
    }


@pytest.fixture
def repetition_code_config_int(repetition_code_dense, repetition_code_error_priors):
    return {
        "check_matrix": repetition_code_dense,
        "error_priors": repetition_code_error_priors,
        "max_iter": 100,
        "alpha": 1.0,
        "data_scale_value": 4,
        "max_data_value": (1 << 16) - 1,
    }


@pytest.fixture
def repetition_code_config_fixed(
    repetition_code_sparse_csc, repetition_code_error_priors
):
    return {
        "check_matrix": repetition_code_sparse_csc,
        "error_priors": repetition_code_error_priors,
        "max_iter": 100,
        "alpha": 1.0,
        "int_bits": 5,
        "frac_bits": 2,
    }
