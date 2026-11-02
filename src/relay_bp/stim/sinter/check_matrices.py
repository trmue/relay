# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

from dataclasses import dataclass

from scipy.sparse import csc_matrix
import numpy as np

# We use belief matchings utilities to extract the decoding matrices
from beliefmatching import detector_error_model_to_check_matrices

# Priors in [1/2 - MAX_PRIOR_MARGIN, 1/2] are clamped to 1/2 - MAX_PRIOR_MARGIN, so that every
# log prior ratio has a sign.
MAX_PRIOR_MARGIN = 1e-5


def combine_independent_priors(
    combine_matrix: csc_matrix, priors: np.ndarray
) -> np.ndarray:
    """Combine the priors of independent mechanisms flipping the same error.

    The error occurs if an odd number of its mechanisms fire, with probability
    ``(1 - prod(1 - 2 p_j)) / 2``. The result is at most 1/2 for priors in ``[0, 1/2]``.

    Args:
        combine_matrix: Matrix whose row ``i`` selects the mechanisms that flip error ``i``.
        priors: Prior of each mechanism.

    Returns:
        The prior of each error.
    """
    combine_matrix = csc_matrix(combine_matrix)
    combine_matrix.data = np.ones_like(combine_matrix.data, dtype=np.float64)
    factors = 1.0 - 2.0 * np.asarray(priors, dtype=np.float64)
    is_zero = factors == 0.0
    is_negative = factors < 0.0
    with np.errstate(divide="ignore"):
        log_magnitudes = np.where(is_zero, 0.0, np.log(np.abs(factors)))
    num_zero = combine_matrix @ is_zero.astype(np.float64)
    num_negative = combine_matrix @ is_negative.astype(np.float64)
    product = np.exp(combine_matrix @ log_magnitudes)
    product = np.where(num_negative % 2 == 1, -product, product)
    product = np.where(num_zero > 0, 0.0, product)
    return (1.0 - product) / 2.0


def clamp_priors(priors: np.ndarray, margin: float = MAX_PRIOR_MARGIN) -> np.ndarray:
    """Clamp priors in ``[1/2 - margin, 1/2]`` to ``1/2 - margin``.

    Args:
        priors: Error priors.
        margin: Distance below 1/2 that clamped priors are set to.

    Returns:
        A copy of ``priors`` with the clamped values.
    """
    priors = np.array(priors, dtype=np.float64)
    priors[(priors > 0.5 - margin) & (priors <= 0.5)] = 0.5 - margin
    return priors


@dataclass
class CheckMatrices:
    check_matrix: csc_matrix
    observables_matrix: csc_matrix
    error_priors: np.ndarray
    syndrome_bias: np.ndarray | None = None
    observables_bias: np.ndarray | None = None

    @classmethod
    def from_dem(
        cls,
        dem,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
    ) -> CheckMatrices:
        """Get decoding check matrices from dem.

        Args:
            dem: Detector error model to extract matrices from.
            decomposed_hyperedges: If errors have been decomposed we return only the remaining errors impacting observables.
                If none, we try to detect if decomposable to reduce decoding time.
            prune_decided_errors: Prune decided errors at construction time.
            threshold: A threshold magnitude to decimate probabilities of 0+threshold -> 0 and 1-threshold -> 1 for error pruning.
        """
        # We use belief matchings utilities to extract the decoding matrices
        try:
            dem_matrices = detector_error_model_to_check_matrices(
                dem,
                allow_undecomposed_hyperedges=decomposed_hyperedges is not None
                and not decomposed_hyperedges,
            )
            decomposed_hyperedges = True
        except ValueError as err:
            # the dem was not CSS, try again allowing for undecomposed errors
            if decomposed_hyperedges is None:
                decomposed_hyperedges = False
                dem_matrices = detector_error_model_to_check_matrices(
                    dem, allow_undecomposed_hyperedges=True
                )
            else:
                raise ValueError("The input dem is not decomposable") from err

        if decomposed_hyperedges:
            observables_matrix = dem_matrices.edge_observables_matrix
            check_matrix = dem_matrices.edge_check_matrix
            error_priors = combine_independent_priors(
                dem_matrices.hyperedge_to_edge_matrix, dem_matrices.priors
            )
        else:
            observables_matrix = dem_matrices.observables_matrix
            check_matrix = dem_matrices.check_matrix
            error_priors = dem_matrices.priors

        check_matrices = cls(
            check_matrix=check_matrix,
            observables_matrix=observables_matrix,
            error_priors=clamp_priors(error_priors),
        )
        if prune_decided_errors:
            check_matrices = check_matrices.prune_decided_errors(threshold=threshold)

        return check_matrices

    def prune_decided_errors(self, threshold: float = 0.0) -> CheckMatrices:
        """Prune decided errors from the check matrices.

        Args:
            threshold: A threshold magnitude to decimate probabilities of 0+threshold -> 0 and 1-threshold -> 1 for error pruning.
        """
        to_zero = np.argwhere(self.error_priors <= 0.0 + threshold).reshape(-1)
        to_one = np.argwhere(self.error_priors >= 1.0 - threshold).reshape(-1)

        if len(to_one):
            # Sparse sums return a 2-D np.matrix
            syndrome_bias = (
                np.asarray(self.check_matrix[:, to_one].sum(axis=1)).reshape(-1) % 2
            ).astype(np.uint8)
            observables_bias = (
                np.asarray(self.observables_matrix[:, to_one].sum(axis=1)).reshape(-1)
                % 2
            ).astype(np.uint8)
        else:
            syndrome_bias = None
            observables_bias = None

        prune = np.concatenate([to_zero, to_one])
        keep = np.where(
            np.logical_not(np.isin(np.arange(len(self.error_priors)), prune))
        )[0]
        check_matrix = self.check_matrix[:, keep]
        observables_matrix = self.observables_matrix[:, keep]
        error_priors = self.error_priors[keep]

        return CheckMatrices(
            check_matrix=check_matrix,
            observables_matrix=observables_matrix,
            error_priors=error_priors,
            syndrome_bias=syndrome_bias,
            observables_bias=observables_bias,
        )
