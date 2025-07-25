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
            error_priors = dem_matrices.hyperedge_to_edge_matrix @ dem_matrices.priors
        else:
            observables_matrix = dem_matrices.observables_matrix
            check_matrix = dem_matrices.check_matrix
            error_priors = dem_matrices.priors

        check_matrices = cls(
            check_matrix=check_matrix,
            observables_matrix=observables_matrix,
            error_priors=error_priors.astype(np.float64),
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
            syndrome_bias = np.sum(self.check_matrix[:, to_one], axis=1).reshape(-1) % 2
            observables_bias = (
                np.sum(self.check_matrix[:, to_one], axis=1).reshape(-1) % 2
            )
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
