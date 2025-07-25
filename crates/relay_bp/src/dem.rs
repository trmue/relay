// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::decoder::SparseBitMatrix;
use crate::utilities::sparse::load_sparse_npz;
use ndarray_npy::{read_npy, ReadNpzError};
use std::path::PathBuf;

use ndarray::Array1;

pub struct DetectorErrorModel {
    /// Decoding matrix stored in csc format for performance.
    pub detector_error_matrix: SparseBitMatrix,
    /// Decoding matrix stored in csc format for performance.
    pub observable_error_matrix: SparseBitMatrix,
    pub error_priors: Array1<f64>,
}

impl DetectorErrorModel {
    pub fn new(
        detector_error_matrix: SparseBitMatrix,
        observable_error_matrix: SparseBitMatrix,
        error_priors: Array1<f64>,
    ) -> Self {
        DetectorErrorModel {
            detector_error_matrix: detector_error_matrix.into_csc(),
            observable_error_matrix: observable_error_matrix.into_csc(),
            error_priors,
        }
    }

    // Load from disk given a path and a prefix name.
    pub fn load(p: PathBuf) -> Result<Self, ReadNpzError> {
        let mut path_components = p.components();
        // Remove empty
        let code_name = path_components.next_back().unwrap();
        let path_no_file = path_components.as_path();

        let detector_error_matrix_name_name =
            format!("{}_Hdec.npz", code_name.as_os_str().to_str().unwrap());
        let detector_error_matrix =
            load_sparse_npz(path_no_file.join(detector_error_matrix_name_name))?.into_csc();

        let observable_error_matrix_name_name =
            format!("{}_Adec.npz", code_name.as_os_str().to_str().unwrap());
        let observable_error_matrix =
            load_sparse_npz(path_no_file.join(observable_error_matrix_name_name))?.into_csc();

        let prior_name = format!(
            "{}_error_priors.npy",
            code_name.as_os_str().to_str().unwrap()
        );
        let error_priors: Array1<f32> = read_npy(path_no_file.join(prior_name))?;

        Ok(DetectorErrorModel {
            detector_error_matrix,
            observable_error_matrix,
            error_priors: error_priors.mapv(|elem| elem as f64),
        })
    }
}
