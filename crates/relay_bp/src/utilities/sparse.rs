// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ndarray::Array1;

use ndarray_npy::{NpzReader, ReadNpzError, ReadableElement};

use sprs::CsMat;
use std::fs::File;
use std::path::PathBuf;

pub fn load_sparse_npz<N: Clone + ReadableElement>(p: PathBuf) -> Result<CsMat<N>, ReadNpzError> {
    let mut npz = NpzReader::new(File::open(p).expect("Could not open file."))?;

    let shape: Array1<i64> = npz.by_name("shape")?;
    let data: Array1<N> = npz.by_name("data")?;
    let indices: Array1<i32> = npz.by_name("indices")?;
    let indptr: Array1<i32> = npz.by_name("indptr")?;

    Ok(CsMat::new(
        (shape[0].try_into().unwrap(), shape[1].try_into().unwrap()),
        indptr.mapv(|elem| elem as usize).to_vec(),
        indices.mapv(|elem| elem as usize).to_vec(),
        data.mapv(|elem| elem).to_vec(),
    ))
}
