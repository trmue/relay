// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

pub mod bp;
pub mod decoder;
pub mod observable_decoder;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn _relay_bp<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    decoder::init_decoder(_py, m)?;
    observable_decoder::init_observable_decoder(_py, m)?;
    bp::init_bp(_py, m)?;

    Ok(())
}
