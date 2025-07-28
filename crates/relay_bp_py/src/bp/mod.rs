// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

pub mod min_sum;
pub mod min_sum_fixed;
pub mod relay;

use pyo3::prelude::*;
use pyo3::{Bound, PyResult};

/// A Python module implemented in Rust.
#[pymodule]
pub fn _bp<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<min_sum::MinSumBPDecoderF32>()?;
    m.add_class::<min_sum::MinSumBPDecoderF64>()?;
    m.add_class::<min_sum::MinSumBPDecoderI8>()?;
    m.add_class::<min_sum::MinSumBPDecoderI16>()?;
    m.add_class::<min_sum::MinSumBPDecoderI32>()?;
    m.add_class::<min_sum::MinSumBPDecoderI64>()?;
    m.add_class::<min_sum_fixed::MinSumBPDecoderFixed>()?;
    m.add_class::<relay::RelayDecoderF32>()?;
    m.add_class::<relay::RelayDecoderF64>()?;
    m.add_class::<relay::RelayDecoderI32>()?;
    m.add_class::<relay::RelayDecoderI64>()?;
    Ok(())
}

pub fn init_bp<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    // Workaround for https://github.com/PyO3/pyo3/issues/759
    let bp_module = PyModule::new(_py, "_relay_bp._bp")?;

    _bp(_py, &bp_module)?;

    m.add("_bp", &bp_module)?;
    bp_module.setattr("__name__", "_bp")?;
    _py.import("sys")?
        .getattr("modules")?
        .set_item("_relay_bp._bp", &bp_module)?;
    Ok(())
}
