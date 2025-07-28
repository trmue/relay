// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::path::PathBuf;

/// Get the test manifest path from the Cargo manifest
pub fn get_cargo_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Get the test manifest path from the Cargo manifest
pub fn get_test_data_path() -> PathBuf {
    let mut resources = get_cargo_path();
    resources.push("data");
    resources
}
