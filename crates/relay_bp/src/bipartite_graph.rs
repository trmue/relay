// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ndarray::Array2;
use num_traits::Zero;
use sprs::{CompressedStorage, CsMat, TriMat};
use std::ops::Add;

pub trait BipartiteGraph<N: PartialEq>: PartialEq {
    fn from_dense(array: Array2<N>) -> Self;

    /// Number of rows (or check nodes)
    fn rows(&self) -> usize;
    /// Number of cols (or variable nodes)
    fn cols(&self) -> usize;
    /// Insert an element in the bipartite graph. If the element is already present, its value is overwritten.
    fn insert(&mut self, row: usize, col: usize, val: N);
    /// Access the element located at row (check node) i and column (variable node) j.
    /// Will return None if there is no non-zero element at this location.
    fn get(&self, row: usize, col: usize) -> Option<&N>;
}

/// A fast implementation of a bipartite graph which is directly
/// using a CsMat.
/// TODO: In the future if this is an issue we might wrap the CsMat
/// in a struct.
pub type SparseBipartiteGraph<N> = CsMat<N>;

impl<N: PartialEq + Clone + Zero + Add<Output = N>> BipartiteGraph<N> for SparseBipartiteGraph<N> {
    fn rows(&self) -> usize {
        self.rows()
    }
    fn cols(&self) -> usize {
        self.cols()
    }
    /// Access the element located at row (check node) i and column (variable node) j.
    /// Will return None if there is no non-zero element at this location.
    /// Warning: This will only be an efficient operation if it follows the internal
    /// sparse matrix's ordering.
    fn insert(&mut self, row: usize, col: usize, val: N) {
        self.insert(row, col, val)
    }

    /// This access is logarithmic in the number of non-zeros in the corresponding outer slice. It is therefore advisable not to rely on this for algorithms, and prefer outer_iterator which accesses elements in storage order.
    fn get(&self, row: usize, col: usize) -> Option<&N> {
        self.get(row, col)
    }

    fn from_dense(array: Array2<N>) -> Self {
        sparse_bartite_graph_from_dense(array, CompressedStorage::CSR)
    }
}

pub fn sparse_bartite_graph_from_dense<N: PartialEq + Clone + Zero + Add<Output = N>>(
    array: Array2<N>,
    storage: CompressedStorage,
) -> SparseBipartiteGraph<N> {
    let mut tri_graph = TriMat::new((array.shape()[0], array.shape()[1]));
    for ((x, y), value) in array.indexed_iter() {
        if !(*value).is_zero() {
            tri_graph.add_triplet(x, y, value.clone());
        }
    }
    match storage {
        CompressedStorage::CSR => tri_graph.to_csr(),
        CompressedStorage::CSC => tri_graph.to_csc(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_sparse_bipartite_graph() {
        let graph: SparseBipartiteGraph<f64> = CsMat::eye(100);
        assert_eq!(*graph.get(99, 99).unwrap(), 1.0);
    }
}
