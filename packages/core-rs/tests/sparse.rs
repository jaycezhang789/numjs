use num_rs_core::buffer::MatrixBuffer;
use num_rs_core::sparse::{
    sparse_add, sparse_matmul, sparse_matvec, sparse_transpose, CsrMatrixView,
};
use num_rs_core::{add as core_add, matmul as core_matmul, transpose as core_transpose};

fn sample_csr() -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    // 3x3 matrix
    // [1, 0, 2]
    // [0, 3, 0]
    // [4, 0, 5]
    (
        vec![0, 2, 3, 5],
        vec![0, 2, 1, 0, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
    )
}

fn dense_rhs() -> MatrixBuffer {
    MatrixBuffer::from_vec(vec![1.0f64, 2.0, 3.0], 3, 1).expect("vector")
}

#[test]
fn csr_to_dense_roundtrip() {
    let (row_ptr, col_idx, values) = sample_csr();
    let view = CsrMatrixView::new_f64(3, 3, &row_ptr, &col_idx, &values).expect("csr view");
    let dense = view.to_dense().expect("dense");
    assert_eq!(dense.rows(), 3);
    assert_eq!(dense.cols(), 3);
    assert_eq!(
        dense.to_f64_vec(),
        vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0]
    );
}

#[test]
fn csr_validate_rejects_bad_row_ptr() {
    let row_ptr = vec![0, 1, 1]; // length should be rows + 1 (3 + 1)
    let col_idx = vec![0];
    let values = vec![1.0];
    let err = CsrMatrixView::new_f64(3, 3, &row_ptr, &col_idx, &values);
    assert!(err.err().unwrap().contains("rowPtr"));
}

#[test]
fn csr_validate_rejects_col_bounds() {
    let row_ptr = vec![0, 1, 1, 1];
    let col_idx = vec![5]; // out of bounds for 3 columns
    let values = vec![1.0];
    let err = CsrMatrixView::new_f64(3, 3, &row_ptr, &col_idx, &values);
    assert!(err.err().unwrap().contains("colIdx"));
}

#[test]
fn sparse_matmul_matches_dense() {
    let (row_ptr, col_idx, values) = sample_csr();
    let view = CsrMatrixView::new_f64(3, 3, &row_ptr, &col_idx, &values).expect("csr view");
    let rhs = dense_rhs();

    let sparse = sparse_matmul(&view, &rhs).expect("sparse matmul");
    let dense = {
        let lhs = view.to_dense().expect("dense lhs");
        core_matmul(&lhs, &rhs).expect("dense matmul")
    };

    assert_eq!(sparse.to_f64_vec(), dense.to_f64_vec());
}

#[test]
fn sparse_matvec_matches_dense() {
    let (row_ptr, col_idx, values) = sample_csr();
    let view = CsrMatrixView::new_f64(3, 3, &row_ptr, &col_idx, &values).expect("csr view");
    let rhs = dense_rhs();

    let sparse = sparse_matvec(&view, &rhs).expect("sparse matvec");
    let dense = {
        let lhs = view.to_dense().expect("dense lhs");
        core_matmul(&lhs, &rhs).expect("dense matvec")
    };

    assert_eq!(sparse.to_f64_vec(), dense.to_f64_vec());
}

#[test]
fn sparse_add_matches_dense() {
    let (row_ptr, col_idx, values) = sample_csr();
    let view = CsrMatrixView::new_f64(3, 3, &row_ptr, &col_idx, &values).expect("csr view");
    let rhs = view.to_dense().expect("dense");

    let sparse = sparse_add(&view, &rhs).expect("sparse add");
    let dense_sum = {
        let lhs_dense = view.to_dense().expect("lhs");
        core_add(&lhs_dense, &rhs).expect("dense add")
    };

    assert_eq!(sparse.to_f64_vec(), dense_sum.to_f64_vec());
}

#[test]
fn sparse_add_shape_mismatch() {
    let (row_ptr, col_idx, values) = sample_csr();
    let view = CsrMatrixView::new_f64(3, 3, &row_ptr, &col_idx, &values).expect("csr view");
    let rhs = MatrixBuffer::from_vec(vec![1.0f64, 2.0], 2, 1).expect("rhs");
    let err = sparse_add(&view, &rhs).unwrap_err();
    assert!(err.contains("shape"));
}

#[test]
fn sparse_transpose_matches_dense() {
    let (row_ptr, col_idx, values) = sample_csr();
    let view = CsrMatrixView::new_f64(3, 3, &row_ptr, &col_idx, &values).expect("csr view");

    let sparse = sparse_transpose(&view).expect("sparse transpose");
    let dense = {
        let lhs = view.to_dense().expect("dense lhs");
        core_transpose(&lhs).expect("dense transpose")
    };

    assert_eq!(sparse.to_f64_vec(), dense.to_f64_vec());
}

#[cfg(feature = "sparse-native")]
#[test]
fn sparse_native_matches_dense() {
    let (row_ptr, col_idx, values) = sample_csr();
    let view = CsrMatrixView::new_f64(3, 3, &row_ptr, &col_idx, &values).expect("csr view");
    let rhs = dense_rhs();

    let sparse = sparse_matmul(&view, &rhs).expect("native matmul");
    let dense = {
        let lhs = view.to_dense().expect("dense lhs");
        core_matmul(&lhs, &rhs).expect("dense matmul")
    };

    assert_eq!(sparse.to_f64_vec(), dense.to_f64_vec());
}

#[cfg(feature = "sparse-native")]
#[test]
fn sparse_native_matvec_matches_dense() {
    let (row_ptr, col_idx, values) = sample_csr();
    let view = CsrMatrixView::new_f64(3, 3, &row_ptr, &col_idx, &values).expect("csr view");
    let rhs = dense_rhs();

    let sparse = sparse_matvec(&view, &rhs).expect("native matvec");
    let dense = {
        let lhs = view.to_dense().expect("dense lhs");
        core_matmul(&lhs, &rhs).expect("dense matvec")
    };

    assert_eq!(sparse.to_f64_vec(), dense.to_f64_vec());
}

#[cfg(feature = "sparse-suitesparse")]
#[test]
fn sparse_suitesparse_matches_dense() {
    let (row_ptr, col_idx, values) = sample_csr();
    let view = CsrMatrixView::new_f64(3, 3, &row_ptr, &col_idx, &values).expect("csr view");
    let rhs = dense_rhs();

    let sparse = sparse_matmul(&view, &rhs).expect("suitesparse matmul");
    let dense = {
        let lhs = view.to_dense().expect("dense lhs");
        core_matmul(&lhs, &rhs).expect("dense matmul")
    };

    assert_eq!(sparse.to_f64_vec(), dense.to_f64_vec());
}

#[cfg(feature = "sparse-suitesparse")]
#[test]
fn sparse_suitesparse_matvec_matches_dense() {
    let (row_ptr, col_idx, values) = sample_csr();
    let view = CsrMatrixView::new_f64(3, 3, &row_ptr, &col_idx, &values).expect("csr view");
    let rhs = dense_rhs();

    let sparse = sparse_matvec(&view, &rhs).expect("suitesparse matvec");
    let dense = {
        let lhs = view.to_dense().expect("dense lhs");
        core_matmul(&lhs, &rhs).expect("dense matvec")
    };

    assert_eq!(sparse.to_f64_vec(), dense.to_f64_vec());
}
