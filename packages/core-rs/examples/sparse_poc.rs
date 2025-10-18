use num_rs_core::buffer::MatrixBuffer;
use num_rs_core::sparse::{self, CsrMatrixView};
use sprs::{CsMat, CsVec};

fn main() {
    // Construct a simple CSR matrix using sprs.
    // Matrix:
    // [1.0 0.0 2.0]
    // [0.0 3.0 0.0]
    // [4.0 0.0 5.0]
    let indptr = vec![0, 2, 3, 5];
    let indices = vec![0, 2, 1, 0, 2];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let csr: CsMat<f64> = CsMat::new((3, 3), indptr, indices, data);

    // Convert the sprs matrix into a num_rs_core CSR view.
    let view = CsrMatrixView::new_f64(
        csr.rows(),
        csr.cols(),
        csr.indptr().as_slice(),
        csr.indices(),
        csr.data(),
    )
    .expect("valid CSR");

    // Dense vector for multiplication.
    let rhs = MatrixBuffer::from_vec(vec![1.0f64, 2.0, 3.0], 3, 1).expect("vector");

    // Run the fallback sparse matmul and addition helpers.
    let product = sparse::sparse_matmul(&view, &rhs).expect("sparse matmul");
    let sum = sparse::sparse_add(&view, &product).expect("sparse add");

    // Validate against sprs dense multiplication.
    let expected_vec: CsVec<f64> = &csr * sprs::CsVec::new(3, vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
    let expected_dense = expected_vec.to_dense();

    println!(
        "sprs result : {:?}",
        expected_dense.iter().collect::<Vec<_>>()
    );
    println!(
        "num-rs dense result: {:?}",
        product
            .to_f64_vec()
            .into_iter()
            .collect::<Vec<f64>>()
    );
    println!(
        "sparse add (A + Ax): {:?}",
        sum.to_f64_vec().into_iter().collect::<Vec<f64>>()
    );
}
