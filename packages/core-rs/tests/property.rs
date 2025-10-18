use num_rs_core::{add as core_add, buffer::MatrixBuffer, matmul as core_matmul};
use proptest::prelude::*;

fn matrix_from_vec(data: Vec<f64>, rows: usize, cols: usize) -> MatrixBuffer {
    MatrixBuffer::from_vec(data, rows, cols).expect("matrix shape mismatch")
}

prop_compose! {
    fn small_matrix()(rows in 1usize..5, cols in 1usize..5,
                      values in prop::collection::vec(-10f64..10f64, 1..=25))
                      -> (MatrixBuffer, usize, usize) {
        let total = rows * cols;
        let data = values.into_iter().cycle().take(total).collect::<Vec<_>>();
        (matrix_from_vec(data, rows, cols), rows, cols)
    }
}

prop_compose! {
    fn compatible_matrices()(m in 1usize..4, k in 1usize..4, n in 1usize..4,
                             left_values in prop::collection::vec(-5f64..5f64, 1..=16),
                             right_values in prop::collection::vec(-5f64..5f64, 1..=16))
                             -> (MatrixBuffer, MatrixBuffer, usize, usize, usize) {
        let lhs = left_values.into_iter().cycle().take(m * k).collect::<Vec<_>>();
        let rhs = right_values.into_iter().cycle().take(k * n).collect::<Vec<_>>();
        (
            matrix_from_vec(lhs, m, k),
            matrix_from_vec(rhs, k, n),
            m,
            k,
            n,
        )
    }
}

proptest! {
    #[test]
    fn add_matches_reference((matrix, rows, cols) in small_matrix(),
                             second in prop::collection::vec(-10f64..10f64, 1..=25)) {
        let rhs = second.into_iter().cycle().take(rows * cols).collect::<Vec<_>>();
        let rhs_matrix = matrix_from_vec(rhs.clone(), rows, cols);
        let sum = core_add(&matrix, &rhs_matrix).expect("add");
        let expected = matrix
            .to_f64_vec()
            .into_iter()
            .zip(rhs.into_iter())
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        prop_assert!(sum
            .to_f64_vec()
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| (a - b).abs() < 1e-9));
    }

    #[test]
    fn matmul_matches_reference(data in compatible_matrices()) {
        let (lhs, rhs, m, k, n) = data;
        let product = core_matmul(&lhs, &rhs).expect("matmul");
        let lhs_vec = lhs.to_f64_vec();
        let rhs_vec = rhs.to_f64_vec();
        let product_vec = product.to_f64_vec();
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0f64;
                for shared in 0..k {
                    let a = lhs_vec[row * k + shared];
                    let b = rhs_vec[shared * n + col];
                    acc += a * b;
                }
                let idx = row * n + col;
                prop_assert!((product_vec[idx] - acc).abs() < 1e-9);
            }
        }
    }
}
