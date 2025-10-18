use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use num_rs_core::buffer::MatrixBuffer;
use num_rs_core::sparse::{sparse_matmul, CsrMatrixView};

fn build_sample_csr(
    nnz_per_row: usize,
    rows: usize,
    cols: usize,
) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let mut row_ptr = Vec::with_capacity(rows + 1);
    row_ptr.push(0);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for row in 0..rows {
        for k in 0..nnz_per_row {
            let col = (row * nnz_per_row + k) % cols;
            col_idx.push(col as u32);
            values.push(((row + k) % 7) as f64 + 1.0);
        }
        row_ptr.push((col_idx.len()) as u32);
    }
    (row_ptr, col_idx, values)
}

fn dense_rhs(rows: usize, cols: usize) -> MatrixBuffer {
    let data: Vec<f64> = (0..rows * cols).map(|i| (i % 11) as f64).collect();
    MatrixBuffer::from_vec(data, rows, cols).expect("rhs")
}

fn bench_sparse_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_matmul_dense_rhs");
    for &(rows, cols, nnz) in &[(256usize, 128usize, 4usize), (512, 256, 8)] {
        let (row_ptr, col_idx, values) = build_sample_csr(nnz, rows, cols);
        let view =
            CsrMatrixView::new_f64(rows, cols, &row_ptr, &col_idx, &values).expect("csr view");
        let rhs = dense_rhs(cols, 1);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{rows}x{cols}_nnz{nnz}")),
            &rhs,
            |b, rhs| {
                b.iter(|| {
                    let _ = sparse_matmul(&view, rhs).expect("matmul");
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_sparse_matmul);
criterion_main!(benches);
