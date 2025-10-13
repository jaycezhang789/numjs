use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use num_rs_core::{add_inplace, matmul as core_matmul};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

#[wasm_bindgen]
impl Matrix {
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<f64>, rows: usize, cols: usize) -> Matrix {
        assert_eq!(data.len(), rows * cols);
        Matrix { data, rows, cols }
    }

    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> usize {
        self.rows
    }
    #[wasm_bindgen(getter)]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[wasm_bindgen]
    pub fn to_vec(&self) -> Vec<f64> {
        self.data.clone()
    }
}

fn view2(a: &Matrix) -> ArrayView2<'_, f64> {
    ArrayView2::from_shape((a.rows, a.cols), &a.data).unwrap()
}

fn viewmut2(out: &mut Matrix) -> ArrayViewMut2<'_, f64> {
    ArrayViewMut2::from_shape((out.rows, out.cols), &mut out.data).unwrap()
}

#[wasm_bindgen]
pub fn add(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    let mut out = Matrix::new(vec![0.0; a.rows * a.cols], a.rows, a.cols);
    add_inplace(view2(a), view2(b), viewmut2(&mut out));
    out
}

#[wasm_bindgen]
pub fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols, b.rows);
    let c: Array2<f64> = core_matmul(view2(a), view2(b));
    Matrix {
        data: c.into_raw_vec(),
        rows: a.rows,
        cols: b.cols,
    }
}
