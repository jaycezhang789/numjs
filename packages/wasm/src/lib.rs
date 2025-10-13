use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use num_rs_core::{
    add_inplace, clip as core_clip, concat as core_concat, matmul as core_matmul,
    stack_matrices as core_stack, where_select as core_where, CoreResult,
};
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

impl Matrix {
    fn from_array(array: Array2<f64>) -> Matrix {
        let (rows, cols) = array.dim();
        Matrix {
            data: array.into_raw_vec(),
            rows,
            cols,
        }
    }

    fn empty_like(rows: usize, cols: usize) -> Matrix {
        Matrix {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    fn view(&self) -> ArrayView2<'_, f64> {
        ArrayView2::from_shape((self.rows, self.cols), &self.data).unwrap()
    }

    fn view_mut(&mut self) -> ArrayViewMut2<'_, f64> {
        ArrayViewMut2::from_shape((self.rows, self.cols), &mut self.data).unwrap()
    }
}

fn map_core<T>(res: CoreResult<T>) -> Result<T, JsValue> {
    res.map_err(|err| JsValue::from_str(&err))
}

#[wasm_bindgen]
pub fn add(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    let mut out = Matrix::empty_like(a.rows, a.cols);
    add_inplace(a.view(), b.view(), out.view_mut());
    out
}

#[wasm_bindgen]
pub fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols, b.rows);
    let c: Array2<f64> = core_matmul(a.view(), b.view());
    Matrix::from_array(c)
}

#[wasm_bindgen]
pub fn clip(matrix: &Matrix, min: f64, max: f64) -> Matrix {
    let clipped = core_clip(matrix.view(), min, max);
    Matrix::from_array(clipped)
}

#[wasm_bindgen]
pub fn where_select(
    condition: &Matrix,
    truthy: &Matrix,
    falsy: &Matrix,
) -> Result<Matrix, JsValue> {
    let selected = map_core(core_where(condition.view(), truthy.view(), falsy.view()))?;
    Ok(Matrix::from_array(selected))
}

#[wasm_bindgen]
pub fn concat(a: &Matrix, b: &Matrix, axis: usize) -> Result<Matrix, JsValue> {
    if axis > 1 {
        return Err(JsValue::from_str("Axis out of bounds for concat"));
    }
    let views = [a.view(), b.view()];
    let concatenated = map_core(core_concat(axis, &views))?;
    Ok(Matrix::from_array(concatenated))
}

#[wasm_bindgen]
pub fn stack(a: &Matrix, b: &Matrix, axis: usize) -> Result<Matrix, JsValue> {
    if axis > 1 {
        return Err(JsValue::from_str("Axis out of bounds for stack"));
    }
    let views = [a.view(), b.view()];
    let stacked = map_core(core_stack(axis, &views))?;
    Ok(Matrix::from_array(stacked))
}
