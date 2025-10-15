use num_rs_core::buffer::MatrixBuffer;
use num_rs_core::dtype::DType;
use num_rs_core::{
    add as core_add, clip as core_clip, concat as core_concat, gather as core_gather,
    gather_pairs as core_gather_pairs, matmul as core_matmul, put as core_put,
    scatter as core_scatter, scatter_pairs as core_scatter_pairs, stack as core_stack,
    take as core_take, where_select as core_where,
};
use std::convert::TryFrom;
use std::str::FromStr;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Clone)]
pub struct Matrix {
    buffer: MatrixBuffer,
}

#[wasm_bindgen]
impl Matrix {
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<f64>, rows: usize, cols: usize) -> Matrix {
        let buffer = MatrixBuffer::from_vec(data, rows, cols).expect("invalid matrix shape");
        Matrix { buffer }
    }

    #[wasm_bindgen(js_name = "from_bytes")]
    pub fn from_bytes(
        data: Vec<u8>,
        rows: usize,
        cols: usize,
        dtype: String,
    ) -> Result<Matrix, JsValue> {
        let dtype = DType::from_str(&dtype).map_err(|err| JsValue::from_str(&err))?;
        MatrixBuffer::from_bytes(dtype, rows, cols, data)
            .map(Matrix::from_buffer)
            .map_err(|err| JsValue::from_str(&err))
    }

    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> usize {
        self.buffer.rows()
    }

    #[wasm_bindgen(getter)]
    pub fn cols(&self) -> usize {
        self.buffer.cols()
    }

    #[wasm_bindgen(getter)]
    pub fn dtype(&self) -> String {
        self.buffer.dtype().as_str().to_string()
    }

    #[wasm_bindgen]
    pub fn astype(&self, dtype: String, copy: Option<bool>) -> Result<Matrix, JsValue> {
        let dtype = DType::from_str(&dtype).map_err(|err| JsValue::from_str(&err))?;
        let copy = copy.unwrap_or(false);
        if dtype == self.buffer.dtype() {
            if copy {
                let buffer = self
                    .buffer
                    .clone_with_dtype(dtype)
                    .map_err(|err| JsValue::from_str(&err))?;
                return Ok(Matrix::from_buffer(buffer));
            }
            return Ok(Matrix::from_buffer(self.buffer.clone()));
        }
        if !copy && dtype.size_of() == self.buffer.dtype().size_of() {
            let buffer = self
                .buffer
                .reinterpret(dtype)
                .map_err(|err| JsValue::from_str(&err))?;
            return Ok(Matrix::from_buffer(buffer));
        }
        let buffer = self
            .buffer
            .cast(dtype)
            .map_err(|err| JsValue::from_str(&err))?;
        Ok(Matrix::from_buffer(buffer))
    }

    #[wasm_bindgen]
    pub fn to_vec(&self) -> Vec<f64> {
        self.buffer.to_f64_vec()
    }

    #[wasm_bindgen(js_name = "to_bytes")]
    pub fn to_bytes(&self) -> Vec<u8> {
        self.buffer.to_contiguous_bytes_vec()
    }
}

impl Matrix {
    fn from_buffer(buffer: MatrixBuffer) -> Self {
        Matrix { buffer }
    }

    fn buffer(&self) -> &MatrixBuffer {
        &self.buffer
    }
}

fn map_matrix(res: num_rs_core::CoreResult<MatrixBuffer>) -> Result<Matrix, JsValue> {
    res.map(Matrix::from_buffer)
        .map_err(|err| JsValue::from_str(&err))
}

fn convert_indices(indices: &[i32]) -> Result<Vec<isize>, JsValue> {
    indices
        .iter()
        .map(|value| {
            isize::try_from(*value)
                .map_err(|_| JsValue::from_str(&format!("index {value} exceeds platform limits")))
        })
        .collect()
}

#[wasm_bindgen]
pub fn add(a: &Matrix, b: &Matrix) -> Result<Matrix, JsValue> {
    map_matrix(core_add(a.buffer(), b.buffer()))
}

#[wasm_bindgen]
pub fn matmul(a: &Matrix, b: &Matrix) -> Result<Matrix, JsValue> {
    map_matrix(core_matmul(a.buffer(), b.buffer()))
}

#[wasm_bindgen]
pub fn clip(matrix: &Matrix, min: f64, max: f64) -> Result<Matrix, JsValue> {
    map_matrix(core_clip(matrix.buffer(), min, max))
}

#[wasm_bindgen]
pub fn where_select(
    condition: &Matrix,
    truthy: &Matrix,
    falsy: &Matrix,
) -> Result<Matrix, JsValue> {
    map_matrix(core_where(
        condition.buffer(),
        truthy.buffer(),
        falsy.buffer(),
    ))
}

#[wasm_bindgen]
pub fn concat(a: &Matrix, b: &Matrix, axis: usize) -> Result<Matrix, JsValue> {
    let buffers = vec![a.buffer.clone(), b.buffer.clone()];
    map_matrix(core_concat(axis, &buffers))
}

#[wasm_bindgen]
pub fn stack(a: &Matrix, b: &Matrix, axis: usize) -> Result<Matrix, JsValue> {
    let buffers = vec![a.buffer.clone(), b.buffer.clone()];
    map_matrix(core_stack(axis, &buffers))
}

#[wasm_bindgen]
pub fn take(matrix: &Matrix, axis: usize, indices: Vec<i32>) -> Result<Matrix, JsValue> {
    let converted = convert_indices(&indices)?;
    map_matrix(core_take(matrix.buffer(), axis, &converted))
}

#[wasm_bindgen]
pub fn put(
    matrix: &Matrix,
    axis: usize,
    indices: Vec<i32>,
    values: &Matrix,
) -> Result<Matrix, JsValue> {
    let converted = convert_indices(&indices)?;
    map_matrix(core_put(matrix.buffer(), axis, &converted, values.buffer()))
}

#[wasm_bindgen]
pub fn gather(
    matrix: &Matrix,
    row_indices: Vec<i32>,
    col_indices: Vec<i32>,
) -> Result<Matrix, JsValue> {
    let rows = convert_indices(&row_indices)?;
    let cols = convert_indices(&col_indices)?;
    map_matrix(core_gather(matrix.buffer(), &rows, &cols))
}

#[wasm_bindgen(js_name = "gather_pairs")]
pub fn gather_pairs(
    matrix: &Matrix,
    row_indices: Vec<i32>,
    col_indices: Vec<i32>,
) -> Result<Matrix, JsValue> {
    let rows = convert_indices(&row_indices)?;
    let cols = convert_indices(&col_indices)?;
    map_matrix(core_gather_pairs(matrix.buffer(), &rows, &cols))
}

#[wasm_bindgen]
pub fn scatter(
    matrix: &Matrix,
    row_indices: Vec<i32>,
    col_indices: Vec<i32>,
    values: &Matrix,
) -> Result<Matrix, JsValue> {
    let rows = convert_indices(&row_indices)?;
    let cols = convert_indices(&col_indices)?;
    map_matrix(core_scatter(matrix.buffer(), &rows, &cols, values.buffer()))
}

#[wasm_bindgen(js_name = "scatter_pairs")]
pub fn scatter_pairs(
    matrix: &Matrix,
    row_indices: Vec<i32>,
    col_indices: Vec<i32>,
    values: &Matrix,
) -> Result<Matrix, JsValue> {
    let rows = convert_indices(&row_indices)?;
    let cols = convert_indices(&col_indices)?;
    map_matrix(core_scatter_pairs(
        matrix.buffer(),
        &rows,
        &cols,
        values.buffer(),
    ))
}

#[wasm_bindgen]
pub fn copy_bytes_total() -> f64 {
    num_rs_core::copy_bytes_total() as f64
}

#[wasm_bindgen]
pub fn take_copy_bytes() -> f64 {
    num_rs_core::take_copy_bytes() as f64
}

#[wasm_bindgen]
pub fn reset_copy_bytes() {
    num_rs_core::reset_copy_bytes();
}
