use napi::bindgen_prelude::*;
use napi::JsObject;
use napi_derive::napi;

use num_rs_core::buffer::MatrixBuffer;
use num_rs_core::dtype::DType;
use num_rs_core::{
    add as core_add, clip as core_clip, concat as core_concat, gather as core_gather,
    gather_pairs as core_gather_pairs, matmul as core_matmul, put as core_put, read_npy_matrix,
    scatter as core_scatter, scatter_pairs as core_scatter_pairs, stack as core_stack,
    take as core_take, where_select as core_where, where_select_multi as core_where_multi,
    write_npy_matrix,
};
#[cfg(feature = "linalg")]
use num_rs_core::{eigen as core_eigen, qr as core_qr, solve as core_solve, svd as core_svd};
use std::convert::TryFrom;

#[derive(Clone)]
#[napi]
pub struct Matrix {
    buffer: MatrixBuffer,
}

#[napi]
impl Matrix {
    #[napi(constructor)]
    pub fn new(data: Float64Array, rows: u32, cols: u32) -> Result<Self> {
        let data: Vec<f64> = data.to_vec();
        MatrixBuffer::from_vec(data, rows as usize, cols as usize)
            .map(|buffer| Matrix { buffer })
            .map_err(|e| Error::from_reason(e))
    }

    #[napi(factory)]
    pub fn from_bytes(data: Buffer, rows: u32, cols: u32, dtype: String) -> Result<Self> {
        let dtype: DType = dtype.parse().map_err(Error::from_reason)?;
        MatrixBuffer::from_bytes(dtype, rows as usize, cols as usize, data.to_vec())
            .map(Matrix::from_buffer)
            .map_err(|e| Error::from_reason(e))
    }

    #[napi(getter)]
    pub fn rows(&self) -> u32 {
        self.buffer.rows() as u32
    }

    #[napi(getter)]
    pub fn cols(&self) -> u32 {
        self.buffer.cols() as u32
    }

    #[napi(getter)]
    pub fn dtype(&self) -> String {
        self.buffer.dtype().as_str().to_string()
    }

    #[napi]
    pub fn astype(&self, dtype: String, copy: Option<bool>) -> Result<Self> {
        let dtype: DType = dtype.parse().map_err(Error::from_reason)?;
        let copy = copy.unwrap_or(false);
        if dtype == self.buffer.dtype() {
            if copy {
                let buffer = self
                    .buffer
                    .clone_with_dtype(dtype)
                    .map_err(Error::from_reason)?;
                return Ok(Matrix::from_buffer(buffer));
            }
            return Ok(Matrix::from_buffer(self.buffer.clone()));
        }
        if !copy && dtype.size_of() == self.buffer.dtype().size_of() {
            let buffer = self.buffer.reinterpret(dtype).map_err(Error::from_reason)?;
            return Ok(Matrix::from_buffer(buffer));
        }
        let buffer = self.buffer.cast(dtype).map_err(Error::from_reason)?;
        Ok(Matrix::from_buffer(buffer))
    }

    #[napi]
    pub fn to_vec(&self) -> Result<Float64Array> {
        Ok(Float64Array::from(self.buffer.to_f64_vec()))
    }

    #[napi]
    pub fn to_bytes(&self) -> Buffer {
        Buffer::from(self.buffer.to_contiguous_bytes_vec())
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

fn map_matrix(res: num_rs_core::CoreResult<MatrixBuffer>) -> Result<Matrix> {
    res.map(Matrix::from_buffer)
        .map_err(|err| Error::from_reason(err))
}

#[napi]
pub fn add(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    map_matrix(core_add(a.buffer(), b.buffer()))
}

#[napi]
pub fn matmul(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    map_matrix(core_matmul(a.buffer(), b.buffer()))
}

#[napi]
pub fn clip(matrix: &Matrix, min: f64, max: f64) -> Result<Matrix> {
    map_matrix(core_clip(matrix.buffer(), min, max))
}

#[napi]
pub fn where_select(condition: &Matrix, truthy: &Matrix, falsy: &Matrix) -> Result<Matrix> {
    map_matrix(core_where(
        condition.buffer(),
        truthy.buffer(),
        falsy.buffer(),
    ))
}

#[napi]
pub fn where_select_multi(
    conditions: Vec<&Matrix>,
    choices: Vec<&Matrix>,
    default_value: Option<&Matrix>,
) -> Result<Matrix> {
    let condition_refs: Vec<&MatrixBuffer> = conditions
        .into_iter()
        .map(|matrix| matrix.buffer())
        .collect();
    let choice_refs: Vec<&MatrixBuffer> =
        choices.into_iter().map(|matrix| matrix.buffer()).collect();
    let default_ref = default_value.map(|matrix| matrix.buffer());
    map_matrix(core_where_multi(&condition_refs, &choice_refs, default_ref))
}

#[napi]
pub fn concat(a: &Matrix, b: &Matrix, axis: u32) -> Result<Matrix> {
    let buffers = vec![a.buffer.clone(), b.buffer.clone()];
    map_matrix(core_concat(axis as usize, &buffers))
}

#[napi]
pub fn stack(a: &Matrix, b: &Matrix, axis: u32) -> Result<Matrix> {
    let buffers = vec![a.buffer.clone(), b.buffer.clone()];
    map_matrix(core_stack(axis as usize, &buffers))
}

#[napi]
pub fn take(matrix: &Matrix, axis: u32, indices: Vec<i64>) -> Result<Matrix> {
    let converted = convert_indices(&indices)?;
    map_matrix(core_take(matrix.buffer(), axis as usize, &converted))
}

#[napi]
pub fn put(matrix: &Matrix, axis: u32, indices: Vec<i64>, values: &Matrix) -> Result<Matrix> {
    let converted = convert_indices(&indices)?;
    map_matrix(core_put(
        matrix.buffer(),
        axis as usize,
        &converted,
        values.buffer(),
    ))
}

#[napi]
pub fn gather(matrix: &Matrix, row_indices: Vec<i64>, col_indices: Vec<i64>) -> Result<Matrix> {
    let rows = convert_indices(&row_indices)?;
    let cols = convert_indices(&col_indices)?;
    map_matrix(core_gather(matrix.buffer(), &rows, &cols))
}

#[napi]
pub fn gather_pairs(
    matrix: &Matrix,
    row_indices: Vec<i64>,
    col_indices: Vec<i64>,
) -> Result<Matrix> {
    let rows = convert_indices(&row_indices)?;
    let cols = convert_indices(&col_indices)?;
    map_matrix(core_gather_pairs(matrix.buffer(), &rows, &cols))
}

#[napi]
pub fn scatter(
    matrix: &Matrix,
    row_indices: Vec<i64>,
    col_indices: Vec<i64>,
    values: &Matrix,
) -> Result<Matrix> {
    let rows = convert_indices(&row_indices)?;
    let cols = convert_indices(&col_indices)?;
    map_matrix(core_scatter(matrix.buffer(), &rows, &cols, values.buffer()))
}

#[napi]
pub fn scatter_pairs(
    matrix: &Matrix,
    row_indices: Vec<i64>,
    col_indices: Vec<i64>,
    values: &Matrix,
) -> Result<Matrix> {
    let rows = convert_indices(&row_indices)?;
    let cols = convert_indices(&col_indices)?;
    map_matrix(core_scatter_pairs(
        matrix.buffer(),
        &rows,
        &cols,
        values.buffer(),
    ))
}

#[cfg(feature = "linalg")]
#[napi]
pub fn svd(env: Env, matrix: &Matrix) -> Result<JsObject> {
    let (u, s, vt) = core_svd(matrix.buffer()).map_err(|e| Error::from_reason(e))?;
    let mut obj = env.create_object()?;
    obj.set("u", Matrix::from_buffer(u))?;
    obj.set("s", Float64Array::from(s.to_f64_vec()))?;
    obj.set("vt", Matrix::from_buffer(vt))?;
    Ok(obj)
}

#[cfg(feature = "linalg")]
#[napi]
pub fn qr(env: Env, matrix: &Matrix) -> Result<JsObject> {
    let (q, r) = core_qr(matrix.buffer()).map_err(|e| Error::from_reason(e))?;
    let mut obj = env.create_object()?;
    obj.set("q", Matrix::from_buffer(q))?;
    obj.set("r", Matrix::from_buffer(r))?;
    Ok(obj)
}

#[cfg(feature = "linalg")]
#[napi]
pub fn solve(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    core_solve(a.buffer(), b.buffer())
        .map(Matrix::from_buffer)
        .map_err(|e| Error::from_reason(e))
}

#[cfg(feature = "linalg")]
#[napi]
pub fn eigen(env: Env, matrix: &Matrix) -> Result<JsObject> {
    let (values, vectors) = core_eigen(matrix.buffer()).map_err(|e| Error::from_reason(e))?;
    let mut obj = env.create_object()?;
    obj.set("values", Float64Array::from(values.to_f64_vec()))?;
    obj.set("vectors", Matrix::from_buffer(vectors))?;
    Ok(obj)
}

#[napi]
pub fn read_npy(buffer: Buffer) -> Result<Matrix> {
    read_npy_matrix(&buffer)
        .map(Matrix::from_buffer)
        .map_err(|e| Error::from_reason(e))
}

#[napi]
pub fn write_npy(matrix: &Matrix) -> Result<Buffer> {
    write_npy_matrix(matrix.buffer())
        .map(Buffer::from)
        .map_err(|e| Error::from_reason(e))
}

#[napi]
pub fn copy_bytes_total() -> f64 {
    num_rs_core::copy_bytes_total() as f64
}

#[napi]
pub fn take_copy_bytes() -> f64 {
    num_rs_core::take_copy_bytes() as f64
}

#[napi]
pub fn reset_copy_bytes() {
    num_rs_core::reset_copy_bytes();
}

fn convert_indices(indices: &[i64]) -> Result<Vec<isize>> {
    indices
        .iter()
        .map(|value| {
            isize::try_from(*value)
                .map_err(|_| Error::from_reason(format!("index {value} exceeds platform limits")))
        })
        .collect()
}
