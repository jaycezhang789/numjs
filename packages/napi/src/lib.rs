use napi::bindgen_prelude::{
    BigInt64Array, Buffer, Env, Error, Float64Array, Result, TypedArrayType,
};
use napi::JsObject;
use napi_derive::napi;

use num_rs_core::buffer::{CastOptions, CastingKind, MatrixBuffer, SliceSpec};
use num_rs_core::compress::compress as core_compress;
use num_rs_core::dtype::DType;
use num_rs_core::gpu::{self, GpuBackendKind as CoreGpuBackendKind};
use num_rs_core::{
    add as core_add, broadcast_to as core_broadcast_to, clip as core_clip, concat as core_concat,
    cos as core_cos, div as core_div, dot as core_dot, exp as core_exp, gather as core_gather,
    gather_pairs as core_gather_pairs, log as core_log, matmul as core_matmul,
    median as core_median, mul as core_mul, nanmean as core_nanmean, nansum as core_nansum,
    neg as core_neg, percentile as core_percentile, put as core_put, quantile as core_quantile,
    read_npy_matrix, scatter as core_scatter, scatter_pairs as core_scatter_pairs,
    sigmoid as core_sigmoid, sin as core_sin, stack as core_stack, sub as core_sub,
    sum as core_sum, take as core_take, tanh as core_tanh, transpose as core_transpose,
    where_select as core_where, where_select_multi as core_where_multi, write_npy_matrix,
};
#[cfg(feature = "linalg")]
use num_rs_core::{eigen as core_eigen, qr as core_qr, solve as core_solve, svd as core_svd};
use std::convert::TryFrom;
use std::ptr;
use std::sync::Arc;

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
            .map_err(map_core_error)
    }

    #[napi(factory)]
    pub fn from_bytes(data: Buffer, rows: u32, cols: u32, dtype: String) -> Result<Self> {
        let dtype: DType = dtype.parse().map_err(map_core_error)?;
        if dtype == DType::Fixed64 {
            return Err(map_core_error(
                "from_bytes(Fixed64): supply scaled bigint data via Matrix.from_fixed_i64",
            ));
        }
        MatrixBuffer::from_bytes(dtype, rows as usize, cols as usize, data.to_vec())
            .map(Matrix::from_buffer)
            .map_err(map_core_error)
    }

    #[napi(factory)]
    pub fn from_fixed_i64(data: BigInt64Array, rows: u32, cols: u32, scale: i32) -> Result<Self> {
        let vec = data.to_vec();
        MatrixBuffer::from_fixed_i64_vec(vec, rows as usize, cols as usize, scale)
            .map(Matrix::from_buffer)
            .map_err(map_core_error)
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

    #[napi(getter)]
    pub fn fixed_scale(&self) -> Option<i32> {
        self.buffer.fixed_scale()
    }

    #[napi]
    pub fn astype(
        &self,
        dtype: String,
        copy: Option<bool>,
        casting: Option<String>,
    ) -> Result<Self> {
        let dtype: DType = dtype.parse().map_err(map_core_error)?;
        let copy = copy.unwrap_or(false);
        let options = CastOptions::parse(casting.as_deref()).map_err(map_core_error)?;
        if dtype == self.buffer.dtype() {
            if copy {
                let buffer = self
                    .buffer
                    .clone_with_dtype(dtype)
                    .map_err(map_core_error)?;
                return Ok(Matrix::from_buffer(buffer));
            }
            return Ok(Matrix::from_buffer(self.buffer.clone()));
        }
        if !copy
            && matches!(options.casting(), CastingKind::Unsafe)
            && dtype.size_of() == self.buffer.dtype().size_of()
        {
            let buffer = self.buffer.reinterpret(dtype).map_err(map_core_error)?;
            return Ok(Matrix::from_buffer(buffer));
        }
        let buffer = self
            .buffer
            .cast_with_options(dtype, &options)
            .map_err(map_core_error)?;
        Ok(Matrix::from_buffer(buffer))
    }

    #[napi]
    pub fn to_vec(&self) -> Result<Float64Array> {
        Ok(Float64Array::from(self.buffer.to_f64_vec()))
    }

    #[napi]
    pub fn to_bytes(&self, env: Env) -> Result<JsObject> {
        let (arc, offset_bytes, len_bytes) = match self.buffer.try_as_byte_arc() {
            Some(view) => view,
            None => {
                let owned = Arc::new(self.buffer.to_contiguous_bytes_vec());
                let len = owned.len();
                (owned, 0usize, len)
            }
        };

        let holder = arc.clone();
        let ptr = if len_bytes == 0 {
            ptr::null_mut()
        } else {
            unsafe { Arc::as_ref(&arc).as_ptr().add(offset_bytes) as *mut u8 }
        };

        let arraybuffer = unsafe {
            env.create_arraybuffer_with_borrowed_data(
                ptr,
                len_bytes,
                holder,
                |arc: Arc<Vec<u8>>, _env: Env| drop(arc),
            )?
        };
        let typed = arraybuffer
            .into_raw()
            .into_typedarray(TypedArrayType::Uint8, len_bytes, 0)?;
        typed.into_unknown().coerce_to_object()
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
    res.map(Matrix::from_buffer).map_err(map_core_error)
}

fn map_core_error<T: Into<String>>(err: T) -> Error {
    let text: String = err.into();
    if let Some((code, message)) = text.split_once(": ") {
        let status = match code {
            num_rs_core::codes::SHAPE_MISMATCH => napi::Status::InvalidArg,
            num_rs_core::codes::NUMERIC_ISSUE => napi::Status::GenericFailure,
            num_rs_core::codes::CHOLESKY_NOT_SPD => napi::Status::InvalidArg,
            _ => napi::Status::GenericFailure,
        };
        Error::new(status, format!("{code}: {message}"))
    } else {
        Error::from_reason(text)
    }
}

#[napi]
pub fn add(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    map_matrix(core_add(a.buffer(), b.buffer()))
}

#[napi]
pub fn sub(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    map_matrix(core_sub(a.buffer(), b.buffer()))
}

#[napi]
pub fn mul(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    map_matrix(core_mul(a.buffer(), b.buffer()))
}

#[napi]
pub fn div(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    map_matrix(core_div(a.buffer(), b.buffer()))
}

#[napi]
pub fn neg(matrix: &Matrix) -> Result<Matrix> {
    map_matrix(core_neg(matrix.buffer()))
}

#[napi]
pub fn exp(matrix: &Matrix) -> Result<Matrix> {
    map_matrix(core_exp(matrix.buffer()))
}

#[napi]
pub fn log(matrix: &Matrix) -> Result<Matrix> {
    map_matrix(core_log(matrix.buffer()))
}

#[napi]
pub fn sin(matrix: &Matrix) -> Result<Matrix> {
    map_matrix(core_sin(matrix.buffer()))
}

#[napi]
pub fn cos(matrix: &Matrix) -> Result<Matrix> {
    map_matrix(core_cos(matrix.buffer()))
}

#[napi]
pub fn tanh(matrix: &Matrix) -> Result<Matrix> {
    map_matrix(core_tanh(matrix.buffer()))
}

#[napi]
pub fn sigmoid(matrix: &Matrix) -> Result<Matrix> {
    map_matrix(core_sigmoid(matrix.buffer()))
}

#[napi]
pub fn matmul(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    map_matrix(core_matmul(a.buffer(), b.buffer()))
}

#[napi]
pub fn gpu_available() -> bool {
    gpu::active_backend_kind().is_some()
}

#[napi]
pub fn gpu_backend_kind() -> Option<String> {
    gpu::backend_name().map(|kind| kind.to_string())
}

#[napi]
pub fn gpu_matmul(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    if let Some(_kind) = gpu::active_backend_kind() {
        let rows = a.rows() as usize;
        let shared = a.cols() as usize;
        let cols = b.cols() as usize;
        if shared != b.rows() as usize {
            return Err(map_core_error(format!(
                "gpu_matmul: left.cols ({shared}) must equal right.rows ({})",
                b.rows()
            )));
        }
        let lhs = ensure_float32_buffer(a.buffer()).map_err(map_core_error)?;
        let rhs = ensure_float32_buffer(b.buffer()).map_err(map_core_error)?;
        let lhs_view = lhs.try_as_slice::<f32>().map_err(map_core_error)?;
        let rhs_view = rhs.try_as_slice::<f32>().map_err(map_core_error)?;
        match gpu::matmul_f32(lhs_view, rhs_view, rows, shared, cols) {
            Ok(values) => {
                let buffer = MatrixBuffer::from_vec(values, rows, cols).map_err(map_core_error)?;
                return Ok(Matrix::from_buffer(buffer));
            }
            Err(err) => {
                eprintln!("[numjs] GPU matmul failed: {err}. Falling back to CPU.");
            }
        }
    }
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
pub fn transpose(matrix: &Matrix) -> Result<Matrix> {
    map_matrix(core_transpose(matrix.buffer()))
}

#[napi(js_name = "broadcast_to")]
pub fn broadcast_to(matrix: &Matrix, rows: u32, cols: u32) -> Result<Matrix> {
    map_matrix(core_broadcast_to(
        matrix.buffer(),
        rows as usize,
        cols as usize,
    ))
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
    let (u, s, vt) = core_svd(matrix.buffer()).map_err(map_core_error)?;
    let mut obj = env.create_object()?;
    obj.set("u", Matrix::from_buffer(u))?;
    obj.set("s", Float64Array::from(s.to_f64_vec()))?;
    obj.set("vt", Matrix::from_buffer(vt))?;
    Ok(obj)
}

#[cfg(feature = "linalg")]
#[napi]
pub fn qr(env: Env, matrix: &Matrix) -> Result<JsObject> {
    let (q, r) = core_qr(matrix.buffer()).map_err(map_core_error)?;
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
        .map_err(map_core_error)
}

#[cfg(feature = "linalg")]
#[napi]
pub fn eigen(env: Env, matrix: &Matrix) -> Result<JsObject> {
    let (values, vectors) = core_eigen(matrix.buffer()).map_err(map_core_error)?;
    let mut obj = env.create_object()?;
    obj.set("values", Float64Array::from(values.to_f64_vec()))?;
    obj.set("vectors", Matrix::from_buffer(vectors))?;
    Ok(obj)
}

#[napi]
pub fn read_npy(buffer: Buffer) -> Result<Matrix> {
    read_npy_matrix(&buffer)
        .map(Matrix::from_buffer)
        .map_err(map_core_error)
}

#[napi]
pub fn write_npy(matrix: &Matrix) -> Result<Buffer> {
    write_npy_matrix(matrix.buffer())
        .map(Buffer::from)
        .map_err(map_core_error)
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

// ---------------------------------------------------------------------
// Stable reductions
// ---------------------------------------------------------------------

#[napi]
pub fn sum(matrix: &Matrix, dtype: Option<String>) -> Result<Matrix> {
    let target = match dtype {
        Some(value) => Some(value.parse::<DType>().map_err(map_core_error)?),
        None => None,
    };
    map_matrix(core_sum(matrix.buffer(), target))
}

#[napi]
pub fn gpu_sum(matrix: &Matrix, dtype: Option<String>) -> Result<Matrix> {
    let target = match dtype {
        Some(value) => Some(value.parse::<DType>().map_err(map_core_error)?),
        None => None,
    };
    if let Some(_kind) = gpu::active_backend_kind() {
        let buffer = ensure_float32_buffer(matrix.buffer()).map_err(map_core_error)?;
        let view = buffer.try_as_slice::<f32>().map_err(map_core_error)?;
        match gpu::reduce_sum_f32(view) {
            Ok(total) => {
                let base = MatrixBuffer::from_vec(vec![total], 1, 1).map_err(map_core_error)?;
                let mut result = Matrix::from_buffer(base);
                if let Some(target_dtype) = target {
                    if target_dtype != DType::Float32 {
                        result =
                            result.astype(target_dtype.as_str().to_string(), Some(false), None)?;
                    }
                }
                return Ok(result);
            }
            Err(err) => {
                eprintln!("[numjs] GPU sum failed: {err}. Falling back to CPU.");
            }
        }
    }
    map_matrix(core_sum(matrix.buffer(), target))
}

#[napi]
pub fn nansum(matrix: &Matrix, dtype: Option<String>) -> Result<Matrix> {
    let target = match dtype {
        Some(value) => Some(value.parse::<DType>().map_err(map_core_error)?),
        None => None,
    };
    map_matrix(core_nansum(matrix.buffer(), target))
}

#[napi]
pub fn nanmean(matrix: &Matrix, dtype: Option<String>) -> Result<Matrix> {
    let target = match dtype {
        Some(value) => Some(value.parse::<DType>().map_err(map_core_error)?),
        None => None,
    };
    map_matrix(core_nanmean(matrix.buffer(), target))
}

#[napi]
pub fn median(matrix: &Matrix, dtype: Option<String>) -> Result<Matrix> {
    let target = match dtype {
        Some(value) => Some(value.parse::<DType>().map_err(map_core_error)?),
        None => None,
    };
    map_matrix(core_median(matrix.buffer(), target))
}

#[napi]
pub fn quantile(matrix: &Matrix, q: f64, dtype: Option<String>) -> Result<Matrix> {
    let target = match dtype {
        Some(value) => Some(value.parse::<DType>().map_err(map_core_error)?),
        None => None,
    };
    map_matrix(core_quantile(matrix.buffer(), q, target))
}

#[napi]
pub fn percentile(matrix: &Matrix, p: f64, dtype: Option<String>) -> Result<Matrix> {
    let target = match dtype {
        Some(value) => Some(value.parse::<DType>().map_err(map_core_error)?),
        None => None,
    };
    map_matrix(core_percentile(matrix.buffer(), p, target))
}

#[napi]
pub fn dot(a: &Matrix, b: &Matrix, dtype: Option<String>) -> Result<Matrix> {
    let target = match dtype {
        Some(value) => Some(value.parse::<DType>().map_err(map_core_error)?),
        None => None,
    };
    map_matrix(core_dot(a.buffer(), b.buffer(), target))
}

// ---------------------------------------------------------------------
// Views: row/column/slice
// ---------------------------------------------------------------------

#[napi]
pub fn row(matrix: &Matrix, index: i32) -> Result<Matrix> {
    matrix
        .buffer()
        .row(index as isize)
        .map(Matrix::from_buffer)
        .map_err(map_core_error)
}

#[napi]
pub fn column(matrix: &Matrix, index: i32) -> Result<Matrix> {
    matrix
        .buffer()
        .column(index as isize)
        .map(Matrix::from_buffer)
        .map_err(map_core_error)
}

#[napi]
pub fn slice(
    matrix: &Matrix,
    row_start: Option<i32>,
    row_end: Option<i32>,
    row_step: Option<i32>,
    col_start: Option<i32>,
    col_end: Option<i32>,
    col_step: Option<i32>,
) -> Result<Matrix> {
    let rows = SliceSpec::new(
        row_start.map(|v| v as isize),
        row_end.map(|v| v as isize),
        row_step.unwrap_or(1) as isize,
    )
    .map_err(map_core_error)?;
    let cols = SliceSpec::new(
        col_start.map(|v| v as isize),
        col_end.map(|v| v as isize),
        col_step.unwrap_or(1) as isize,
    )
    .map_err(map_core_error)?;
    matrix
        .buffer()
        .slice(rows, cols)
        .map(Matrix::from_buffer)
        .map_err(map_core_error)
}

fn convert_indices(indices: &[i64]) -> Result<Vec<isize>> {
    indices
        .iter()
        .map(|value| {
            isize::try_from(*value)
                .map_err(|_| map_core_error(format!("index {value} exceeds platform limits")))
        })
        .collect()
}

fn ensure_float32_buffer(buffer: &MatrixBuffer) -> Result<MatrixBuffer, String> {
    if buffer.dtype() == DType::Float32 {
        Ok(buffer.clone())
    } else {
        buffer.cast(DType::Float32)
    }
}

// ---------------------------------------------------------------------
// Boolean mask compress
// ---------------------------------------------------------------------

#[napi]
pub fn compress(mask: &Matrix, matrix: &Matrix) -> Result<Matrix> {
    core_compress(mask.buffer(), matrix.buffer())
        .map(Matrix::from_buffer)
        .map_err(map_core_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed64_matrix_reports_scale_and_concat_preserves_it() {
        let base = MatrixBuffer::from_fixed_i64_vec(vec![1200, 2200], 1, 2, 2).unwrap();
        let matrix = Matrix::from_buffer(base.clone());
        assert_eq!(matrix.fixed_scale(), Some(2));
        let reduced = sum(&matrix, None).expect("sum fixed64");
        let total = reduced.buffer().to_f64_vec()[0];
        assert!((total - 34.0).abs() < 1e-9);

        let merged = concat(&matrix, &Matrix::from_buffer(base.clone()), 0).unwrap();
        assert_eq!(merged.fixed_scale(), Some(2));
        assert_eq!(merged.buffer().rows(), 2);
    }

    #[test]
    fn fixed64_from_bytes_rejected() {
        let raw = Buffer::from(vec![0u8; 16]);
        let result = Matrix::from_bytes(raw, 2, 1, "fixed64".to_string());
        assert!(result.is_err());
    }
}
