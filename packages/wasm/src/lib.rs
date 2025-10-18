use js_sys::{Error as JsError, Float32Array, Float64Array, Reflect, Uint8Array};
use num_rs_core::buffer::{CastOptions, CastingKind, MatrixBuffer, SliceSpec};
use num_rs_core::compress::compress as core_compress;
use num_rs_core::dtype::DType;
use num_rs_core::{
    add as core_add, broadcast_to as core_broadcast_to, clip as core_clip, concat as core_concat,
    cos as core_cos, div as core_div, dot as core_dot, exp as core_exp, fft2d as core_fft2d,
    fft_axis as core_fft_axis, gather as core_gather, gather_pairs as core_gather_pairs,
    ifft2d as core_ifft2d, ifft_axis as core_ifft_axis, log as core_log, matmul as core_matmul,
    median as core_median, mul as core_mul, nanmean as core_nanmean, nansum as core_nansum,
    neg as core_neg, percentile as core_percentile, put as core_put, quantile as core_quantile,
    scatter as core_scatter, scatter_pairs as core_scatter_pairs, sigmoid as core_sigmoid,
    sin as core_sin, stack as core_stack, sub as core_sub, sum as core_sum, take as core_take,
    tanh as core_tanh, transpose as core_transpose, where_select as core_where,
};
use std::convert::TryFrom;
use std::str::FromStr;
use wasm_bindgen::prelude::*;

#[cfg(feature = "threads")]
use std::sync::OnceLock;

#[cfg(feature = "threads")]
static THREAD_POOL: OnceLock<()> = OnceLock::new();

#[wasm_bindgen]
#[derive(Clone)]
pub struct Matrix {
    buffer: MatrixBuffer,
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct ComplexPair {
    real: Matrix,
    imag: Matrix,
}

#[wasm_bindgen]
impl ComplexPair {
    #[wasm_bindgen(getter)]
    pub fn real(&self) -> Matrix {
        self.real.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn imag(&self) -> Matrix {
        self.imag.clone()
    }
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
        if dtype == DType::Fixed64 {
            return Err(JsValue::from_str(
                "from_bytes(Fixed64): supply scaled bigint data via Matrix.fromFixedI64",
            ));
        }
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

    #[wasm_bindgen(getter, js_name = fixedScale)]
    pub fn fixed_scale(&self) -> Option<i32> {
        self.buffer.fixed_scale()
    }

    #[wasm_bindgen]
    pub fn astype(
        &self,
        dtype: String,
        copy: Option<bool>,
        casting: Option<String>,
    ) -> Result<Matrix, JsValue> {
        let dtype = DType::from_str(&dtype).map_err(|err| JsValue::from_str(&err))?;
        let copy = copy.unwrap_or(false);
        let options =
            CastOptions::parse(casting.as_deref()).map_err(|err| JsValue::from_str(&err))?;
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
        if !copy
            && matches!(options.casting(), CastingKind::Unsafe)
            && dtype.size_of() == self.buffer.dtype().size_of()
        {
            let buffer = self
                .buffer
                .reinterpret(dtype)
                .map_err(|err| JsValue::from_str(&err))?;
            return Ok(Matrix::from_buffer(buffer));
        }
        let buffer = self
            .buffer
            .cast_with_options(dtype, &options)
            .map_err(|err| JsValue::from_str(&err))?;
        Ok(Matrix::from_buffer(buffer))
    }

    #[wasm_bindgen]
    pub fn to_vec(&self) -> Vec<f64> {
        self.buffer.to_f64_vec()
    }

    #[wasm_bindgen(js_name = "to_bytes")]
    pub fn to_bytes(&self) -> Result<Uint8Array, JsValue> {
        if let Some(bytes) = self.buffer.as_byte_slice() {
            // SAFETY: the slice references memory owned by this matrix. The caller must ensure
            // the matrix outlives the typed array view.
            let view = unsafe { Uint8Array::view(bytes) };
            Ok(view)
        } else {
            let data = self.buffer.to_contiguous_bytes_vec();
            Ok(Uint8Array::from(data.as_slice()))
        }
    }

    #[wasm_bindgen(js_name = "toFloat32Array")]
    pub fn to_float32_array(&self) -> Result<Float32Array, JsValue> {
        if self.buffer.dtype() != DType::Float32 {
            return Err(JsValue::from_str(
                "toFloat32Array requires a matrix with dtype \"float32\"",
            ));
        }
        match self.buffer.try_as_slice::<f32>() {
            Ok(slice) => Ok(unsafe { Float32Array::view(slice) }),
            Err(_) => {
                let contiguous = self
                    .buffer
                    .to_contiguous()
                    .map_err(|err| JsValue::from_str(&err))?;
                let values = contiguous
                    .try_as_slice::<f32>()
                    .map_err(|err| JsValue::from_str(&err))?;
                Ok(Float32Array::from(values))
            }
        }
    }

    #[wasm_bindgen(js_name = "toFloat64Array")]
    pub fn to_float64_array(&self) -> Result<Float64Array, JsValue> {
        if self.buffer.dtype() != DType::Float64 {
            return Err(JsValue::from_str(
                "toFloat64Array requires a matrix with dtype \"float64\"",
            ));
        }
        match self.buffer.try_as_slice::<f64>() {
            Ok(slice) => Ok(unsafe { Float64Array::view(slice) }),
            Err(_) => {
                let data = self.buffer.to_f64_vec();
                Ok(Float64Array::from(data.as_slice()))
            }
        }
    }

    #[wasm_bindgen(js_name = "fromFixedI64")]
    pub fn from_fixed_i64(
        data: Vec<i64>,
        rows: usize,
        cols: usize,
        scale: i32,
    ) -> Result<Matrix, JsValue> {
        MatrixBuffer::from_fixed_i64_vec(data, rows, cols, scale)
            .map(Matrix::from_buffer)
            .map_err(|err| JsValue::from_str(&err))
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
        .map_err(|err| map_core_error(&err))
}

fn map_core_error(err: &str) -> JsValue {
    if let Some((code, message)) = err.split_once(": ") {
        let js_error = JsError::new(message);
        let _ = Reflect::set(
            &js_error,
            &JsValue::from_str("code"),
            &JsValue::from_str(code),
        );
        js_error.into()
    } else {
        JsError::new(err).into()
    }
}

fn complex_pair_from_buffers(real: MatrixBuffer, imag: MatrixBuffer) -> ComplexPair {
    ComplexPair {
        real: Matrix::from_buffer(real),
        imag: Matrix::from_buffer(imag),
    }
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
pub fn sub(a: &Matrix, b: &Matrix) -> Result<Matrix, JsValue> {
    map_matrix(core_sub(a.buffer(), b.buffer()))
}

#[wasm_bindgen]
pub fn mul(a: &Matrix, b: &Matrix) -> Result<Matrix, JsValue> {
    map_matrix(core_mul(a.buffer(), b.buffer()))
}

#[wasm_bindgen]
pub fn div(a: &Matrix, b: &Matrix) -> Result<Matrix, JsValue> {
    map_matrix(core_div(a.buffer(), b.buffer()))
}

#[wasm_bindgen]
pub fn neg(matrix: &Matrix) -> Result<Matrix, JsValue> {
    map_matrix(core_neg(matrix.buffer()))
}

#[wasm_bindgen]
pub fn exp(matrix: &Matrix) -> Result<Matrix, JsValue> {
    map_matrix(core_exp(matrix.buffer()))
}

#[wasm_bindgen]
pub fn log(matrix: &Matrix) -> Result<Matrix, JsValue> {
    map_matrix(core_log(matrix.buffer()))
}

#[wasm_bindgen]
pub fn sin(matrix: &Matrix) -> Result<Matrix, JsValue> {
    map_matrix(core_sin(matrix.buffer()))
}

#[wasm_bindgen]
pub fn cos(matrix: &Matrix) -> Result<Matrix, JsValue> {
    map_matrix(core_cos(matrix.buffer()))
}

#[wasm_bindgen]
pub fn tanh(matrix: &Matrix) -> Result<Matrix, JsValue> {
    map_matrix(core_tanh(matrix.buffer()))
}

#[wasm_bindgen]
pub fn sigmoid(matrix: &Matrix) -> Result<Matrix, JsValue> {
    map_matrix(core_sigmoid(matrix.buffer()))
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
pub fn transpose(matrix: &Matrix) -> Result<Matrix, JsValue> {
    map_matrix(core_transpose(matrix.buffer()))
}

#[wasm_bindgen(js_name = "broadcast_to")]
pub fn broadcast_to(matrix: &Matrix, rows: usize, cols: usize) -> Result<Matrix, JsValue> {
    map_matrix(core_broadcast_to(matrix.buffer(), rows, cols))
}

#[wasm_bindgen]
pub fn row(matrix: &Matrix, index: i32) -> Result<Matrix, JsValue> {
    matrix
        .buffer()
        .row(index as isize)
        .map(Matrix::from_buffer)
        .map_err(|err| JsValue::from_str(&err))
}

#[wasm_bindgen]
pub fn column(matrix: &Matrix, index: i32) -> Result<Matrix, JsValue> {
    matrix
        .buffer()
        .column(index as isize)
        .map(Matrix::from_buffer)
        .map_err(|err| JsValue::from_str(&err))
}

#[wasm_bindgen]
pub fn slice(
    matrix: &Matrix,
    row_start: Option<i32>,
    row_end: Option<i32>,
    row_step: Option<i32>,
    col_start: Option<i32>,
    col_end: Option<i32>,
    col_step: Option<i32>,
) -> Result<Matrix, JsValue> {
    let rows = SliceSpec::new(
        row_start.map(|v| v as isize),
        row_end.map(|v| v as isize),
        row_step.unwrap_or(1) as isize,
    )
    .map_err(|e| JsValue::from_str(&e))?;
    let cols = SliceSpec::new(
        col_start.map(|v| v as isize),
        col_end.map(|v| v as isize),
        col_step.unwrap_or(1) as isize,
    )
    .map_err(|e| JsValue::from_str(&e))?;
    matrix
        .buffer()
        .slice(rows, cols)
        .map(Matrix::from_buffer)
        .map_err(|e| JsValue::from_str(&e))
}

#[wasm_bindgen]
pub fn compress(mask: &Matrix, matrix: &Matrix) -> Result<Matrix, JsValue> {
    core_compress(mask.buffer(), matrix.buffer())
        .map(Matrix::from_buffer)
        .map_err(|e| JsValue::from_str(&e))
}
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

#[wasm_bindgen(js_name = "fft_axis")]
pub fn fft_axis_wasm(matrix: &Matrix, axis: usize) -> Result<ComplexPair, JsValue> {
    core_fft_axis(matrix.buffer(), axis)
        .map(|(real, imag)| complex_pair_from_buffers(real, imag))
        .map_err(|err| map_core_error(&err))
}

#[wasm_bindgen(js_name = "ifft_axis")]
pub fn ifft_axis_wasm(real: &Matrix, imag: &Matrix, axis: usize) -> Result<ComplexPair, JsValue> {
    core_ifft_axis(real.buffer(), imag.buffer(), axis)
        .map(|(real_buf, imag_buf)| complex_pair_from_buffers(real_buf, imag_buf))
        .map_err(|err| map_core_error(&err))
}

#[wasm_bindgen(js_name = "fft2d")]
pub fn fft2d_wasm(matrix: &Matrix) -> Result<ComplexPair, JsValue> {
    core_fft2d(matrix.buffer())
        .map(|(real, imag)| complex_pair_from_buffers(real, imag))
        .map_err(|err| map_core_error(&err))
}

#[wasm_bindgen(js_name = "ifft2d")]
pub fn ifft2d_wasm(real: &Matrix, imag: &Matrix) -> Result<ComplexPair, JsValue> {
    core_ifft2d(real.buffer(), imag.buffer())
        .map(|(real_buf, imag_buf)| complex_pair_from_buffers(real_buf, imag_buf))
        .map_err(|err| map_core_error(&err))
}

#[cfg(feature = "threads")]
#[wasm_bindgen(js_name = "initThreads")]
pub async fn init_threads(workers: Option<u32>) -> Result<(), JsValue> {
    if THREAD_POOL.get().is_some() {
        return Ok(());
    }
    let request = workers
        .map(|value| usize::try_from(value).unwrap_or(1).max(1))
        .map(Some)
        .unwrap_or(None);
    wasm_bindgen_rayon::init_thread_pool(request)
        .await
        .map_err(|err| JsValue::from(err))?;
    let _ = THREAD_POOL.set(());
    Ok(())
}

#[cfg(not(feature = "threads"))]
#[wasm_bindgen(js_name = "initThreads")]
pub async fn init_threads(_workers: Option<u32>) -> Result<(), JsValue> {
    Err(JsValue::from_str(
        "wasm threads support disabled; rebuild num_rs_wasm with the `threads` feature",
    ))
}
// ---------------------------------------------------------------------
// Stable reductions
// ---------------------------------------------------------------------

#[wasm_bindgen]
pub fn sum(matrix: &Matrix, dtype: Option<String>) -> Result<Matrix, JsValue> {
    let target = match dtype {
        Some(value) => Some(DType::from_str(&value).map_err(|err| JsValue::from_str(&err))?),
        None => None,
    };
    map_matrix(core_sum(matrix.buffer(), target))
}

#[wasm_bindgen]
pub fn nansum(matrix: &Matrix, dtype: Option<String>) -> Result<Matrix, JsValue> {
    let target = match dtype {
        Some(value) => Some(DType::from_str(&value).map_err(|err| JsValue::from_str(&err))?),
        None => None,
    };
    map_matrix(core_nansum(matrix.buffer(), target))
}

#[wasm_bindgen]
pub fn nanmean(matrix: &Matrix, dtype: Option<String>) -> Result<Matrix, JsValue> {
    let target = match dtype {
        Some(value) => Some(DType::from_str(&value).map_err(|err| JsValue::from_str(&err))?),
        None => None,
    };
    map_matrix(core_nanmean(matrix.buffer(), target))
}

#[wasm_bindgen]
pub fn median(matrix: &Matrix, dtype: Option<String>) -> Result<Matrix, JsValue> {
    let target = match dtype {
        Some(value) => Some(DType::from_str(&value).map_err(|err| JsValue::from_str(&err))?),
        None => None,
    };
    map_matrix(core_median(matrix.buffer(), target))
}

#[wasm_bindgen]
pub fn quantile(matrix: &Matrix, q: f64, dtype: Option<String>) -> Result<Matrix, JsValue> {
    let target = match dtype {
        Some(value) => Some(DType::from_str(&value).map_err(|err| JsValue::from_str(&err))?),
        None => None,
    };
    map_matrix(core_quantile(matrix.buffer(), q, target))
}

#[wasm_bindgen]
pub fn percentile(matrix: &Matrix, p: f64, dtype: Option<String>) -> Result<Matrix, JsValue> {
    let target = match dtype {
        Some(value) => Some(DType::from_str(&value).map_err(|err| JsValue::from_str(&err))?),
        None => None,
    };
    map_matrix(core_percentile(matrix.buffer(), p, target))
}

#[wasm_bindgen]
pub fn dot(a: &Matrix, b: &Matrix, dtype: Option<String>) -> Result<Matrix, JsValue> {
    let target = match dtype {
        Some(value) => Some(DType::from_str(&value).map_err(|err| JsValue::from_str(&err))?),
        None => None,
    };
    map_matrix(core_dot(a.buffer(), b.buffer(), target))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed64_getter_exposes_scale() {
        let buffer = MatrixBuffer::from_fixed_i64_vec(vec![500, 600], 1, 2, 2).unwrap();
        let matrix = Matrix::from_buffer(buffer.clone());
        assert_eq!(matrix.fixed_scale(), Some(2));
        let reduced = sum(&matrix, None).expect("sum fixed64");
        let total = reduced.buffer().to_f64_vec()[0];
        assert!((total - 11.0).abs() < 1e-9);

        let combined = concat(&matrix, &Matrix::from_buffer(buffer), 0).unwrap();
        assert_eq!(combined.fixed_scale(), Some(2));
    }
}
