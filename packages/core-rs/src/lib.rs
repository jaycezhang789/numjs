pub mod buffer;
pub mod dtype;
pub mod element;
mod macros;
pub mod metrics;

use buffer::MatrixBuffer;
use dtype::{promote_many, promote_pair, DType};
use element::Element as ElementTrait;
use num_traits::{Bounded, Float, FromPrimitive, PrimInt, Signed, ToPrimitive, Unsigned};

use ndarray::Array2;
use std::convert::TryInto;

#[cfg(feature = "npy")]
use ndarray_npy::{NpzReader, NpzWriter, ReadNpyExt, WriteNpyExt};

#[cfg(feature = "npy")]
use std::io::Cursor;

#[cfg(feature = "linalg")]
use nalgebra::{DMatrix, SymmetricEigen};

pub use metrics::{copy_bytes_total, reset_copy_bytes, take_copy_bytes};
pub type CoreResult<T> = Result<T, String>;

// ---------------------------------------------------------------------
// Numerically stable reductions (pairwise sum/dot, Welford mean/variance)
// ---------------------------------------------------------------------
fn reduction_accumulator_dtype(dtype: DType) -> DType {
    match dtype {
        DType::Bool => DType::Int64,
        DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64 => DType::Int64,
        DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64 => DType::UInt64,
        DType::Float32 => DType::Float32,
        DType::Float64 => DType::Float64,
        DType::Fixed64 => DType::Fixed64,
    }
}

fn apply_reduce_output_dtype(
    result: MatrixBuffer,
    target: Option<DType>,
) -> CoreResult<MatrixBuffer> {
    if let Some(target_dtype) = target {
        if target_dtype == result.dtype() {
            Ok(result)
        } else {
            result.cast(target_dtype)
        }
    } else {
        Ok(result)
    }
}

fn sum_bool(matrix: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    let contiguous = matrix.to_contiguous()?;
    let slice = contiguous.try_as_slice::<bool>()?;
    let mut total: i128 = 0;
    for &value in slice {
        if value {
            total += 1;
        }
    }
    if total > i64::MAX as i128 {
        return Err("sum(bool): overflow while counting true values".into());
    }
    MatrixBuffer::from_vec(vec![total as i64], 1, 1).map_err(Into::into)
}

fn sum_signed_numeric<T>(matrix: &MatrixBuffer) -> CoreResult<MatrixBuffer>
where
    T: PrimInt + Signed + ToPrimitive + ElementTrait,
{
    let contiguous = matrix.to_contiguous()?;
    let slice = contiguous.try_as_slice::<T>()?;
    let mut total: i128 = 0;
    for &value in slice {
        let v = value
            .to_i64()
            .ok_or_else(|| "sum: failed to convert signed value to i64".to_string())?;
        total = total
            .checked_add(v as i128)
            .ok_or_else(|| "sum: signed accumulator overflow".to_string())?;
    }
    if total < i64::MIN as i128 || total > i64::MAX as i128 {
        return Err("sum: signed result exceeds int64 range".into());
    }
    MatrixBuffer::from_vec(vec![total as i64], 1, 1).map_err(Into::into)
}

fn sum_unsigned_numeric<T>(matrix: &MatrixBuffer) -> CoreResult<MatrixBuffer>
where
    T: PrimInt + Unsigned + ToPrimitive + ElementTrait,
{
    let contiguous = matrix.to_contiguous()?;
    let slice = contiguous.try_as_slice::<T>()?;
    let mut total: u128 = 0;
    for &value in slice {
        let v = value
            .to_u64()
            .ok_or_else(|| "sum: failed to convert unsigned value to u64".to_string())?;
        total = total
            .checked_add(v as u128)
            .ok_or_else(|| "sum: unsigned accumulator overflow".to_string())?;
    }
    if total > u64::MAX as u128 {
        return Err("sum: unsigned result exceeds uint64 range".into());
    }
    MatrixBuffer::from_vec(vec![total as u64], 1, 1).map_err(Into::into)
}

fn sum_float_numeric<T>(matrix: &MatrixBuffer) -> CoreResult<MatrixBuffer>
where
    T: Float + ElementTrait,
{
    let contiguous = matrix.to_contiguous()?;
    let slice = contiguous.try_as_slice::<T>()?;
    let mut sum = T::zero();
    let mut compensation = T::zero();
    for &value in slice {
        let y = value - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    MatrixBuffer::from_vec(vec![sum], 1, 1).map_err(Into::into)
}

fn sum_fixed64(matrix: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    let scale = ensure_fixed64(matrix)?;
    let data = matrix_to_fixed_vec(matrix)?;
    let mut total: i128 = 0;
    for value in data {
        total = total
            .checked_add(value as i128)
            .ok_or_else(|| "sum(Fixed64): overflow".to_string())?;
    }
    if total < i64::MIN as i128 || total > i64::MAX as i128 {
        return Err("sum(Fixed64): overflow".into());
    }
    MatrixBuffer::from_fixed_i64_vec(vec![total as i64], 1, 1, scale)
}

pub fn sum(matrix: &MatrixBuffer, target_dtype: Option<DType>) -> CoreResult<MatrixBuffer> {
    let result = match matrix.dtype() {
        DType::Bool => sum_bool(matrix),
        DType::Int8 => sum_signed_numeric::<i8>(matrix),
        DType::Int16 => sum_signed_numeric::<i16>(matrix),
        DType::Int32 => sum_signed_numeric::<i32>(matrix),
        DType::Int64 => sum_signed_numeric::<i64>(matrix),
        DType::UInt8 => sum_unsigned_numeric::<u8>(matrix),
        DType::UInt16 => sum_unsigned_numeric::<u16>(matrix),
        DType::UInt32 => sum_unsigned_numeric::<u32>(matrix),
        DType::UInt64 => sum_unsigned_numeric::<u64>(matrix),
        DType::Float32 => sum_float_numeric::<f32>(matrix),
        DType::Float64 => sum_float_numeric::<f64>(matrix),
        DType::Fixed64 => sum_fixed64(matrix),
    }?;
    apply_reduce_output_dtype(result, target_dtype)
}

fn dot_bool(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    let lhs = a.try_as_slice::<bool>()?;
    let rhs = b.try_as_slice::<bool>()?;
    let mut total: i128 = 0;
    for (&x, &y) in lhs.iter().zip(rhs.iter()) {
        if x && y {
            total = total
                .checked_add(1)
                .ok_or_else(|| "dot(bool, bool): overflow".to_string())?;
        }
    }
    MatrixBuffer::from_vec(vec![total as i64], 1, 1).map_err(Into::into)
}

fn dot_signed_numeric<T>(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer>
where
    T: PrimInt + Signed + ToPrimitive + ElementTrait,
{
    let lhs = a.try_as_slice::<T>()?;
    let rhs = b.try_as_slice::<T>()?;
    let mut total: i128 = 0;
    for (&x, &y) in lhs.iter().zip(rhs.iter()) {
        let xi = x
            .to_i64()
            .ok_or_else(|| "dot: failed to convert signed multiplicand".to_string())?;
        let yi = y
            .to_i64()
            .ok_or_else(|| "dot: failed to convert signed multiplicand".to_string())?;
        total = total
            .checked_add((xi as i128) * (yi as i128))
            .ok_or_else(|| "dot: signed accumulator overflow".to_string())?;
    }
    if total < i64::MIN as i128 || total > i64::MAX as i128 {
        return Err("dot: signed result exceeds int64 range".into());
    }
    MatrixBuffer::from_vec(vec![total as i64], 1, 1).map_err(Into::into)
}

fn dot_unsigned_numeric<T>(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer>
where
    T: PrimInt + Unsigned + ToPrimitive + ElementTrait,
{
    let lhs = a.try_as_slice::<T>()?;
    let rhs = b.try_as_slice::<T>()?;
    let mut total: u128 = 0;
    for (&x, &y) in lhs.iter().zip(rhs.iter()) {
        let xi = x
            .to_u64()
            .ok_or_else(|| "dot: failed to convert unsigned multiplicand".to_string())?;
        let yi = y
            .to_u64()
            .ok_or_else(|| "dot: failed to convert unsigned multiplicand".to_string())?;
        total = total
            .checked_add((xi as u128) * (yi as u128))
            .ok_or_else(|| "dot: unsigned accumulator overflow".to_string())?;
    }
    if total > u64::MAX as u128 {
        return Err("dot: unsigned result exceeds uint64 range".into());
    }
    MatrixBuffer::from_vec(vec![total as u64], 1, 1).map_err(Into::into)
}

fn dot_float_numeric<T>(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer>
where
    T: Float + ElementTrait,
{
    let lhs = a.try_as_slice::<T>()?;
    let rhs = b.try_as_slice::<T>()?;
    let mut sum = T::zero();
    let mut compensation = T::zero();
    for (&x, &y) in lhs.iter().zip(rhs.iter()) {
        let product = x * y;
        let y_comp = product - compensation;
        let t = sum + y_comp;
        compensation = (t - sum) - y_comp;
        sum = t;
    }
    MatrixBuffer::from_vec(vec![sum], 1, 1).map_err(Into::into)
}

pub fn dot(
    a: &MatrixBuffer,
    b: &MatrixBuffer,
    target_dtype: Option<DType>,
) -> CoreResult<MatrixBuffer> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err("dot: shape mismatch".into());
    }
    if a.dtype() == DType::Fixed64 || b.dtype() == DType::Fixed64 {
        return Err("dot(Fixed64): convert operands to float64 before reducing".into());
    }
    let mul_dtype = promote_pair(a.dtype(), b.dtype());
    let acc_dtype = reduction_accumulator_dtype(mul_dtype);

    let left_cast = if a.dtype() == mul_dtype {
        a.to_contiguous()?
    } else {
        a.cast(mul_dtype)?
            .to_contiguous()?
    };
    let right_cast = if b.dtype() == mul_dtype {
        b.to_contiguous()?
    } else {
        b.cast(mul_dtype)?
            .to_contiguous()?
    };

    let result = match mul_dtype {
        DType::Bool => dot_bool(&left_cast, &right_cast),
        DType::Int8 => dot_signed_numeric::<i8>(&left_cast, &right_cast),
        DType::Int16 => dot_signed_numeric::<i16>(&left_cast, &right_cast),
        DType::Int32 => dot_signed_numeric::<i32>(&left_cast, &right_cast),
        DType::Int64 => dot_signed_numeric::<i64>(&left_cast, &right_cast),
        DType::UInt8 => dot_unsigned_numeric::<u8>(&left_cast, &right_cast),
        DType::UInt16 => dot_unsigned_numeric::<u16>(&left_cast, &right_cast),
        DType::UInt32 => dot_unsigned_numeric::<u32>(&left_cast, &right_cast),
        DType::UInt64 => dot_unsigned_numeric::<u64>(&left_cast, &right_cast),
        DType::Float32 => dot_float_numeric::<f32>(&left_cast, &right_cast),
        DType::Float64 => dot_float_numeric::<f64>(&left_cast, &right_cast),
        DType::Fixed64 => unreachable!(),
    }?;

    let normalized = if acc_dtype == result.dtype() {
        result
    } else {
        result.cast(acc_dtype)?
    };

    apply_reduce_output_dtype(normalized, target_dtype)
}

pub fn welford_mean_variance(buffer: &MatrixBuffer, sample: bool) -> (f64, f64) {
    let mut n: f64 = 0.0;
    let mut mean: f64 = 0.0;
    let mut m2: f64 = 0.0;
    for v in buffer.to_f64_vec() {
        n += 1.0;
        let delta = v - mean;
        mean += delta / n;
        let delta2 = v - mean;
        m2 += delta * delta2;
    }
    if n < 1.0 {
        return (f64::NAN, f64::NAN);
    }
    let var = if sample && n > 1.0 {
        m2 / (n - 1.0)
    } else {
        m2 / n
    };
    (mean, var)
}

fn bytes_to_i64_vec(bytes: &[u8]) -> Vec<i64> {
    bytes
        .chunks_exact(8)
        .map(|chunk| i64::from_ne_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn ensure_fixed64(matrix: &MatrixBuffer) -> CoreResult<i32> {
    if matrix.dtype() != DType::Fixed64 {
        return Err("expected fixed64 matrix".into());
    }
    matrix
        .fixed_scale()
        .ok_or_else(|| "fixed64 matrix missing scale metadata".into())
}

fn matrix_to_fixed_vec(matrix: &MatrixBuffer) -> CoreResult<Vec<i64>> {
    if matrix.dtype() != DType::Fixed64 {
        return Err("expected fixed64 matrix".into());
    }
    let contiguous = matrix
        .to_contiguous()
        .map_err(|e| format!("fixed64 contiguous: {e}"))?;
    let bytes = contiguous
        .as_byte_slice()
        .ok_or_else(|| "fixed64 contiguous bytes unavailable".to_string())?;
    Ok(bytes_to_i64_vec(bytes))
}

fn concat_fixed64(axis: usize, matrices: &[MatrixBuffer]) -> CoreResult<MatrixBuffer> {
    let first = matrices
        .first()
        .ok_or_else(|| "concat: expected at least one matrix".to_string())?;
    let scale = ensure_fixed64(first)?;
    match axis {
        0 => {
            let cols = first.cols();
            let mut total_rows = 0usize;
            let mut data: Vec<i64> = Vec::new();
            for m in matrices {
                ensure_fixed64(m)?;
                if m.fixed_scale() != Some(scale) {
                    return Err("concat axis 0: fixed64 scale mismatch".into());
                }
                if m.cols() != cols {
                    return Err("concat axis 0: column sizes differ".into());
                }
                let vec = matrix_to_fixed_vec(m)?;
                total_rows += m.rows();
                data.extend_from_slice(&vec);
            }
            MatrixBuffer::from_fixed_i64_vec(data, total_rows, cols, scale)
        }
        1 => {
            let rows = first.rows();
            let mut total_cols = 0usize;
            let mut parts: Vec<(usize, Vec<i64>)> = Vec::with_capacity(matrices.len());
            for m in matrices {
                ensure_fixed64(m)?;
                if m.fixed_scale() != Some(scale) {
                    return Err("concat axis 1: fixed64 scale mismatch".into());
                }
                if m.rows() != rows {
                    return Err("concat axis 1: row sizes differ".into());
                }
                let cols = m.cols();
                total_cols += cols;
                parts.push((cols, matrix_to_fixed_vec(m)?));
            }
            let mut data: Vec<i64> = Vec::with_capacity(rows * total_cols);
            for row in 0..rows {
                for (cols, vec) in parts.iter() {
                    let start = row * cols;
                    let end = start + cols;
                    data.extend_from_slice(&vec[start..end]);
                }
            }
            MatrixBuffer::from_fixed_i64_vec(data, rows, total_cols, scale)
        }
        _ => unreachable!(),
    }
}

fn where_select_multi_fixed64(
    conditions: &[&MatrixBuffer],
    choices: &[&MatrixBuffer],
    default: Option<&MatrixBuffer>,
    rows: usize,
    cols: usize,
) -> CoreResult<MatrixBuffer> {
    if choices.is_empty() {
        return Err("where_select_multi: requires at least one choice".into());
    }
    let scale = ensure_fixed64(choices[0])?;
    let mut result_data: Vec<i64> = if let Some(default_matrix) = default {
        ensure_fixed64(default_matrix)?;
        if default_matrix.fixed_scale() != Some(scale) {
            return Err("where_select_multi: fixed64 scale mismatch".into());
        }
        let broadcast = default_matrix
            .broadcast_to(rows, cols)
            .map_err(|e| format!("where_select_multi: {e}"))?;
        matrix_to_fixed_vec(&broadcast)?
    } else {
        vec![0; rows * cols]
    };

    for (cond, choice) in conditions.iter().zip(choices.iter()) {
        ensure_fixed64(choice)?;
        if choice.fixed_scale() != Some(scale) {
            return Err("where_select_multi: fixed64 scale mismatch".into());
        }
        let mask = cond
            .broadcast_to(rows, cols)
            .map_err(|e| format!("where_select_multi: {e}"))?
            .to_bool_vec();
        let choice_view = choice
            .broadcast_to(rows, cols)
            .map_err(|e| format!("where_select_multi: {e}"))?;
        let values = matrix_to_fixed_vec(&choice_view)?;
        for (idx, flag) in mask.iter().enumerate() {
            if *flag {
                result_data[idx] = values[idx];
            }
        }
    }

    MatrixBuffer::from_fixed_i64_vec(result_data, rows, cols, scale)
}

pub fn add(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    // Draft: integer add for Fixed64 when scales match
    if a.dtype() == DType::Fixed64 && b.dtype() == DType::Fixed64 {
        ensure_same_shape(a, b)?;
        if a.fixed_scale() != b.fixed_scale() {
            return Err("add(Fixed64): scale mismatch".into());
        }
        let scale = a.fixed_scale().unwrap_or(0);
        let vec_a = matrix_to_fixed_vec(a)?;
        let vec_b = matrix_to_fixed_vec(b)?;
        let mut out: Vec<i64> = Vec::with_capacity(vec_a.len());
        for (lhs, rhs) in vec_a.into_iter().zip(vec_b.into_iter()) {
            let (sum, overflow) = lhs.overflowing_add(rhs);
            if overflow {
                return Err("add(Fixed64): overflow".into());
            }
            out.push(sum);
        }
        return MatrixBuffer::from_fixed_i64_vec(out, a.rows(), a.cols(), scale);
    }
    ensure_same_shape(a, b)?;
    let dtype = promote_pair(a.dtype(), b.dtype());

    if dtype == DType::Bool {
        if a.dtype() != DType::Bool || b.dtype() != DType::Bool {
            return Err("add(bool, bool): unsupported mixed boolean addition".into());
        }
        let lhs_std = a.to_contiguous()?;
        let rhs_std = b.to_contiguous()?;
        let lhs = lhs_std.try_as_slice::<bool>()?;
        let rhs = rhs_std.try_as_slice::<bool>()?;
        let mut out: Vec<bool> = Vec::with_capacity(lhs.len());
        for (&x, &y) in lhs.iter().zip(rhs.iter()) {
            out.push(x || y);
        }
        return MatrixBuffer::from_vec(out, a.rows(), a.cols()).map_err(Into::into);
    }

    let lhs_cast = a.cast(dtype)?;
    let rhs_cast = b.cast(dtype)?;
    let lhs_std = lhs_cast.to_contiguous()?;
    let rhs_std = rhs_cast.to_contiguous()?;

    match dtype {
        DType::Int8 => add_signed_numeric::<i8>(&lhs_std, &rhs_std),
        DType::Int16 => add_signed_numeric::<i16>(&lhs_std, &rhs_std),
        DType::Int32 => add_signed_numeric::<i32>(&lhs_std, &rhs_std),
        DType::Int64 => add_signed_numeric::<i64>(&lhs_std, &rhs_std),
        DType::UInt8 => add_unsigned_numeric::<u8>(&lhs_std, &rhs_std),
        DType::UInt16 => add_unsigned_numeric::<u16>(&lhs_std, &rhs_std),
        DType::UInt32 => add_unsigned_numeric::<u32>(&lhs_std, &rhs_std),
        DType::UInt64 => add_unsigned_numeric::<u64>(&lhs_std, &rhs_std),
        DType::Float32 => add_float_numeric::<f32>(&lhs_std, &rhs_std),
        DType::Float64 => add_float_numeric::<f64>(&lhs_std, &rhs_std),
        DType::Bool | DType::Fixed64 => unreachable!(),
    }
}

pub fn sub(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    if a.dtype() == DType::Fixed64 && b.dtype() == DType::Fixed64 {
        ensure_same_shape(a, b)?;
        return sub_fixed64(a, b);
    }
    ensure_same_shape(a, b)?;
    let dtype = promote_pair(a.dtype(), b.dtype());

    if dtype == DType::Bool {
        if a.dtype() != DType::Bool || b.dtype() != DType::Bool {
            return Err("sub(bool, bool): unsupported mixed boolean subtraction".into());
        }
        let lhs_std = a.to_contiguous()?;
        let rhs_std = b.to_contiguous()?;
        let lhs = lhs_std.try_as_slice::<bool>()?;
        let rhs = rhs_std.try_as_slice::<bool>()?;
        let mut out: Vec<bool> = Vec::with_capacity(lhs.len());
        for (&x, &y) in lhs.iter().zip(rhs.iter()) {
            out.push(x & !y);
        }
        return MatrixBuffer::from_vec(out, a.rows(), a.cols()).map_err(Into::into);
    }

    let lhs_cast = a.cast(dtype)?;
    let rhs_cast = b.cast(dtype)?;
    let lhs_std = lhs_cast.to_contiguous()?;
    let rhs_std = rhs_cast.to_contiguous()?;

    match dtype {
        DType::Int8 => sub_signed_numeric::<i8>(&lhs_std, &rhs_std),
        DType::Int16 => sub_signed_numeric::<i16>(&lhs_std, &rhs_std),
        DType::Int32 => sub_signed_numeric::<i32>(&lhs_std, &rhs_std),
        DType::Int64 => sub_signed_numeric::<i64>(&lhs_std, &rhs_std),
        DType::UInt8 => sub_unsigned_numeric::<u8>(&lhs_std, &rhs_std),
        DType::UInt16 => sub_unsigned_numeric::<u16>(&lhs_std, &rhs_std),
        DType::UInt32 => sub_unsigned_numeric::<u32>(&lhs_std, &rhs_std),
        DType::UInt64 => sub_unsigned_numeric::<u64>(&lhs_std, &rhs_std),
        DType::Float32 => sub_float_numeric::<f32>(&lhs_std, &rhs_std),
        DType::Float64 => sub_float_numeric::<f64>(&lhs_std, &rhs_std),
        DType::Bool | DType::Fixed64 => unreachable!(),
    }
}

pub fn mul(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    if a.dtype() == DType::Fixed64 || b.dtype() == DType::Fixed64 {
        return Err("mul(Fixed64): convert operands to float64 before multiplying".into());
    }
    ensure_same_shape(a, b)?;
    let dtype = promote_pair(a.dtype(), b.dtype());

    if dtype == DType::Bool {
        if a.dtype() != DType::Bool || b.dtype() != DType::Bool {
            return Err("mul(bool, bool): unsupported mixed boolean multiplication".into());
        }
        let lhs_std = a.to_contiguous()?;
        let rhs_std = b.to_contiguous()?;
        let lhs = lhs_std.try_as_slice::<bool>()?;
        let rhs = rhs_std.try_as_slice::<bool>()?;
        let mut out: Vec<bool> = Vec::with_capacity(lhs.len());
        for (&x, &y) in lhs.iter().zip(rhs.iter()) {
            out.push(x && y);
        }
        return MatrixBuffer::from_vec(out, a.rows(), a.cols()).map_err(Into::into);
    }

    let lhs_cast = a.cast(dtype)?;
    let rhs_cast = b.cast(dtype)?;
    let lhs_std = lhs_cast.to_contiguous()?;
    let rhs_std = rhs_cast.to_contiguous()?;

    match dtype {
        DType::Int8 => mul_signed_numeric::<i8>(&lhs_std, &rhs_std),
        DType::Int16 => mul_signed_numeric::<i16>(&lhs_std, &rhs_std),
        DType::Int32 => mul_signed_numeric::<i32>(&lhs_std, &rhs_std),
        DType::Int64 => mul_signed_numeric::<i64>(&lhs_std, &rhs_std),
        DType::UInt8 => mul_unsigned_numeric::<u8>(&lhs_std, &rhs_std),
        DType::UInt16 => mul_unsigned_numeric::<u16>(&lhs_std, &rhs_std),
        DType::UInt32 => mul_unsigned_numeric::<u32>(&lhs_std, &rhs_std),
        DType::UInt64 => mul_unsigned_numeric::<u64>(&lhs_std, &rhs_std),
        DType::Float32 => mul_float_numeric::<f32>(&lhs_std, &rhs_std),
        DType::Float64 => mul_float_numeric::<f64>(&lhs_std, &rhs_std),
        DType::Bool | DType::Fixed64 => unreachable!(),
    }
}

pub fn div(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    if a.dtype() == DType::Fixed64 || b.dtype() == DType::Fixed64 {
        return Err("div(Fixed64): convert operands to float64 before dividing".into());
    }
    ensure_same_shape(a, b)?;
    let dtype = promote_pair(a.dtype(), b.dtype());

    if dtype == DType::Bool {
        if a.dtype() != DType::Bool || b.dtype() != DType::Bool {
            return Err("div(bool, bool): unsupported mixed boolean division".into());
        }
        let lhs_std = a.to_contiguous()?;
        let rhs_std = b.to_contiguous()?;
        let lhs = lhs_std.try_as_slice::<bool>()?;
        let rhs = rhs_std.try_as_slice::<bool>()?;
        let mut out: Vec<bool> = Vec::with_capacity(lhs.len());
        for (&x, &y) in lhs.iter().zip(rhs.iter()) {
            if !y {
                return Err("div(bool, bool): division by false".into());
            }
            out.push(x);
        }
        return MatrixBuffer::from_vec(out, a.rows(), a.cols()).map_err(Into::into);
    }

    let lhs_cast = a.cast(dtype)?;
    let rhs_cast = b.cast(dtype)?;
    let lhs_std = lhs_cast.to_contiguous()?;
    let rhs_std = rhs_cast.to_contiguous()?;

    match dtype {
        DType::Int8 => div_signed_numeric::<i8>(&lhs_std, &rhs_std),
        DType::Int16 => div_signed_numeric::<i16>(&lhs_std, &rhs_std),
        DType::Int32 => div_signed_numeric::<i32>(&lhs_std, &rhs_std),
        DType::Int64 => div_signed_numeric::<i64>(&lhs_std, &rhs_std),
        DType::UInt8 => div_unsigned_numeric::<u8>(&lhs_std, &rhs_std),
        DType::UInt16 => div_unsigned_numeric::<u16>(&lhs_std, &rhs_std),
        DType::UInt32 => div_unsigned_numeric::<u32>(&lhs_std, &rhs_std),
        DType::UInt64 => div_unsigned_numeric::<u64>(&lhs_std, &rhs_std),
        DType::Float32 => div_float_numeric::<f32>(&lhs_std, &rhs_std),
        DType::Float64 => div_float_numeric::<f64>(&lhs_std, &rhs_std),
        DType::Bool | DType::Fixed64 => unreachable!(),
    }
}

pub fn neg(matrix: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    match matrix.dtype() {
        DType::Fixed64 => neg_fixed64(matrix),
        DType::Bool => {
            let std = matrix.to_contiguous()?;
            let src = std.try_as_slice::<bool>()?;
            let mut out: Vec<bool> = Vec::with_capacity(src.len());
            for &value in src {
                out.push(!value);
            }
            MatrixBuffer::from_vec(out, matrix.rows(), matrix.cols()).map_err(Into::into)
        }
        DType::Int8 => {
            let std = matrix.to_contiguous()?;
            neg_signed_numeric::<i8>(&std)
        }
        DType::Int16 => {
            let std = matrix.to_contiguous()?;
            neg_signed_numeric::<i16>(&std)
        }
        DType::Int32 => {
            let std = matrix.to_contiguous()?;
            neg_signed_numeric::<i32>(&std)
        }
        DType::Int64 => {
            let std = matrix.to_contiguous()?;
            neg_signed_numeric::<i64>(&std)
        }
        DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64 => Err(
            "neg: unsigned dtypes are not supported; cast to a signed dtype first".into(),
        ),
        DType::Float32 => {
            let std = matrix.to_contiguous()?;
            neg_float_numeric::<f32>(&std)
        }
        DType::Float64 => {
            let std = matrix.to_contiguous()?;
            neg_float_numeric::<f64>(&std)
        }
    }
}

fn add_signed_numeric<T>(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer>
where
    T: PrimInt + Signed + Bounded + ToPrimitive + FromPrimitive + ElementTrait,
{
    let lhs = a.try_as_slice::<T>()?;
    let rhs = b.try_as_slice::<T>()?;
    let min = T::min_value()
        .to_i128()
        .ok_or_else(|| "failed to convert minimum to i128".to_string())?;
    let max = T::max_value()
        .to_i128()
        .ok_or_else(|| "failed to convert maximum to i128".to_string())?;
    let mut out: Vec<T> = Vec::with_capacity(lhs.len());
    for (&x, &y) in lhs.iter().zip(rhs.iter()) {
        let sum = x.to_i128().unwrap() + y.to_i128().unwrap();
        let clamped = sum.clamp(min, max);
        let value = T::from_i128(clamped)
            .ok_or_else(|| "failed to cast sum back to target dtype".to_string())?;
        out.push(value);
    }
    MatrixBuffer::from_vec(out, a.rows(), a.cols()).map_err(Into::into)
}

fn add_unsigned_numeric<T>(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer>
where
    T: PrimInt + Unsigned + Bounded + ToPrimitive + FromPrimitive + ElementTrait,
{
    let lhs = a.try_as_slice::<T>()?;
    let rhs = b.try_as_slice::<T>()?;
    let max = T::max_value()
        .to_u128()
        .ok_or_else(|| "failed to convert maximum to u128".to_string())?;
    let mut out: Vec<T> = Vec::with_capacity(lhs.len());
    for (&x, &y) in lhs.iter().zip(rhs.iter()) {
        let sum = x.to_u128().unwrap() + y.to_u128().unwrap();
        let clamped = if sum > max { max } else { sum };
        let value = T::from_u128(clamped)
            .ok_or_else(|| "failed to cast sum back to target dtype".to_string())?;
        out.push(value);
    }
    MatrixBuffer::from_vec(out, a.rows(), a.cols()).map_err(Into::into)
}

fn add_float_numeric<T>(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer>
where
    T: Float + ElementTrait,
{
    let lhs = a.try_as_slice::<T>()?;
    let rhs = b.try_as_slice::<T>()?;
    let mut out: Vec<T> = Vec::with_capacity(lhs.len());
    for (&x, &y) in lhs.iter().zip(rhs.iter()) {
        out.push(x + y);
    }
    MatrixBuffer::from_vec(out, a.rows(), a.cols()).map_err(Into::into)
}

fn sub_signed_numeric<T>(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer>
where
    T: PrimInt + Signed + Bounded + ToPrimitive + FromPrimitive + ElementTrait,
{
    let lhs = a.try_as_slice::<T>()?;
    let rhs = b.try_as_slice::<T>()?;
    let min = T::min_value()
        .to_i128()
        .ok_or_else(|| "failed to convert minimum to i128".to_string())?;
    let max = T::max_value()
        .to_i128()
        .ok_or_else(|| "failed to convert maximum to i128".to_string())?;
    let mut out: Vec<T> = Vec::with_capacity(lhs.len());
    for (&x, &y) in lhs.iter().zip(rhs.iter()) {
        let diff = x.to_i128().unwrap() - y.to_i128().unwrap();
        let clamped = diff.clamp(min, max);
        let value = T::from_i128(clamped)
            .ok_or_else(|| "failed to cast difference back to target dtype".to_string())?;
        out.push(value);
    }
    MatrixBuffer::from_vec(out, a.rows(), a.cols()).map_err(Into::into)
}

fn sub_unsigned_numeric<T>(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer>
where
    T: PrimInt + Unsigned + Bounded + ToPrimitive + FromPrimitive + ElementTrait,
{
    let lhs = a.try_as_slice::<T>()?;
    let rhs = b.try_as_slice::<T>()?;
    let mut out: Vec<T> = Vec::with_capacity(lhs.len());
    for (&x, &y) in lhs.iter().zip(rhs.iter()) {
        let left = x.to_u128().unwrap();
        let right = y.to_u128().unwrap();
        let diff = left.saturating_sub(right);
        let value = T::from_u128(diff)
            .ok_or_else(|| "failed to cast difference back to target dtype".to_string())?;
        out.push(value);
    }
    MatrixBuffer::from_vec(out, a.rows(), a.cols()).map_err(Into::into)
}

fn sub_float_numeric<T>(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer>
where
    T: Float + ElementTrait,
{
    let lhs = a.try_as_slice::<T>()?;
    let rhs = b.try_as_slice::<T>()?;
    let mut out: Vec<T> = Vec::with_capacity(lhs.len());
    for (&x, &y) in lhs.iter().zip(rhs.iter()) {
        out.push(x - y);
    }
    MatrixBuffer::from_vec(out, a.rows(), a.cols()).map_err(Into::into)
}

fn mul_signed_numeric<T>(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer>
where
    T: PrimInt + Signed + Bounded + ToPrimitive + FromPrimitive + ElementTrait,
{
    let lhs = a.try_as_slice::<T>()?;
    let rhs = b.try_as_slice::<T>()?;
    let min = T::min_value()
        .to_i128()
        .ok_or_else(|| "failed to convert minimum to i128".to_string())?;
    let max = T::max_value()
        .to_i128()
        .ok_or_else(|| "failed to convert maximum to i128".to_string())?;
    let mut out: Vec<T> = Vec::with_capacity(lhs.len());
    for (&x, &y) in lhs.iter().zip(rhs.iter()) {
        let product = x.to_i128().unwrap() * y.to_i128().unwrap();
        let clamped = product.clamp(min, max);
        let value = T::from_i128(clamped)
            .ok_or_else(|| "failed to cast product back to target dtype".to_string())?;
        out.push(value);
    }
    MatrixBuffer::from_vec(out, a.rows(), a.cols()).map_err(Into::into)
}

fn mul_unsigned_numeric<T>(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer>
where
    T: PrimInt + Unsigned + Bounded + ToPrimitive + FromPrimitive + ElementTrait,
{
    let lhs = a.try_as_slice::<T>()?;
    let rhs = b.try_as_slice::<T>()?;
    let max = T::max_value()
        .to_u128()
        .ok_or_else(|| "failed to convert maximum to u128".to_string())?;
    let mut out: Vec<T> = Vec::with_capacity(lhs.len());
    for (&x, &y) in lhs.iter().zip(rhs.iter()) {
        let product = x.to_u128().unwrap() * y.to_u128().unwrap();
        let clamped = if product > max { max } else { product };
        let value = T::from_u128(clamped)
            .ok_or_else(|| "failed to cast product back to target dtype".to_string())?;
        out.push(value);
    }
    MatrixBuffer::from_vec(out, a.rows(), a.cols()).map_err(Into::into)
}

fn mul_float_numeric<T>(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer>
where
    T: Float + ElementTrait,
{
    let lhs = a.try_as_slice::<T>()?;
    let rhs = b.try_as_slice::<T>()?;
    let mut out: Vec<T> = Vec::with_capacity(lhs.len());
    for (&x, &y) in lhs.iter().zip(rhs.iter()) {
        out.push(x * y);
    }
    MatrixBuffer::from_vec(out, a.rows(), a.cols()).map_err(Into::into)
}

fn div_signed_numeric<T>(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer>
where
    T: PrimInt + Signed + Bounded + ToPrimitive + FromPrimitive + ElementTrait,
{
    let lhs = a.try_as_slice::<T>()?;
    let rhs = b.try_as_slice::<T>()?;
    let min = T::min_value()
        .to_i128()
        .ok_or_else(|| "failed to convert minimum to i128".to_string())?;
    let max = T::max_value()
        .to_i128()
        .ok_or_else(|| "failed to convert maximum to i128".to_string())?;
    let mut out: Vec<T> = Vec::with_capacity(lhs.len());
    for (&x, &y) in lhs.iter().zip(rhs.iter()) {
        let divisor = y.to_i128().unwrap();
        if divisor == 0 {
            return Err("div: division by zero".into());
        }
        let dividend = x.to_i128().unwrap();
        let quotient = dividend / divisor;
        let clamped = quotient.clamp(min, max);
        let value = T::from_i128(clamped)
            .ok_or_else(|| "failed to cast quotient back to target dtype".to_string())?;
        out.push(value);
    }
    MatrixBuffer::from_vec(out, a.rows(), a.cols()).map_err(Into::into)
}

fn div_unsigned_numeric<T>(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer>
where
    T: PrimInt + Unsigned + Bounded + ToPrimitive + FromPrimitive + ElementTrait,
{
    let lhs = a.try_as_slice::<T>()?;
    let rhs = b.try_as_slice::<T>()?;
    let mut out: Vec<T> = Vec::with_capacity(lhs.len());
    for (&x, &y) in lhs.iter().zip(rhs.iter()) {
        let divisor = y.to_u128().unwrap();
        if divisor == 0 {
            return Err("div: division by zero".into());
        }
        let dividend = x.to_u128().unwrap();
        let quotient = dividend / divisor;
        let value = T::from_u128(quotient)
            .ok_or_else(|| "failed to cast quotient back to target dtype".to_string())?;
        out.push(value);
    }
    MatrixBuffer::from_vec(out, a.rows(), a.cols()).map_err(Into::into)
}

fn div_float_numeric<T>(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer>
where
    T: Float + ElementTrait,
{
    let lhs = a.try_as_slice::<T>()?;
    let rhs = b.try_as_slice::<T>()?;
    let mut out: Vec<T> = Vec::with_capacity(lhs.len());
    for (&x, &y) in lhs.iter().zip(rhs.iter()) {
        out.push(x / y);
    }
    MatrixBuffer::from_vec(out, a.rows(), a.cols()).map_err(Into::into)
}

fn neg_signed_numeric<T>(matrix: &MatrixBuffer) -> CoreResult<MatrixBuffer>
where
    T: PrimInt + Signed + Bounded + ToPrimitive + FromPrimitive + ElementTrait,
{
    let values = matrix.try_as_slice::<T>()?;
    let min = T::min_value()
        .to_i128()
        .ok_or_else(|| "failed to convert minimum to i128".to_string())?;
    let max = T::max_value()
        .to_i128()
        .ok_or_else(|| "failed to convert maximum to i128".to_string())?;
    let mut out: Vec<T> = Vec::with_capacity(values.len());
    for &value in values.iter() {
        let negated = -value.to_i128().unwrap();
        let clamped = negated.clamp(min, max);
        let casted = T::from_i128(clamped)
            .ok_or_else(|| "failed to cast negated value back to target dtype".to_string())?;
        out.push(casted);
    }
    MatrixBuffer::from_vec(out, matrix.rows(), matrix.cols()).map_err(Into::into)
}

fn neg_float_numeric<T>(matrix: &MatrixBuffer) -> CoreResult<MatrixBuffer>
where
    T: Float + ElementTrait,
{
    let values = matrix.try_as_slice::<T>()?;
    let mut out: Vec<T> = Vec::with_capacity(values.len());
    for &value in values.iter() {
        out.push(-value);
    }
    MatrixBuffer::from_vec(out, matrix.rows(), matrix.cols()).map_err(Into::into)
}

fn sub_fixed64(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    if a.fixed_scale() != b.fixed_scale() {
        return Err("sub(Fixed64): scale mismatch".into());
    }
    let scale = ensure_fixed64(a)?;
    let vec_a = matrix_to_fixed_vec(a)?;
    let vec_b = matrix_to_fixed_vec(b)?;
    let mut out: Vec<i64> = Vec::with_capacity(vec_a.len());
    for (lhs, rhs) in vec_a.into_iter().zip(vec_b.into_iter()) {
        let (diff, overflow) = lhs.overflowing_sub(rhs);
        if overflow {
            return Err("sub(Fixed64): overflow".into());
        }
        out.push(diff);
    }
    MatrixBuffer::from_fixed_i64_vec(out, a.rows(), a.cols(), scale)
}

fn neg_fixed64(matrix: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    let scale = ensure_fixed64(matrix)?;
    let vec = matrix_to_fixed_vec(matrix)?;
    let mut out: Vec<i64> = Vec::with_capacity(vec.len());
    for value in vec.into_iter() {
        let (negated, overflow) = value.overflowing_neg();
        if overflow {
            return Err("neg(Fixed64): overflow".into());
        }
        out.push(negated);
    }
    MatrixBuffer::from_fixed_i64_vec(out, matrix.rows(), matrix.cols(), scale)
}

pub fn matmul(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    if a.cols() != b.rows() {
        return Err("matmul: inner dimensions do not match".into());
    }
    if a.dtype() == DType::Fixed64 || b.dtype() == DType::Fixed64 {
        return Err("matmul(Fixed64): convert operands to float64 before multiplying".into());
    }
    let dtype = promote_pair(a.dtype(), b.dtype());

    let a_matrix = Array2::from_shape_vec((a.rows(), a.cols()), a.to_f64_vec())
        .map_err(|_| "failed to reshape left matrix")?;
    let b_matrix = Array2::from_shape_vec((b.rows(), b.cols()), b.to_f64_vec())
        .map_err(|_| "failed to reshape right matrix")?;
    let product = a_matrix.dot(&b_matrix);

    MatrixBuffer::from_f64_vec(
        dtype,
        product.nrows(),
        product.ncols(),
        product.into_raw_vec(),
    )
    .map_err(Into::into)
}

pub fn clip(buffer: &MatrixBuffer, min: f64, max: f64) -> CoreResult<MatrixBuffer> {
    if min > max {
        return Err("clip: min must be <= max".into());
    }
    if buffer.dtype() == DType::Fixed64 {
        return Err("clip(Fixed64): convert to float64 before clipping".into());
    }
    let dtype = buffer.dtype();
    let clipped: Vec<f64> = buffer
        .to_f64_vec()
        .into_iter()
        .map(|v| v.clamp(min, max))
        .collect();

    MatrixBuffer::from_f64_vec(dtype, buffer.rows(), buffer.cols(), clipped).map_err(Into::into)
}

pub fn where_select(
    condition: &MatrixBuffer,
    a: &MatrixBuffer,
    b: &MatrixBuffer,
) -> CoreResult<MatrixBuffer> {
    where_select_multi(&[condition], &[a], Some(b))
}

pub fn where_select_multi(
    conditions: &[&MatrixBuffer],
    choices: &[&MatrixBuffer],
    default: Option<&MatrixBuffer>,
) -> CoreResult<MatrixBuffer> {
    if conditions.len() != choices.len() {
        return Err("where_select_multi: number of conditions must equal number of choices".into());
    }
    let mut target_shape: Option<(usize, usize)> = None;
    for matrix in conditions.iter().chain(choices.iter()).copied() {
        target_shape = Some(match target_shape {
            None => (matrix.rows(), matrix.cols()),
            Some(shape) => broadcast_pair(shape, (matrix.rows(), matrix.cols()))
                .ok_or_else(|| "where_select_multi: cannot broadcast inputs".to_string())?,
        });
    }
    if let Some(default_matrix) = default {
        target_shape = Some(match target_shape {
            None => (default_matrix.rows(), default_matrix.cols()),
            Some(shape) => broadcast_pair(shape, (default_matrix.rows(), default_matrix.cols()))
                .ok_or_else(|| "where_select_multi: cannot broadcast default".to_string())?,
        });
    }
    let (rows, cols) = target_shape
        .ok_or_else(|| "where_select_multi: requires at least one input".to_string())?;

    let mut dtype_candidates: Vec<DType> = choices.iter().map(|m| m.dtype()).collect();
    if let Some(default_matrix) = default {
        dtype_candidates.push(default_matrix.dtype());
    }
    let dtype =
        promote_many(&dtype_candidates).ok_or("where_select_multi: unable to determine dtype")?;

    if dtype == DType::Fixed64 {
        return where_select_multi_fixed64(conditions, choices, default, rows, cols);
    }

    let mut result = if let Some(default_matrix) = default {
        default_matrix
            .broadcast_to(rows, cols)
            .and_then(|view| view.cast(dtype))
            .and_then(|view| view.to_contiguous())
            .map_err(|err| format!("where_select_multi: {err}"))?
    } else {
        let zeros = vec![0.0; rows * cols];
        MatrixBuffer::from_f64_vec(dtype, rows, cols, zeros)
            .map_err(|err| format!("where_select_multi: {err}"))?
    };

    match dtype {
        DType::Bool => assign_where::<bool>(&mut result, dtype, conditions, choices, rows, cols)?,
        DType::Int8 => assign_where::<i8>(&mut result, dtype, conditions, choices, rows, cols)?,
        DType::Int16 => assign_where::<i16>(&mut result, dtype, conditions, choices, rows, cols)?,
        DType::Int32 => assign_where::<i32>(&mut result, dtype, conditions, choices, rows, cols)?,
        DType::Int64 => assign_where::<i64>(&mut result, dtype, conditions, choices, rows, cols)?,
        DType::UInt8 => assign_where::<u8>(&mut result, dtype, conditions, choices, rows, cols)?,
        DType::UInt16 => assign_where::<u16>(&mut result, dtype, conditions, choices, rows, cols)?,
        DType::UInt32 => assign_where::<u32>(&mut result, dtype, conditions, choices, rows, cols)?,
        DType::UInt64 => assign_where::<u64>(&mut result, dtype, conditions, choices, rows, cols)?,
        DType::Float32 => assign_where::<f32>(&mut result, dtype, conditions, choices, rows, cols)?,
        DType::Float64 => assign_where::<f64>(&mut result, dtype, conditions, choices, rows, cols)?,
        DType::Fixed64 => unreachable!("fixed64 handled earlier"),
    }

    Ok(result)
}

pub fn concat(axis: usize, matrices: &[MatrixBuffer]) -> CoreResult<MatrixBuffer> {
    if matrices.is_empty() {
        return Err("concat: expected at least one matrix".into());
    }
    if axis > 1 {
        return Err("concat: axis out of range".into());
    }

    let dtype = promote_many(&matrices.iter().map(|m| m.dtype()).collect::<Vec<_>>())
        .ok_or("concat: unable to determine dtype")?;
    let casted: Vec<MatrixBuffer> = matrices
        .iter()
        .map(|m| m.cast(dtype))
        .collect::<Result<_, _>>()?;

    if dtype == DType::Fixed64 {
        return concat_fixed64(axis, &casted);
    }

    let rows = casted[0].rows();
    let cols = casted[0].cols();

    match axis {
        0 => {
            let mut total_rows = 0;
            for m in &casted {
                if m.cols() != cols {
                    return Err("concat axis 0: column sizes differ".into());
                }
                total_rows += m.rows();
            }
            let mut data = Vec::with_capacity(total_rows * cols * dtype.size_of());
            for m in &casted {
                if let Some(bytes) = m.as_byte_slice() {
                    data.extend_from_slice(bytes);
                } else {
                    let bytes = m.to_contiguous_bytes_vec();
                    data.extend_from_slice(&bytes);
                }
            }
            MatrixBuffer::from_bytes(dtype, total_rows, cols, data).map_err(Into::into)
        }
        1 => {
            let mut total_cols = 0;
            for m in &casted {
                if m.rows() != rows {
                    return Err("concat axis 1: row sizes differ".into());
                }
                total_cols += m.cols();
            }
            let mut data = Vec::with_capacity(rows * total_cols * dtype.size_of());
            for row in 0..rows {
                for m in &casted {
                    let row_width = m.cols() * dtype.size_of();
                    if let Some(bytes) = m.as_byte_slice() {
                        let start = row * row_width;
                        let end = start + row_width;
                        data.extend_from_slice(&bytes[start..end]);
                    } else {
                        let mut temp = vec![0u8; row_width];
                        m.copy_row_into(row, &mut temp);
                        data.extend_from_slice(&temp);
                    }
                }
            }
            MatrixBuffer::from_bytes(dtype, rows, total_cols, data).map_err(Into::into)
        }
        _ => unreachable!(),
    }
}

pub fn stack(axis: usize, matrices: &[MatrixBuffer]) -> CoreResult<MatrixBuffer> {
    concat(axis, matrices)
}

pub fn transpose(matrix: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    matrix
        .transpose()
        .map_err(|err| format!("transpose: {err}"))
}

pub fn broadcast_to(matrix: &MatrixBuffer, rows: usize, cols: usize) -> CoreResult<MatrixBuffer> {
    matrix
        .broadcast_to(rows, cols)
        .map_err(|err| format!("broadcast_to: {err}"))
}

pub fn take(matrix: &MatrixBuffer, axis: usize, indices: &[isize]) -> CoreResult<MatrixBuffer> {
    matrix
        .take(axis, indices)
        .map_err(|err| format!("take: {err}"))
}

pub fn put(
    matrix: &MatrixBuffer,
    axis: usize,
    indices: &[isize],
    values: &MatrixBuffer,
) -> CoreResult<MatrixBuffer> {
    match axis {
        0 => {
            if values.cols() != matrix.cols() || values.rows() != indices.len() {
                return Err("put axis 0: values shape must match indices x cols".into());
            }
            let col_indices: Vec<isize> = (0..matrix.cols()).map(|c| c as isize).collect();
            matrix
                .scatter(indices, &col_indices, values, false)
                .map_err(|err| format!("put axis 0: {err}"))
        }
        1 => {
            if values.rows() != matrix.rows() || values.cols() != indices.len() {
                return Err("put axis 1: values shape must match rows x indices".into());
            }
            let row_indices: Vec<isize> = (0..matrix.rows()).map(|r| r as isize).collect();
            matrix
                .scatter(&row_indices, indices, values, false)
                .map_err(|err| format!("put axis 1: {err}"))
        }
        _ => Err("put: axis must be 0 or 1".into()),
    }
}

pub fn gather(
    matrix: &MatrixBuffer,
    row_indices: &[isize],
    col_indices: &[isize],
) -> CoreResult<MatrixBuffer> {
    matrix
        .gather(row_indices, col_indices, false)
        .map_err(|err| format!("gather: {err}"))
}

pub fn gather_pairs(
    matrix: &MatrixBuffer,
    row_indices: &[isize],
    col_indices: &[isize],
) -> CoreResult<MatrixBuffer> {
    matrix
        .gather(row_indices, col_indices, true)
        .map_err(|err| format!("gather_pairs: {err}"))
}

pub fn scatter(
    matrix: &MatrixBuffer,
    row_indices: &[isize],
    col_indices: &[isize],
    values: &MatrixBuffer,
) -> CoreResult<MatrixBuffer> {
    matrix
        .scatter(row_indices, col_indices, values, false)
        .map_err(|err| format!("scatter: {err}"))
}

pub fn scatter_pairs(
    matrix: &MatrixBuffer,
    row_indices: &[isize],
    col_indices: &[isize],
    values: &MatrixBuffer,
) -> CoreResult<MatrixBuffer> {
    matrix
        .scatter(row_indices, col_indices, values, true)
        .map_err(|err| format!("scatter_pairs: {err}"))
}

#[cfg(feature = "linalg")]
pub fn svd(buffer: &MatrixBuffer) -> CoreResult<(MatrixBuffer, MatrixBuffer, MatrixBuffer)> {
    let matrix = DMatrix::from_row_slice(buffer.rows(), buffer.cols(), &buffer.to_f64_vec());
    let svd = matrix.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| "SVD did not return U matrix".to_string())?;
    let vt = svd
        .v_t
        .ok_or_else(|| "SVD did not return V^T matrix".to_string())?;
    let sigma = MatrixBuffer::from_f64_vec(
        DType::Float64,
        1,
        svd.singular_values.len(),
        svd.singular_values.iter().copied().collect(),
    )?;
    let u_buf =
        MatrixBuffer::from_f64_vec(DType::Float64, u.nrows(), u.ncols(), u.as_slice().to_vec())?;
    let vt_buf = MatrixBuffer::from_f64_vec(
        DType::Float64,
        vt.nrows(),
        vt.ncols(),
        vt.as_slice().to_vec(),
    )?;
    Ok((u_buf, sigma, vt_buf))
}

#[cfg(feature = "linalg")]
pub fn qr(buffer: &MatrixBuffer) -> CoreResult<(MatrixBuffer, MatrixBuffer)> {
    let matrix = DMatrix::from_row_slice(buffer.rows(), buffer.cols(), &buffer.to_f64_vec());
    let qr = matrix.qr();
    let q_buf = MatrixBuffer::from_f64_vec(
        DType::Float64,
        qr.q().nrows(),
        qr.q().ncols(),
        qr.q().as_slice().to_vec(),
    )?;
    let r_buf = MatrixBuffer::from_f64_vec(
        DType::Float64,
        qr.r().nrows(),
        qr.r().ncols(),
        qr.r().as_slice().to_vec(),
    )?;
    Ok((q_buf, r_buf))
}

#[cfg(feature = "linalg")]
pub fn solve(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    if a.rows() != a.cols() {
        return Err("solve: matrix A must be square".into());
    }
    if a.rows() != b.rows() {
        return Err("solve: RHS has incompatible shape".into());
    }
    let a_mat = DMatrix::from_row_slice(a.rows(), a.cols(), &a.to_f64_vec());
    let b_mat = DMatrix::from_row_slice(b.rows(), b.cols(), &b.to_f64_vec());
    let solution = a_mat
        .lu()
        .solve(&b_mat)
        .ok_or_else(|| "solve failed: singular matrix".to_string())?;
    MatrixBuffer::from_f64_vec(
        DType::Float64,
        solution.nrows(),
        solution.ncols(),
        solution.as_slice().to_vec(),
    )
    .map_err(Into::into)
}

#[cfg(feature = "linalg")]
pub fn eigen(buffer: &MatrixBuffer) -> CoreResult<(MatrixBuffer, MatrixBuffer)> {
    if buffer.rows() != buffer.cols() {
        return Err("eigen: matrix must be square".into());
    }
    let mat = DMatrix::from_row_slice(buffer.rows(), buffer.cols(), &buffer.to_f64_vec());
    let eigen = SymmetricEigen::new(mat);
    let values = MatrixBuffer::from_f64_vec(
        DType::Float64,
        eigen.eigenvalues.len(),
        1,
        eigen.eigenvalues.as_slice().to_vec(),
    )?;
    let vectors = MatrixBuffer::from_f64_vec(
        DType::Float64,
        eigen.eigenvectors.nrows(),
        eigen.eigenvectors.ncols(),
        eigen.eigenvectors.as_slice().to_vec(),
    )?;
    Ok((values, vectors))
}

#[cfg(feature = "npy")]
pub fn read_npy_matrix(data: &[u8]) -> CoreResult<MatrixBuffer> {
    let mut cursor = Cursor::new(data);
    let array: Array2<f64> =
        Array2::read_npy(&mut cursor).map_err(|e| format!("failed to read npy: {e}"))?;
    MatrixBuffer::from_f64_vec(
        DType::Float64,
        array.nrows(),
        array.ncols(),
        array.into_raw_vec(),
    )
    .map_err(Into::into)
}

#[cfg(feature = "npy")]
pub fn write_npy_matrix(buffer: &MatrixBuffer) -> CoreResult<Vec<u8>> {
    if buffer.dtype() == DType::Fixed64 {
        return Err(
            "write_npy(Fixed64): convert to float64 or serialise bigint payload manually".into(),
        );
    }
    let array = Array2::from_shape_vec((buffer.rows(), buffer.cols()), buffer.to_f64_vec())
        .map_err(|_| "write_npy: failed to reshape matrix")?;
    let mut cursor = Cursor::new(Vec::new());
    array
        .write_npy(&mut cursor)
        .map_err(|e| format!("failed to write npy: {e}"))?;
    Ok(cursor.into_inner())
}

#[cfg(feature = "npy")]
pub fn read_npz_matrices(data: &[u8]) -> CoreResult<Vec<(String, MatrixBuffer)>> {
    let cursor = Cursor::new(data);
    let mut reader = NpzReader::new(cursor).map_err(|e| format!("failed to read npz: {e}"))?;
    let mut results = Vec::new();
    let names = reader
        .names()
        .map_err(|e| format!("failed to list npz names: {e}"))?;
    for name in names {
        let array: Array2<f64> = reader
            .by_name(&name)
            .map_err(|e| format!("npz entry {name} failed: {e}"))?;
        let buffer = MatrixBuffer::from_f64_vec(
            DType::Float64,
            array.nrows(),
            array.ncols(),
            array.into_raw_vec(),
        )?;
        results.push((name, buffer));
    }
    Ok(results)
}

#[cfg(feature = "npy")]
pub fn write_npz_matrices(entries: &[(&str, MatrixBuffer)]) -> CoreResult<Vec<u8>> {
    let cursor = Cursor::new(Vec::new());
    let mut writer = NpzWriter::new(cursor);
    for (name, buffer) in entries {
        if buffer.dtype() == DType::Fixed64 {
            return Err(format!(
                "write_npz(Fixed64): entry '{name}' cannot be written; convert to float64 first"
            ));
        }
        let array = Array2::from_shape_vec((buffer.rows(), buffer.cols()), buffer.to_f64_vec())
            .map_err(|_| "write_npz: failed to reshape matrix")?;
        writer
            .add_array(*name, &array)
            .map_err(|e| format!("failed to add array {name}: {e}"))?;
    }
    writer
        .finish()
        .map(|cursor| cursor.into_inner())
        .map_err(|e| format!("failed to finish npz: {e}"))
}

#[cfg(not(feature = "npy"))]
pub fn read_npy_matrix(_data: &[u8]) -> CoreResult<MatrixBuffer> {
    Err("npy support is disabled".into())
}

#[cfg(not(feature = "npy"))]
pub fn write_npy_matrix(_buffer: &MatrixBuffer) -> CoreResult<Vec<u8>> {
    Err("npy support is disabled".into())
}

#[cfg(not(feature = "npy"))]
pub fn read_npz_matrices(_data: &[u8]) -> CoreResult<Vec<(String, MatrixBuffer)>> {
    Err("npz support is disabled".into())
}

#[cfg(not(feature = "npy"))]
pub fn write_npz_matrices(_entries: &[(&str, MatrixBuffer)]) -> CoreResult<Vec<u8>> {
    Err("npz support is disabled".into())
}

fn ensure_same_shape(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<()> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err("shape mismatch".into());
    }
    Ok(())
}

fn broadcast_pair(a: (usize, usize), b: (usize, usize)) -> Option<(usize, usize)> {
    Some((broadcast_dim(a.0, b.0)?, broadcast_dim(a.1, b.1)?))
}

fn broadcast_dim(a: usize, b: usize) -> Option<usize> {
    if a == b {
        Some(a)
    } else if a == 1 {
        Some(b)
    } else if b == 1 {
        Some(a)
    } else {
        None
    }
}

fn assign_where<T: crate::element::Element + Copy>(
    result: &mut MatrixBuffer,
    dtype: DType,
    conditions: &[&MatrixBuffer],
    choices: &[&MatrixBuffer],
    rows: usize,
    cols: usize,
) -> Result<(), String> {
    let dest = result.as_slice_mut::<T>().ok_or_else(|| {
        format!(
            "where_select_multi: failed to access {} slice",
            dtype.as_str()
        )
    })?;
    for (cond, choice) in conditions.iter().zip(choices.iter()) {
        let mask = cond
            .broadcast_to(rows, cols)
            .map_err(|e| format!("where_select_multi condition broadcast failed: {e}"))?
            .to_bool_vec();
        let cast_choice = choice
            .broadcast_to(rows, cols)
            .map_err(|e| format!("where_select_multi choice broadcast failed: {e}"))?
            .cast(dtype)
            .and_then(|buf| buf.to_contiguous())
            .map_err(|e| format!("where_select_multi choice cast failed: {e}"))?;
        let src = cast_choice
            .as_slice::<T>()
            .ok_or_else(|| "where_select_multi: unable to read casted choice slice".to_string())?;
        apply_mask(dest, src, &mask);
    }
    Ok(())
}

fn apply_mask<T: Copy>(dest: &mut [T], src: &[T], mask: &[bool]) {
    for (index, flag) in mask.iter().enumerate() {
        if *flag {
            dest[index] = src[index];
        }
    }
}

// ---------------------------------------------------------------------
// Tests for stable reductions
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MatrixBuffer;

    #[test]
    fn test_pairwise_sum_simple() {
        let buf = MatrixBuffer::from_f64_vec(
            crate::dtype::DType::Float64,
            1,
            5,
            vec![1e16, 1.0, 1.0, -1e16, 1.0],
        )
        .unwrap();
        let naive: f64 = buf.to_f64_vec().iter().sum();
        let stable_buf = sum(&buf, None).expect("sum");
        assert_eq!(stable_buf.dtype(), DType::Float64);
        let stable = stable_buf.try_as_slice::<f64>().unwrap()[0];
        // naive may be 1.0 or 0.0 depending on accumulation; stable should be 3.0 or close
        assert!(
            stable > 0.5,
            "stable sum too small: {} (naive={})",
            stable,
            naive
        );
    }

    #[test]
    fn test_welford_mean_var() {
        let data: Vec<f64> = (0..1000).map(|i| (i as f64) / 10.0).collect();
        let buf =
            MatrixBuffer::from_f64_vec(crate::dtype::DType::Float64, 1, data.len(), data).unwrap();
        let (mean, var) = welford_mean_variance(&buf, false);
        assert!((mean - 49.95).abs() < 1e-9);
        assert!(var > 0.0);
    }

    #[test]
    fn test_fixed64_concat_preserves_scale() {
        let a = MatrixBuffer::from_fixed_i64_vec(vec![150, 250], 1, 2, 2).unwrap();
        let b = MatrixBuffer::from_fixed_i64_vec(vec![350, 450], 1, 2, 2).unwrap();
        let stacked = concat(0, &[a.clone(), b.clone()]).expect("concat axis 0");
        assert_eq!(stacked.fixed_scale(), Some(2));
        assert_eq!(stacked.rows(), 2);
        assert_eq!(stacked.cols(), 2);
        assert_eq!(stacked.to_f64_vec(), vec![1.5, 2.5, 3.5, 4.5]);

        let wide = concat(1, &[a.clone(), b.clone()]).expect("concat axis 1");
        assert_eq!(wide.fixed_scale(), Some(2));
        assert_eq!(wide.rows(), 1);
        assert_eq!(wide.cols(), 4);
        assert_eq!(wide.to_f64_vec(), vec![1.5, 2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_fixed64_take_and_gather_preserve_scale() {
        let base = MatrixBuffer::from_fixed_i64_vec(vec![100, 200, 300, 400], 2, 2, 1).unwrap();
        let take_row = take(&base, 0, &[1]).expect("take row");
        assert_eq!(take_row.fixed_scale(), Some(1));
        assert_eq!(take_row.rows(), 1);
        assert_eq!(take_row.to_f64_vec(), vec![30.0, 40.0]);

        let gather_pairs = gather(&base, &[0, 1], &[1]).expect("gather outer");
        assert_eq!(gather_pairs.fixed_scale(), Some(1));
        assert_eq!(gather_pairs.to_f64_vec(), vec![20.0, 40.0]);
    }

    #[test]
    fn test_fixed64_where_select() {
        let condition =
            MatrixBuffer::from_vec(vec![true, false, true, false], 2, 2).expect("condition");
        let a = MatrixBuffer::from_fixed_i64_vec(vec![1234, 2234, 3234, 4234], 2, 2, 2).unwrap();
        let b = MatrixBuffer::from_fixed_i64_vec(vec![1000, 2000, 3000, 4000], 2, 2, 2).unwrap();
        let result = where_select(&condition, &a, &b).expect("where select");
        assert_eq!(result.fixed_scale(), Some(2));
        assert_eq!(result.to_f64_vec(), vec![12.34, 20.00, 32.34, 40.00]);
    }

    #[test]
    fn test_fixed64_transpose_preserves_scale() {
        let base = MatrixBuffer::from_fixed_i64_vec(vec![100, 200, 300, 400], 2, 2, 1).unwrap();
        let transposed = transpose(&base).expect("transpose");
        assert_eq!(transposed.fixed_scale(), Some(1));
        assert_eq!(transposed.rows(), 2);
        assert_eq!(transposed.cols(), 2);
        assert_eq!(transposed.to_f64_vec(), vec![10.0, 30.0, 20.0, 40.0]);
    }

    #[test]
    fn test_fixed64_broadcast_preserves_scale() {
        let base = MatrixBuffer::from_fixed_i64_vec(vec![1050, 2050], 1, 2, 2).unwrap();
        let expanded = broadcast_to(&base, 3, 2).expect("broadcast");
        assert_eq!(expanded.fixed_scale(), Some(2));
        assert_eq!(expanded.rows(), 3);
        assert_eq!(expanded.cols(), 2);
        assert_eq!(
            expanded.to_f64_vec(),
            vec![10.5, 20.5, 10.5, 20.5, 10.5, 20.5]
        );
    }

    #[test]
    fn test_sub_bool_difference() {
        let lhs = MatrixBuffer::from_vec(vec![true, true, false], 1, 3).unwrap();
        let rhs = MatrixBuffer::from_vec(vec![false, true, true], 1, 3).unwrap();
        let diff = sub(&lhs, &rhs).expect("bool subtraction");
        assert_eq!(diff.dtype(), DType::Bool);
        assert_eq!(diff.to_bool_vec(), vec![true, false, false]);
    }

    #[test]
    fn test_mul_uint8_saturates() {
        let lhs = MatrixBuffer::from_vec(vec![200u8, 255u8], 1, 2).unwrap();
        let rhs = MatrixBuffer::from_vec(vec![2u8, 2u8], 1, 2).unwrap();
        let prod = mul(&lhs, &rhs).expect("uint8 multiply");
        assert_eq!(prod.dtype(), DType::UInt8);
        let values = prod.try_as_slice::<u8>().unwrap();
        assert_eq!(values, &[255, 255]);
    }

    #[test]
    fn test_div_detects_zero() {
        let lhs = MatrixBuffer::from_vec(vec![7i32, -9i32], 1, 2).unwrap();
        let zeros = MatrixBuffer::from_vec(vec![0i32, 1i32], 1, 2).unwrap();
        let err = div(&lhs, &zeros).expect_err("division by zero should fail");
        assert!(err.contains("division by zero"));
    }

    #[test]
    fn test_neg_behaviour() {
        let signed = MatrixBuffer::from_vec(vec![1i32, -2i32], 1, 2).unwrap();
        let neg_signed = neg(&signed).expect("neg signed");
        assert_eq!(neg_signed.try_as_slice::<i32>().unwrap(), &[-1, 2]);

        let unsigned = MatrixBuffer::from_vec(vec![1u8, 2u8], 1, 2).unwrap();
        let err = neg(&unsigned).expect_err("neg unsigned should error");
        assert!(err.contains("unsigned dtypes"));

        let fixed = MatrixBuffer::from_fixed_i64_vec(vec![150, -20], 1, 2, 1).unwrap();
        let neg_fixed = neg(&fixed).expect("neg fixed");
        assert_eq!(neg_fixed.fixed_scale(), Some(1));
        assert_eq!(neg_fixed.to_f64_vec(), vec![-15.0, 2.0]);
    }

    #[test]
    fn test_sum_int32_accumulates_to_int64() {
        let buf = MatrixBuffer::from_vec(vec![1i32, 2, 3, 4], 2, 2).unwrap();
        let reduced = sum(&buf, None).expect("sum int32");
        assert_eq!(reduced.dtype(), DType::Int64);
        let values = reduced.try_as_slice::<i64>().unwrap();
        assert_eq!(values, &[10]);
    }

    #[test]
    fn test_sum_with_target_dtype() {
        let buf = MatrixBuffer::from_vec(vec![1u16, 2u16, 3u16], 1, 3).unwrap();
        let reduced = sum(&buf, Some(DType::UInt32)).expect("sum u16 -> u32");
        assert_eq!(reduced.dtype(), DType::UInt32);
        let values = reduced.try_as_slice::<u32>().unwrap();
        assert_eq!(values, &[6]);
    }

    #[test]
    fn test_dot_accumulates_to_int64() {
        let lhs = MatrixBuffer::from_vec(vec![1i16, 2i16, 3i16], 1, 3).unwrap();
        let rhs = MatrixBuffer::from_vec(vec![4i16, 5i16, 6i16], 1, 3).unwrap();
        let reduced = dot(&lhs, &rhs, None).expect("dot i16");
        assert_eq!(reduced.dtype(), DType::Int64);
        let values = reduced.try_as_slice::<i64>().unwrap();
        assert_eq!(values, &[32]);
    }
}
