pub mod buffer;
pub mod dtype;
pub mod element;
pub mod metrics;

use buffer::MatrixBuffer;
use dtype::{promote_many, promote_pair, DType};

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

fn pairwise_sum_slice(data: &[f64]) -> f64 {
    match data.len() {
        0 => 0.0,
        1 => data[0],
        2 => data[0] + data[1],
        _ => {
            let mid = data.len() / 2;
            pairwise_sum_slice(&data[..mid]) + pairwise_sum_slice(&data[mid..])
        }
    }
}

pub fn sum_pairwise(buffer: &MatrixBuffer) -> f64 {
    let v = buffer.to_f64_vec();
    pairwise_sum_slice(&v)
}

pub fn dot_pairwise(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<f64> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err("dot_pairwise: shape mismatch".into());
    }
    let av = a.to_f64_vec();
    let bv = b.to_f64_vec();
    let mut prod: Vec<f64> = Vec::with_capacity(av.len());
    for i in 0..av.len() {
        prod.push(av[i] * bv[i]);
    }
    Ok(pairwise_sum_slice(&prod))
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
    let var = if sample && n > 1.0 { m2 / (n - 1.0) } else { m2 / n };
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
    let out_rows = a.rows();
    let out_cols = a.cols();

    let lhs = a.to_f64_vec();
    let rhs = b.to_f64_vec();
    let result: Vec<f64> = lhs.into_iter().zip(rhs).map(|(x, y)| x + y).collect();

    MatrixBuffer::from_f64_vec(dtype, out_rows, out_cols, result).map_err(Into::into)
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
        return Err("write_npy(Fixed64): convert to float64 or serialise bigint payload manually".into());
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
// Compatibility helpers (legacy f64 API)
// ---------------------------------------------------------------------

use ndarray::{concatenate, Array1, ArrayView2, ArrayViewMut2, Axis};

pub fn add_inplace(
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
    mut out: ArrayViewMut2<'_, f64>,
) {
    assert_eq!(a.dim(), b.dim());
    assert_eq!(a.dim(), out.dim());
    out.assign(&(&a + &b));
}

pub fn matmul_legacy(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Array2<f64> {
    a.dot(&b)
}

pub fn clip_legacy(a: ArrayView2<'_, f64>, min: f64, max: f64) -> Array2<f64> {
    a.map(|value| value.clamp(min, max))
}

pub fn where_select_legacy(
    condition: ArrayView2<'_, f64>,
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
) -> CoreResult<Array2<f64>> {
    if condition.dim() != a.dim() || condition.dim() != b.dim() {
        return Err("Shape mismatch in where operation".into());
    }
    let mut out = Array2::<f64>::zeros(a.dim());
    for ((cond, (lhs, rhs)), out_cell) in condition
        .iter()
        .zip(a.iter().zip(b.iter()))
        .zip(out.iter_mut())
    {
        *out_cell = if *cond != 0.0 { *lhs } else { *rhs };
    }
    Ok(out)
}

pub fn concat_legacy(axis: usize, matrices: &[ArrayView2<'_, f64>]) -> CoreResult<Array2<f64>> {
    if matrices.is_empty() {
        return Err("concat expects at least one matrix".into());
    }
    let axis = Axis(axis);
    concatenate(axis, matrices).map_err(|e| e.to_string())
}

#[cfg(feature = "linalg")]
pub fn svd_legacy(a: ArrayView2<'_, f64>) -> CoreResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    let matrix = DMatrix::from_row_slice(a.nrows(), a.ncols(), a.as_slice().unwrap());
    let svd = matrix.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| "SVD did not return U matrix".to_string())?;
    let vt = svd
        .v_t
        .ok_or_else(|| "SVD did not return V^T matrix".to_string())?;
    let singular = Array1::from_vec(svd.singular_values.iter().copied().collect());
    let u_array = Array2::from_shape_vec((u.nrows(), u.ncols()), u.as_slice().to_vec()).unwrap();
    let vt_array =
        Array2::from_shape_vec((vt.nrows(), vt.ncols()), vt.as_slice().to_vec()).unwrap();
    Ok((u_array, singular, vt_array))
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
        let stable = sum_pairwise(&buf);
        // naive may be 1.0 or 0.0 depending on accumulation; stable should be 3.0 or close
        assert!(stable > 0.5, "stable sum too small: {} (naive={})", stable, naive);
    }

    #[test]
    fn test_welford_mean_var() {
        let data: Vec<f64> = (0..1000).map(|i| (i as f64) / 10.0).collect();
        let buf = MatrixBuffer::from_f64_vec(crate::dtype::DType::Float64, 1, data.len(), data).unwrap();
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
        let base =
            MatrixBuffer::from_fixed_i64_vec(vec![100, 200, 300, 400], 2, 2, 1).unwrap();
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
        assert_eq!(expanded.to_f64_vec(), vec![10.5, 20.5, 10.5, 20.5, 10.5, 20.5]);
    }
}


