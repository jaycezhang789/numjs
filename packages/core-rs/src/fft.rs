use crate::{buffer::MatrixBuffer, CoreResult};
use rustfft::{num_complex::Complex64, Fft, FftPlanner};

fn ensure_float64<'a>(matrix: &'a MatrixBuffer, name: &str) -> CoreResult<&'a [f64]> {
    if matrix.dtype() != crate::dtype::DType::Float64 {
        return Err(format!("{name}: only float64 matrices are supported").into());
    }
    matrix
        .try_as_slice::<f64>()
        .map_err(|_| format!("{name}: matrix is not contiguous float64").into())
}

fn clone_data(matrix: &MatrixBuffer, name: &str) -> CoreResult<Vec<f64>> {
    let slice = ensure_float64(matrix, name)?;
    Ok(slice.to_vec())
}

fn fft_axis_inplace(
    real: &mut [f64],
    imag: &mut [f64],
    rows: usize,
    cols: usize,
    axis: usize,
    inverse: bool,
) -> CoreResult<()> {
    match axis {
        0 => {
            let len = rows;
            if len == 0 {
                return Ok(());
            }
            let mut planner = FftPlanner::<f64>::new();
            let fft = if inverse {
                planner.plan_fft_inverse(len)
            } else {
                planner.plan_fft_forward(len)
            };
            for col in 0..cols {
                process_axis(real, imag, cols, col, len, &*fft, inverse, true);
            }
            Ok(())
        }
        1 => {
            let len = cols;
            if len == 0 {
                return Ok(());
            }
            let mut planner = FftPlanner::<f64>::new();
            let fft = if inverse {
                planner.plan_fft_inverse(len)
            } else {
                planner.plan_fft_forward(len)
            };
            for row in 0..rows {
                process_axis(real, imag, cols, row, len, &*fft, inverse, false);
            }
            Ok(())
        }
        _ => Err("fft: axis must be 0 or 1".into()),
    }
}

fn process_axis(
    real: &mut [f64],
    imag: &mut [f64],
    cols: usize,
    index: usize,
    len: usize,
    fft: &dyn Fft<f64>,
    inverse: bool,
    column_major: bool,
) {
    let mut buffer = Vec::with_capacity(len);
    if column_major {
        for row in 0..len {
            let idx = row * cols + index;
            buffer.push(Complex64::new(real[idx], imag[idx]));
        }
    } else {
        let offset = index * cols;
        for col in 0..len {
            let idx = offset + col;
            buffer.push(Complex64::new(real[idx], imag[idx]));
        }
    }
    fft.process(&mut buffer);
    if inverse {
        let scale = len as f64;
        for value in &mut buffer {
            *value /= scale;
        }
    }
    if column_major {
        for row in 0..len {
            let idx = row * cols + index;
            real[idx] = buffer[row].re;
            imag[idx] = buffer[row].im;
        }
    } else {
        let offset = index * cols;
        for col in 0..len {
            let idx = offset + col;
            real[idx] = buffer[col].re;
            imag[idx] = buffer[col].im;
        }
    }
}

fn matrices_from_components(
    real: Vec<f64>,
    imag: Vec<f64>,
    rows: usize,
    cols: usize,
) -> CoreResult<(MatrixBuffer, MatrixBuffer)> {
    let real_matrix = MatrixBuffer::from_vec(real, rows, cols).map_err(|err| format!("fft: {err}"))?;
    let imag_matrix = MatrixBuffer::from_vec(imag, rows, cols).map_err(|err| format!("fft: {err}"))?;
    Ok((real_matrix, imag_matrix))
}

pub fn fft1d(matrix: &MatrixBuffer, axis: usize) -> CoreResult<(MatrixBuffer, MatrixBuffer)> {
    let rows = matrix.rows();
    let cols = matrix.cols();
    let mut real = clone_data(matrix, "fft1d")?;
    let mut imag = vec![0f64; real.len()];
    fft_axis_inplace(&mut real, &mut imag, rows, cols, axis, false)?;
    matrices_from_components(real, imag, rows, cols)
}

pub fn ifft1d(
    real_matrix: &MatrixBuffer,
    imag_matrix: &MatrixBuffer,
    axis: usize,
) -> CoreResult<(MatrixBuffer, MatrixBuffer)> {
    if real_matrix.rows() != imag_matrix.rows() || real_matrix.cols() != imag_matrix.cols() {
        return Err("ifft1d: real and imag matrices must match shape".into());
    }
    let rows = real_matrix.rows();
    let cols = real_matrix.cols();
    let mut real = clone_data(real_matrix, "ifft1d(real)")?;
    let mut imag = clone_data(imag_matrix, "ifft1d(imag)")?;
    fft_axis_inplace(&mut real, &mut imag, rows, cols, axis, true)?;
    matrices_from_components(real, imag, rows, cols)
}

pub fn fft2d(matrix: &MatrixBuffer) -> CoreResult<(MatrixBuffer, MatrixBuffer)> {
    let rows = matrix.rows();
    let cols = matrix.cols();
    let mut real = clone_data(matrix, "fft2d")?;
    let mut imag = vec![0f64; real.len()];
    fft_axis_inplace(&mut real, &mut imag, rows, cols, 1, false)?;
    fft_axis_inplace(&mut real, &mut imag, rows, cols, 0, false)?;
    matrices_from_components(real, imag, rows, cols)
}

pub fn ifft2d(
    real_matrix: &MatrixBuffer,
    imag_matrix: &MatrixBuffer,
) -> CoreResult<(MatrixBuffer, MatrixBuffer)> {
    if real_matrix.rows() != imag_matrix.rows() || real_matrix.cols() != imag_matrix.cols() {
        return Err("ifft2d: real and imag matrices must match shape".into());
    }
    let rows = real_matrix.rows();
    let cols = real_matrix.cols();
    let mut real = clone_data(real_matrix, "ifft2d(real)")?;
    let mut imag = clone_data(imag_matrix, "ifft2d(imag)")?;
    fft_axis_inplace(&mut real, &mut imag, rows, cols, 1, true)?;
    fft_axis_inplace(&mut real, &mut imag, rows, cols, 0, true)?;
    matrices_from_components(real, imag, rows, cols)
}
