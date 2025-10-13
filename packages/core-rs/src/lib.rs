use cfg_if::cfg_if;
use ndarray::{concatenate, Array1, Array2, ArrayView2, ArrayViewMut2, Axis};
use ndarray_npy::{NpzReader, NpzWriter, ReadNpyExt, WriteNpyExt};
use std::io::Cursor;

cfg_if! {
    if #[cfg(feature = "linalg")] {
        use nalgebra::{DMatrix, SymmetricEigen};
    }
}

pub type CoreResult<T> = Result<T, String>;

pub fn add_inplace(
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
    mut out: ArrayViewMut2<'_, f64>,
) {
    assert_eq!(a.dim(), b.dim());
    assert_eq!(a.dim(), out.dim());
    out.assign(&(&a + &b));
}

pub fn matmul(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Array2<f64> {
    a.dot(&b)
}

pub fn sum_axis0(a: ArrayView2<'_, f64>) -> Array2<f64> {
    a.sum_axis(Axis(0)).insert_axis(Axis(0))
}

pub fn clip(a: ArrayView2<'_, f64>, min: f64, max: f64) -> Array2<f64> {
    a.map(|value| value.clamp(min, max))
}

pub fn where_select(
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

pub fn concat(
    axis: usize,
    matrices: &[ArrayView2<'_, f64>],
) -> CoreResult<Array2<f64>> {
    if matrices.is_empty() {
        return Err("concat expects at least one matrix".into());
    }
    let axis = Axis(axis);
    concatenate(axis, matrices).map_err(|e| e.to_string())
}

pub fn stack_matrices(
    axis: usize,
    matrices: &[ArrayView2<'_, f64>],
) -> CoreResult<Array2<f64>> {
    concat(axis, matrices)
}

#[cfg(feature = "linalg")]
pub fn svd(a: ArrayView2<'_, f64>) -> CoreResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    let matrix = array_to_dmatrix(a);
    let svd = matrix.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| "SVD did not return U matrix".to_string())?;
    let vt = svd
        .v_t
        .ok_or_else(|| "SVD did not return V^T matrix".to_string())?;
    let s = Array1::from_vec(svd.singular_values.iter().copied().collect());
    Ok((dmatrix_to_array(u), s, dmatrix_to_array(vt)))
}

#[cfg(feature = "linalg")]
pub fn qr(a: ArrayView2<'_, f64>) -> CoreResult<(Array2<f64>, Array2<f64>)> {
    let matrix = array_to_dmatrix(a);
    let qr = matrix.qr();
    Ok((dmatrix_to_array(qr.q()), dmatrix_to_array(qr.r())))
}

#[cfg(feature = "linalg")]
pub fn solve(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> CoreResult<Array2<f64>> {
    let left = array_to_dmatrix(a);
    let right = array_to_dmatrix(b);
    left.lu()
        .solve(&right)
        .map(|m| dmatrix_to_array(m))
        .ok_or_else(|| "Solve failed: singular matrix".to_string())
}

#[cfg(feature = "linalg")]
pub fn eigen(a: ArrayView2<'_, f64>) -> CoreResult<(Array1<f64>, Array2<f64>)> {
    let matrix = array_to_dmatrix(a);
    if matrix.nrows() != matrix.ncols() {
        return Err("Eigen decomposition requires a square matrix".into());
    }
    let eigen = SymmetricEigen::new(matrix);
    let values = Array1::from_vec(eigen.eigenvalues.iter().copied().collect());
    Ok((values, dmatrix_to_array(eigen.eigenvectors)))
}

pub fn read_npy_matrix(data: &[u8]) -> CoreResult<Array2<f64>> {
    let mut cursor = Cursor::new(data);
    Array2::<f64>::read_npy(&mut cursor).map_err(|e| e.to_string())
}

pub fn write_npy_matrix(matrix: ArrayView2<'_, f64>) -> CoreResult<Vec<u8>> {
    let mut cursor = Cursor::new(Vec::new());
    matrix
        .to_owned()
        .write_npy(&mut cursor)
        .map_err(|e| e.to_string())?;
    Ok(cursor.into_inner())
}

pub fn read_npz_matrices(data: &[u8]) -> CoreResult<Vec<(String, Array2<f64>)>> {
    let cursor = Cursor::new(data);
    let mut reader = NpzReader::new(cursor).map_err(|e| e.to_string())?;
    let mut results = Vec::new();
    let names = reader.names().map_err(|e| e.to_string())?;
    for name in names {
        let array: Array2<f64> = reader.by_name(&name).map_err(|e| e.to_string())?;
        results.push((name, array));
    }
    Ok(results)
}

pub fn write_npz_matrices(entries: &[(&str, ArrayView2<'_, f64>)]) -> CoreResult<Vec<u8>> {
    let cursor = Cursor::new(Vec::new());
    let mut npz = NpzWriter::new(cursor);
    for (name, array) in entries {
        npz.add_array(*name, array)
            .map_err(|e| format!("Failed to add array {name}: {e}"))?;
    }
    let cursor = npz.finish().map_err(|e| e.to_string())?;
    Ok(cursor.into_inner())
}

#[cfg(feature = "linalg")]
fn array_to_dmatrix(view: ArrayView2<'_, f64>) -> DMatrix<f64> {
    let (rows, cols) = view.dim();
    DMatrix::from_iterator(rows, cols, view.iter().copied())
}

#[cfg(feature = "linalg")]
fn dmatrix_to_array(matrix: DMatrix<f64>) -> Array2<f64> {
    let (rows, cols) = (matrix.nrows(), matrix.ncols());
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            data.push(matrix[(r, c)]);
        }
    }
    Array2::from_shape_vec((rows, cols), data)
        .expect("nalgebra matrix should map to ndarray shape")
}
