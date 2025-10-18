use crate::buffer::MatrixBuffer;
use crate::dtype::DType;
use crate::error;
use crate::{add as core_add, matmul as core_matmul, transpose as core_transpose, CoreResult};

#[cfg(feature = "sparse-native")]
use ndarray::Array2;
#[cfg(feature = "sparse-native")]
use sprs::CsMat;

#[derive(Clone, Copy)]
pub enum CsrValues<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
}

pub struct CsrMatrixView<'a> {
    rows: usize,
    cols: usize,
    dtype: DType,
    row_ptr: &'a [u32],
    col_idx: &'a [u32],
    values: CsrValues<'a>,
}

impl<'a> CsrMatrixView<'a> {
    pub fn new_f32(
        rows: usize,
        cols: usize,
        row_ptr: &'a [u32],
        col_idx: &'a [u32],
        values: &'a [f32],
    ) -> CoreResult<Self> {
        let view = CsrMatrixView {
            rows,
            cols,
            dtype: DType::Float32,
            row_ptr,
            col_idx,
            values: CsrValues::F32(values),
        };
        view.validate()?;
        Ok(view)
    }

    pub fn new_f64(
        rows: usize,
        cols: usize,
        row_ptr: &'a [u32],
        col_idx: &'a [u32],
        values: &'a [f64],
    ) -> CoreResult<Self> {
        let view = CsrMatrixView {
            rows,
            cols,
            dtype: DType::Float64,
            row_ptr,
            col_idx,
            values: CsrValues::F64(values),
        };
        view.validate()?;
        Ok(view)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn row_ptr(&self) -> &'a [u32] {
        self.row_ptr
    }

    pub fn col_idx(&self) -> &'a [u32] {
        self.col_idx
    }

    pub fn values_f32(&self) -> Option<&'a [f32]> {
        match self.values {
            CsrValues::F32(values) => Some(values),
            _ => None,
        }
    }

    pub fn values_f64(&self) -> Option<&'a [f64]> {
        match self.values {
            CsrValues::F64(values) => Some(values),
            _ => None,
        }
    }

    pub fn to_dense(&self) -> CoreResult<MatrixBuffer> {
        let nnz = self.nnz()?;
        let expected = self.row_ptr.last().copied().unwrap_or(0) as usize;
        if expected != nnz {
            return Err(error::shape_mismatch("CSR rowPtr does not match nnz"));
        }
        let total = self.rows.saturating_mul(self.cols);
        match self.values {
            CsrValues::F32(values) => {
                if values.len() != nnz {
                    return Err(error::shape_mismatch(
                        "CSR values length does not match nnz",
                    ));
                }
                let mut dense = vec![0f32; total];
                for row in 0..self.rows {
                    let start = self.row_ptr[row] as usize;
                    let end = self.row_ptr[row + 1] as usize;
                    if end > values.len() || end > self.col_idx.len() {
                        return Err(error::shape_mismatch(
                            "CSR rowPtr points past colIdx/values buffers",
                        ));
                    }
                    for idx in start..end {
                        let col = self.col_idx[idx] as usize;
                        if col >= self.cols {
                            return Err(error::shape_mismatch("CSR column index out of bounds"));
                        }
                        dense[row * self.cols + col] = values[idx];
                    }
                }
                MatrixBuffer::from_vec(dense, self.rows, self.cols).map_err(Into::into)
            }
            CsrValues::F64(values) => {
                if values.len() != nnz {
                    return Err(error::shape_mismatch(
                        "CSR values length does not match nnz",
                    ));
                }
                let mut dense = vec![0f64; total];
                for row in 0..self.rows {
                    let start = self.row_ptr[row] as usize;
                    let end = self.row_ptr[row + 1] as usize;
                    if end > values.len() || end > self.col_idx.len() {
                        return Err(error::shape_mismatch(
                            "CSR rowPtr points past colIdx/values buffers",
                        ));
                    }
                    for idx in start..end {
                        let col = self.col_idx[idx] as usize;
                        if col >= self.cols {
                            return Err(error::shape_mismatch("CSR column index out of bounds"));
                        }
                        dense[row * self.cols + col] = values[idx];
                    }
                }
                MatrixBuffer::from_vec(dense, self.rows, self.cols).map_err(Into::into)
            }
        }
    }

    fn validate(&self) -> CoreResult<()> {
        if self.row_ptr.len() != self.rows + 1 {
            return Err(error::shape_mismatch(
                "CSR rowPtr length must equal rows + 1",
            ));
        }
        if let Some(last) = self.row_ptr.last() {
            let nnz = *last as usize;
            if self.col_idx.len() != nnz {
                return Err(error::shape_mismatch("CSR colIdx length must equal nnz"));
            }
        }
        let mut prev = 0u32;
        for &value in self.row_ptr {
            if value < prev {
                return Err(error::shape_mismatch("CSR rowPtr must be non-decreasing"));
            }
            prev = value;
        }
        for &col in self.col_idx {
            if col as usize >= self.cols {
                return Err(error::shape_mismatch(
                    "CSR colIdx contains value out of bounds",
                ));
            }
        }
        Ok(())
    }

    fn nnz(&self) -> CoreResult<usize> {
        self.row_ptr
            .last()
            .copied()
            .map(|v| v as usize)
            .ok_or_else(|| error::shape_mismatch("CSR rowPtr missing last entry"))
    }
}

pub fn sparse_matmul(csr: &CsrMatrixView<'_>, dense: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    #[cfg(feature = "sparse-native")]
    {
        match sparse_matmul_native(csr, dense) {
            Ok(result) => return Ok(result),
            Err(err) => {
                eprintln!(
                    "[num_rs_core][sparse] native sparse_matmul failed: {err}. Falling back to dense implementation."
                );
            }
        }
    }
    sparse_matmul_fallback(csr, dense)
}

pub fn sparse_add(csr: &CsrMatrixView<'_>, dense: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    #[cfg(feature = "sparse-native")]
    {
        match sparse_add_native(csr, dense) {
            Ok(result) => return Ok(result),
            Err(err) => {
                eprintln!(
                    "[num_rs_core][sparse] native sparse_add failed: {err}. Falling back to dense implementation."
                );
            }
        }
    }
    sparse_add_fallback(csr, dense)
}

pub fn sparse_transpose(csr: &CsrMatrixView<'_>) -> CoreResult<MatrixBuffer> {
    #[cfg(feature = "sparse-native")]
    {
        match sparse_transpose_native(csr) {
            Ok(result) => return Ok(result),
            Err(err) => {
                eprintln!(
                    "[num_rs_core][sparse] native sparse_transpose failed: {err}. Falling back to dense implementation."
                );
            }
        }
    }
    sparse_transpose_fallback(csr)
}

fn sparse_matmul_fallback(
    csr: &CsrMatrixView<'_>,
    dense: &MatrixBuffer,
) -> CoreResult<MatrixBuffer> {
    let lhs = csr.to_dense()?;
    core_matmul(&lhs, dense)
}

fn sparse_add_fallback(csr: &CsrMatrixView<'_>, dense: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    let lhs = csr.to_dense()?;
    core_add(&lhs, dense)
}

fn sparse_transpose_fallback(csr: &CsrMatrixView<'_>) -> CoreResult<MatrixBuffer> {
    let dense = csr.to_dense()?;
    core_transpose(&dense)
}

#[cfg(feature = "sparse-native")]
fn csr_to_sprs_f64(csr: &CsrMatrixView<'_>) -> CoreResult<CsMat<f64>> {
    let indptr: Vec<usize> = csr.row_ptr().iter().map(|&v| v as usize).collect();
    let indices: Vec<usize> = csr.col_idx().iter().map(|&v| v as usize).collect();
    let values: Vec<f64> = match csr.dtype() {
        DType::Float32 => csr
            .values_f32()
            .ok_or_else(|| error::numeric_issue("expected float32 values"))?
            .iter()
            .map(|&v| v as f64)
            .collect(),
        DType::Float64 => csr
            .values_f64()
            .ok_or_else(|| error::numeric_issue("expected float64 values"))?
            .to_vec(),
        other => {
            return Err(error::numeric_issue(format!(
                "sparse native operations currently support only float32/float64 (got {other})"
            )))
        }
    };
    Ok(CsMat::new(
        (csr.rows(), csr.cols()),
        indptr,
        indices,
        values,
    ))
}

#[cfg(feature = "sparse-native")]
fn sparse_matmul_native(csr: &CsrMatrixView<'_>, dense: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    let csr = csr_to_sprs_f64(csr)?;
    let rhs_vec = dense
        .to_f64_vec()
        .map_err(|err| error::numeric_issue(format!("dense conversion failed: {err}")))?;
    let rhs = Array2::from_shape_vec((dense.rows(), dense.cols()), rhs_vec)
        .map_err(|err| error::numeric_issue(format!("reshape failure: {err}")))?;
    let result = &csr * &rhs;
    let data = result.into_raw_vec();
    MatrixBuffer::from_vec(data, csr.rows(), rhs.ncols()).map_err(Into::into)
}

#[cfg(feature = "sparse-native")]
fn sparse_add_native(csr: &CsrMatrixView<'_>, dense: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    if csr.rows() != dense.rows() || csr.cols() != dense.cols() {
        return Err(error::shape_mismatch("sparse_add: shape mismatch"));
    }
    let csr = csr_to_sprs_f64(csr)?;
    let mut dense_vec = dense
        .to_f64_vec()
        .map_err(|err| error::numeric_issue(format!("dense conversion failed: {err}")))?;
    let mut dense_array = Array2::from_shape_vec((dense.rows(), dense.cols()), dense_vec)
        .map_err(|err| error::numeric_issue(format!("reshape failure: {err}")))?;
    let csr_dense = Array2::from_shape_vec((csr.rows(), csr.cols()), csr.to_dense())
        .map_err(|err| error::numeric_issue(format!("csr to dense reshape failure: {err}")))?;
    dense_array += &csr_dense;
    let data = dense_array.into_raw_vec();
    MatrixBuffer::from_vec(data, csr.rows(), csr.cols()).map_err(Into::into)
}

#[cfg(feature = "sparse-native")]
fn sparse_transpose_native(csr: &CsrMatrixView<'_>) -> CoreResult<MatrixBuffer> {
    let csr = csr_to_sprs_f64(csr)?;
    let transposed = csr.transpose_view().to_owned();
    let data = transposed.to_dense();
    MatrixBuffer::from_vec(data, transposed.rows(), transposed.cols()).map_err(Into::into)
}
