use crate::buffer::MatrixBuffer;
use crate::dtype::DType;
use crate::error;
use crate::{add as core_add, matmul as core_matmul, transpose as core_transpose, CoreResult};

#[cfg(feature = "sparse-native")]
mod sprs;
#[cfg(feature = "sparse-native")]
use sprs::SprsBackend;

#[cfg(feature = "sparse-suitesparse")]
mod suitesparse;
#[cfg(feature = "sparse-suitesparse")]
use suitesparse::SuiteSparseBackend;

/// Backend contract for sparse CSR operations. Concrete implementations can use
/// software fallbacks (dense) or accelerated bindings such as SuiteSparse.
pub trait SparseBackend {
    fn csrmv(&self, csr: &CsrMatrixView<'_>, dense: &MatrixBuffer) -> CoreResult<MatrixBuffer>;
    fn csrgemm(&self, csr: &CsrMatrixView<'_>, dense: &MatrixBuffer) -> CoreResult<MatrixBuffer>;
    fn csradd(&self, csr: &CsrMatrixView<'_>, dense: &MatrixBuffer) -> CoreResult<MatrixBuffer>;
    fn transpose(&self, csr: &CsrMatrixView<'_>) -> CoreResult<MatrixBuffer>;
}

struct FallbackBackend;

impl SparseBackend for FallbackBackend {
    fn csrmv(&self, csr: &CsrMatrixView<'_>, dense: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
        if dense.cols() != 1 {
            return Err(error::shape_mismatch(
                "csrmv expects a column vector (dense.ncols == 1)",
            ));
        }
        self.csrgemm(csr, dense)
    }

    fn csrgemm(&self, csr: &CsrMatrixView<'_>, dense: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
        let lhs = csr.to_dense()?;
        core_matmul(&lhs, dense)
    }

    fn csradd(&self, csr: &CsrMatrixView<'_>, dense: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
        let lhs = csr.to_dense()?;
        core_add(&lhs, dense)
    }

    fn transpose(&self, csr: &CsrMatrixView<'_>) -> CoreResult<MatrixBuffer> {
        let dense = csr.to_dense()?;
        core_transpose(&dense)
    }
}

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

    pub fn nnz(&self) -> CoreResult<usize> {
        self.row_ptr
            .last()
            .copied()
            .map(|v| v as usize)
            .ok_or_else(|| error::shape_mismatch("CSR rowPtr missing last entry"))
    }
}

pub fn sparse_matmul(csr: &CsrMatrixView<'_>, dense: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    #[cfg(feature = "sparse-suitesparse")]
    {
        let backend = SuiteSparseBackend;
        match backend.csrgemm(csr, dense) {
            Ok(result) => return Ok(result),
            Err(err) => {
                eprintln!(
                    "[num_rs_core][sparse] suitesparse sparse_matmul failed: {err}. Falling back to alternate implementation."
                );
            }
        }
    }
    #[cfg(feature = "sparse-native")]
    {
        let backend = SprsBackend;
        match backend.csrgemm(csr, dense) {
            Ok(result) => return Ok(result),
            Err(err) => {
                eprintln!(
                    "[num_rs_core][sparse] native sparse_matmul failed: {err}. Falling back to dense implementation."
                );
            }
        }
    }
    FallbackBackend.csrgemm(csr, dense)
}

pub fn sparse_add(csr: &CsrMatrixView<'_>, dense: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    #[cfg(feature = "sparse-suitesparse")]
    {
        let backend = SuiteSparseBackend;
        match backend.csradd(csr, dense) {
            Ok(result) => return Ok(result),
            Err(err) => {
                eprintln!(
                    "[num_rs_core][sparse] suitesparse sparse_add failed: {err}. Falling back to alternate implementation."
                );
            }
        }
    }
    #[cfg(feature = "sparse-native")]
    {
        let backend = SprsBackend;
        match backend.csradd(csr, dense) {
            Ok(result) => return Ok(result),
            Err(err) => {
                eprintln!(
                    "[num_rs_core][sparse] native sparse_add failed: {err}. Falling back to dense implementation."
                );
            }
        }
    }
    FallbackBackend.csradd(csr, dense)
}

pub fn sparse_transpose(csr: &CsrMatrixView<'_>) -> CoreResult<MatrixBuffer> {
    #[cfg(feature = "sparse-suitesparse")]
    {
        let backend = SuiteSparseBackend;
        match backend.transpose(csr) {
            Ok(result) => return Ok(result),
            Err(err) => {
                eprintln!(
                    "[num_rs_core][sparse] suitesparse sparse_transpose failed: {err}. Falling back to alternate implementation."
                );
            }
        }
    }
    #[cfg(feature = "sparse-native")]
    {
        let backend = SprsBackend;
        match backend.transpose(csr) {
            Ok(result) => return Ok(result),
            Err(err) => {
                eprintln!(
                    "[num_rs_core][sparse] native sparse_transpose failed: {err}. Falling back to dense implementation."
                );
            }
        }
    }
    FallbackBackend.transpose(csr)
}

pub fn sparse_matvec(csr: &CsrMatrixView<'_>, vector: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    #[cfg(feature = "sparse-suitesparse")]
    {
        let backend = SuiteSparseBackend;
        match backend.csrmv(csr, vector) {
            Ok(result) => return Ok(result),
            Err(err) => {
                eprintln!(
                    "[num_rs_core][sparse] suitesparse sparse_matvec failed: {err}. Falling back to alternate implementation."
                );
            }
        }
    }
    #[cfg(feature = "sparse-native")]
    {
        let backend = SprsBackend;
        match backend.csrmv(csr, vector) {
            Ok(result) => return Ok(result),
            Err(err) => {
                eprintln!(
                    "[num_rs_core][sparse] native sparse_matvec failed: {err}. Falling back to dense implementation."
                );
            }
        }
    }
    FallbackBackend.csrmv(csr, vector)
}
