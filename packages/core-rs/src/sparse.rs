use crate::buffer::MatrixBuffer;
use crate::dtype::DType;
use crate::error;
use crate::{add as core_add, matmul as core_matmul, transpose as core_transpose, CoreResult};

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
    let lhs = csr.to_dense()?;
    core_matmul(&lhs, dense)
}

pub fn sparse_add(csr: &CsrMatrixView<'_>, dense: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    let lhs = csr.to_dense()?;
    core_add(&lhs, dense)
}

pub fn sparse_transpose(csr: &CsrMatrixView<'_>) -> CoreResult<MatrixBuffer> {
    let dense = csr.to_dense()?;
    core_transpose(&dense)
}
