mod ffi;

use super::{CsrMatrixView, SparseBackend};
use crate::buffer::MatrixBuffer;
use crate::dtype::DType;
use crate::{error, CoreResult};

pub(crate) struct SuiteSparseBackend;

struct PreparedCsr<'a> {
    rows: usize,
    cols: usize,
    row_ptr: &'a [u32],
    col_idx: &'a [u32],
    values: Vec<f64>,
}

impl<'a> PreparedCsr<'a> {
    fn from_view(csr: &'a CsrMatrixView<'a>) -> CoreResult<Self> {
        csr.nnz()?;
        let values = match csr.dtype() {
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
                    "SuiteSparse backend currently supports float32/float64 only (got {:?})",
                    other
                )))
            }
        };

        Ok(PreparedCsr {
            rows: csr.rows(),
            cols: csr.cols(),
            row_ptr: csr.row_ptr(),
            col_idx: csr.col_idx(),
            values,
        })
    }
}

fn dense_to_f64(dense: &MatrixBuffer) -> CoreResult<Vec<f64>> {
    Ok(dense.to_f64_vec())
}

fn suitesparse_status(op: &str, status: i32) -> CoreResult<()> {
    match status {
        ffi::STATUS_SUCCESS => Ok(()),
        ffi::STATUS_BAD_SHAPE | ffi::STATUS_OUT_OF_BOUNDS => Err(error::shape_mismatch(format!(
            "SuiteSparse {op} reported invalid shape or indices"
        ))),
        ffi::STATUS_ALLOC_FAILED => Err(error::numeric_issue(format!(
            "SuiteSparse {op} failed to allocate internal buffers"
        ))),
        ffi::STATUS_INTERNAL => Err(error::numeric_issue(format!(
            "SuiteSparse {op} reported an internal failure"
        ))),
        other => Err(error::numeric_issue(format!(
            "SuiteSparse {op} failed with status {other}"
        ))),
    }
}

impl SparseBackend for SuiteSparseBackend {
    fn csrmv(&self, csr: &CsrMatrixView<'_>, dense: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
        if dense.cols() != 1 {
            return Err(error::shape_mismatch(
                "csrmv expects dense.cols() == 1 for column vector input",
            ));
        }
        if dense.rows() != csr.cols() {
            return Err(error::shape_mismatch(
                "csrmv expects dense.rows() to match CSR columns",
            ));
        }
        let prepared = PreparedCsr::from_view(csr)?;
        let rhs = dense_to_f64(dense)?;
        let mut out = vec![0f64; prepared.rows];
        let status = unsafe {
            ffi::numrs_suitesparse_csrmv(
                prepared.rows,
                prepared.cols,
                prepared.row_ptr.as_ptr(),
                prepared.col_idx.as_ptr(),
                prepared.values.as_ptr(),
                rhs.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        suitesparse_status("csrmv", status)?;
        MatrixBuffer::from_vec(out, prepared.rows, 1).map_err(Into::into)
    }

    fn csrgemm(&self, csr: &CsrMatrixView<'_>, dense: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
        if dense.rows() != csr.cols() {
            return Err(error::shape_mismatch(
                "csrgemm expects dense.rows() to match CSR columns",
            ));
        }
        let prepared = PreparedCsr::from_view(csr)?;
        let rhs_cols = dense.cols();
        let rhs = dense_to_f64(dense)?;
        let mut out = vec![0f64; prepared.rows.saturating_mul(rhs_cols)];
        let status = unsafe {
            ffi::numrs_suitesparse_csrgemm(
                prepared.rows,
                prepared.cols,
                rhs_cols,
                prepared.row_ptr.as_ptr(),
                prepared.col_idx.as_ptr(),
                prepared.values.as_ptr(),
                rhs.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        suitesparse_status("csrgemm", status)?;
        MatrixBuffer::from_vec(out, prepared.rows, rhs_cols).map_err(Into::into)
    }

    fn csradd(&self, csr: &CsrMatrixView<'_>, dense: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
        if dense.rows() != csr.rows() || dense.cols() != csr.cols() {
            return Err(error::shape_mismatch(
                "csradd expects dense shape to match CSR shape",
            ));
        }
        let prepared = PreparedCsr::from_view(csr)?;
        let dense_vec = dense_to_f64(dense)?;
        let mut out = vec![0f64; prepared.rows.saturating_mul(prepared.cols)];
        let status = unsafe {
            ffi::numrs_suitesparse_csradd(
                prepared.rows,
                prepared.cols,
                prepared.row_ptr.as_ptr(),
                prepared.col_idx.as_ptr(),
                prepared.values.as_ptr(),
                dense_vec.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        suitesparse_status("csradd", status)?;
        MatrixBuffer::from_vec(out, prepared.rows, prepared.cols).map_err(Into::into)
    }

    fn transpose(&self, csr: &CsrMatrixView<'_>) -> CoreResult<MatrixBuffer> {
        let prepared = PreparedCsr::from_view(csr)?;
        let mut out = vec![0f64; prepared.rows.saturating_mul(prepared.cols)];
        let status = unsafe {
            ffi::numrs_suitesparse_csrtranspose(
                prepared.rows,
                prepared.cols,
                prepared.row_ptr.as_ptr(),
                prepared.col_idx.as_ptr(),
                prepared.values.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        suitesparse_status("csrtranspose", status)?;
        MatrixBuffer::from_vec(out, prepared.cols, prepared.rows).map_err(Into::into)
    }
}
