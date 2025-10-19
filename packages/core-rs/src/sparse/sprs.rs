use super::{CsrMatrixView, SparseBackend};
use crate::buffer::MatrixBuffer;
use crate::dtype::DType;
use crate::error;
use crate::CoreResult;
use ndarray::Array2;
use sprs::CsMat;

pub(crate) struct SprsBackend;

impl SparseBackend for SprsBackend {
    fn csrmv(&self, csr: &CsrMatrixView<'_>, dense: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
        if dense.cols() != 1 {
            return Err(error::shape_mismatch(
                "csrmv expects a column vector (dense.ncols == 1)",
            ));
        }
        self.csrgemm(csr, dense)
    }

    fn csrgemm(&self, csr: &CsrMatrixView<'_>, dense: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
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

    fn csradd(&self, csr: &CsrMatrixView<'_>, dense: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
        if csr.rows() != dense.rows() || csr.cols() != dense.cols() {
            return Err(error::shape_mismatch("sparse_add: shape mismatch"));
        }
        let csr = csr_to_sprs_f64(csr)?;
        let dense_vec = dense
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

    fn transpose(&self, csr: &CsrMatrixView<'_>) -> CoreResult<MatrixBuffer> {
        let csr = csr_to_sprs_f64(csr)?;
        let transposed = csr.transpose_view().to_owned();
        let data = transposed.to_dense();
        MatrixBuffer::from_vec(data, transposed.rows(), transposed.cols()).map_err(Into::into)
    }
}

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
