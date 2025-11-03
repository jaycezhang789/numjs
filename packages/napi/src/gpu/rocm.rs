//! ROCm backend scaffolding.
//!
//! Like the CUDA module, this file keeps the trait surface ready while
//! returning placeholder errors until a full HIP/rocBLAS integration lands.

use super::{GpuBackendKind, GpuContext, GpuError, GpuResult};

#[allow(dead_code)]
pub struct RocmContext;

pub fn create_context() -> GpuResult<Box<dyn GpuContext>> {
    Err(GpuError::unavailable(
        "ROCm backend not yet implemented; enable `gpu-rocm` and provide an implementation",
    ))
}

impl GpuContext for RocmContext {
    fn backend(&self) -> GpuBackendKind {
        GpuBackendKind::Rocm
    }

    fn name(&self) -> &'static str {
        "rocm"
    }

    fn matmul_f32(
        &self,
        _lhs: &[f32],
        _rhs: &[f32],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> GpuResult<Vec<f32>> {
        Err(GpuError::unsupported(
            "ROCm matmul is not yet implemented in this build",
        ))
    }

    fn reduce_sum_f32(&self, _data: &[f32]) -> GpuResult<f32> {
        Err(GpuError::unsupported(
            "ROCm reduce_sum is not yet implemented in this build",
        ))
    }
}
