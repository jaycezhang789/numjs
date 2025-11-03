//! CUDA backend scaffolding.
//!
//! This module intentionally returns placeholder errors until a full CUDA
//! integration is wired up. The structure is in place so that future work can
//! focus on actual FFI bindings without disturbing the rest of the crate.

use super::{GpuBackendKind, GpuContext, GpuError, GpuResult};

#[allow(dead_code)]
pub struct CudaContext;

pub fn create_context() -> GpuResult<Box<dyn GpuContext>> {
    Err(GpuError::unavailable(
        "CUDA backend not yet implemented; enable `gpu-cuda` and provide an implementation",
    ))
}

impl GpuContext for CudaContext {
    fn backend(&self) -> GpuBackendKind {
        GpuBackendKind::Cuda
    }

    fn name(&self) -> &'static str {
        "cuda"
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
            "CUDA matmul is not yet implemented in this build",
        ))
    }

    fn reduce_sum_f32(&self, _data: &[f32]) -> GpuResult<f32> {
        Err(GpuError::unsupported(
            "CUDA reduce_sum is not yet implemented in this build",
        ))
    }
}
