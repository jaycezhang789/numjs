#![cfg(feature = "gpu-rocm")]

//! Placeholder ROCm backend wiring.
//!
//! Mirrors the CUDA-side abstraction so that HIP/rocBLAS bindings can slot in
//! later without reshaping the public N-API.

use super::{GpuBackendKind, GpuContext, GpuError, GpuResult};

/// Minimal stub used to satisfy the shared trait while the implementation lands.
pub struct RocmContext;

pub fn create_context() -> GpuResult<Box<dyn GpuContext>> {
    Err(GpuError::unavailable(
        "ROCm backend is not implemented yet; enable `gpu-rocm` once HIP/rocBLAS bindings exist.",
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
            "ROCm matmul is not implemented yet; future HIP/rocBLAS kernels will mirror CPU results.",
        ))
    }

    fn reduce_sum_f32(&self, _data: &[f32]) -> GpuResult<f32> {
        Err(GpuError::unsupported(
            "ROCm reduce_sum is not implemented yet; dedicated kernels will mirror CPU results.",
        ))
    }
}
