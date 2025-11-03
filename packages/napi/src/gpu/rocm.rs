#![cfg(feature = "gpu-rocm")]

//! Placeholder ROCm backend wiring.
//!
//! Mirrors the CUDA-side abstraction so that HIP/rocBLAS bindings can slot in
//! later without reshaping the public N-API.

use super::{GpuBackendKind, GpuContext, GpuError, GpuResult};
use num_rs_core::gpu as core_gpu;
use num_rs_core::gpu::{
    GpuBackendKind as CoreGpuBackendKind, MatmulTensorCorePolicy, SumPrecisionPolicy,
};

/// Minimal ROCm context that defers to the shared core bindings.
pub struct RocmContext;

pub fn create_context() -> GpuResult<Box<dyn GpuContext>> {
    if !matches!(
        core_gpu::active_backend_kind(),
        Some(CoreGpuBackendKind::Rocm)
    ) {
        return Err(GpuError::unavailable(
            "ROCm backend not initialised in core runtime",
        ));
    }
    Ok(Box::new(RocmContext))
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
        lhs: &[f32],
        rhs: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> GpuResult<Vec<f32>> {
        if !matches!(
            core_gpu::active_backend_kind(),
            Some(CoreGpuBackendKind::Rocm)
        ) {
            return Err(GpuError::unavailable(
                "ROCm backend not initialised in core runtime",
            ));
        }
        core_gpu::matmul_f32_with_policy(lhs, rhs, m, k, n, MatmulTensorCorePolicy::Accuracy)
            .map_err(|err| GpuError::backend(format!("ROCm matmul failed: {err}")))
    }

    fn reduce_sum_f32(&self, data: &[f32]) -> GpuResult<f32> {
        if !matches!(
            core_gpu::active_backend_kind(),
            Some(CoreGpuBackendKind::Rocm)
        ) {
            return Err(GpuError::unavailable(
                "ROCm backend not initialised in core runtime",
            ));
        }
        core_gpu::reduce_sum_f32_with_policy(data, SumPrecisionPolicy::Default)
            .map_err(|err| GpuError::backend(format!("ROCm reduce_sum failed: {err}")))
    }
}
