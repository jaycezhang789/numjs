//! GPU backend scaffolding for the N-API bridge.
//!
//! This module defines shared abstractions (`GpuContext`, `GpuResult`, etc.)
//! and exposes feature-gated constructors for CUDA / ROCm implementations.
//! Real backends can fill in the trait methods while the rest of the crate
//! stays agnostic about the underlying runtime.

use std::fmt;

#[cfg(feature = "gpu-cuda")]
mod cuda;
#[cfg(feature = "gpu-rocm")]
mod rocm;

/// Result type used across GPU helpers.
pub type GpuResult<T> = Result<T, GpuError>;

/// Error metadata describing why a GPU operation failed.
#[derive(Debug, Clone)]
pub struct GpuError {
    kind: GpuErrorKind,
    message: String,
}

impl GpuError {
    pub fn new(kind: GpuErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new(GpuErrorKind::Unavailable, message)
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(GpuErrorKind::Unsupported, message)
    }

    #[allow(dead_code)]
    pub fn backend(message: impl Into<String>) -> Self {
        Self::new(GpuErrorKind::Backend, message)
    }
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for GpuError {}

/// High-level error classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuErrorKind {
    Unavailable,
    Unsupported,
    InvalidArgument,
    Backend,
}

/// Enumerates the GPU backends we plan to support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackendKind {
    Cpu,
    Cuda,
    Rocm,
}

/// Shared interface that concrete GPU backends must implement.
pub trait GpuContext: Send + Sync {
    /// Which backend produced this context.
    fn backend(&self) -> GpuBackendKind;

    /// Human readable backend name.
    fn name(&self) -> &'static str;

    /// Matrix multiplication specialised for f32.
    fn matmul_f32(
        &self,
        lhs: &[f32],
        rhs: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> GpuResult<Vec<f32>>;

    /// Sum-reduction specialised for f32.
    fn reduce_sum_f32(&self, data: &[f32]) -> GpuResult<f32>;

    /// Synchronise pending work. Default implementation is a no-op.
    fn synchronize(&self) -> GpuResult<()> {
        Ok(())
    }
}

/// Placeholder context used when no GPU backend is linked.
#[derive(Debug, Default)]
pub struct NoGpuContext;

impl GpuContext for NoGpuContext {
    fn backend(&self) -> GpuBackendKind {
        GpuBackendKind::Cpu
    }

    fn name(&self) -> &'static str {
        "cpu"
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
            "GPU backend not initialised; CPU path should be used instead",
        ))
    }

    fn reduce_sum_f32(&self, _data: &[f32]) -> GpuResult<f32> {
        Err(GpuError::unsupported(
            "GPU backend not initialised; CPU path should be used instead",
        ))
    }
}

/// Attempt to construct a CUDA context if the feature is enabled.
#[cfg(feature = "gpu-cuda")]
fn try_cuda_context() -> GpuResult<Box<dyn GpuContext>> {
    cuda::create_context()
}

#[cfg(not(feature = "gpu-cuda"))]
fn try_cuda_context() -> GpuResult<Box<dyn GpuContext>> {
    Err(GpuError::unavailable(
        "CUDA backend not enabled; rebuild with the `gpu-cuda` feature",
    ))
}

/// Attempt to construct a ROCm context if the feature is enabled.
#[cfg(feature = "gpu-rocm")]
fn try_rocm_context() -> GpuResult<Box<dyn GpuContext>> {
    rocm::create_context()
}

#[cfg(not(feature = "gpu-rocm"))]
fn try_rocm_context() -> GpuResult<Box<dyn GpuContext>> {
    Err(GpuError::unavailable(
        "ROCm backend not enabled; rebuild with the `gpu-rocm` feature",
    ))
}

/// Return the best available GPU backend kind, if any.
#[allow(dead_code)]
pub fn active_backend_kind() -> Option<GpuBackendKind> {
    if try_cuda_context().is_ok() {
        return Some(GpuBackendKind::Cuda);
    }
    if try_rocm_context().is_ok() {
        return Some(GpuBackendKind::Rocm);
    }
    None
}

/// Convenience helper mirroring the existing CPU API.
#[allow(dead_code)]
pub fn create_best_context(prefer: Option<GpuBackendKind>) -> GpuResult<Box<dyn GpuContext>> {
    if let Some(GpuBackendKind::Cuda) = prefer {
        return try_cuda_context();
    }
    if let Some(GpuBackendKind::Rocm) = prefer {
        return try_rocm_context();
    }

    try_cuda_context().or_else(|_| try_rocm_context())
}

/// Whether any GPU backend is considered available at runtime.
#[allow(dead_code)]
pub fn gpu_available() -> bool {
    active_backend_kind().is_some()
}

/// Convenience utility returning the human-readable backend name.
#[allow(dead_code)]
pub fn backend_name() -> Option<&'static str> {
    active_backend_kind().map(|kind| match kind {
        GpuBackendKind::Cpu => "cpu",
        GpuBackendKind::Cuda => "cuda",
        GpuBackendKind::Rocm => "rocm",
    })
}
