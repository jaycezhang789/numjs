#![cfg(feature = "gpu-cuda")]

//! Placeholder CUDA backend wiring.
//!
//! Provides the minimum scheduling and resource-management skeleton so that
//! future cuBLAS/cuSOLVER integration can reuse the same N-API surface.

use super::{GpuBackendKind, GpuContext, GpuError, GpuResult};
use cust::context::Context;
use cust::device::Device;
use cust::memory::{DeviceBuffer, DeviceCopy};
use cust::stream::{Stream, StreamFlags};
use cust::CudaFlags;
use std::fmt;

pub struct CudaContext {
    _context: Context,
    stream: Stream,
    device: Device,
}

pub fn create_context() -> GpuResult<Box<dyn GpuContext>> {
    cust::init(CudaFlags::empty()).map_err(map_cuda_error)?;
    let device = Device::get_device(0).map_err(map_cuda_error)?;
    let context = Context::new(device).map_err(map_cuda_error)?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(map_cuda_error)?;

    Ok(Box::new(CudaContext {
        _context: context,
        stream,
        device,
    }))
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
            "CUDA matmul is not implemented yet; future work will delegate to cuBLAS.",
        ))
    }

    fn reduce_sum_f32(&self, _data: &[f32]) -> GpuResult<f32> {
        Err(GpuError::unsupported(
            "CUDA reduce_sum is not implemented yet; a custom kernel will align with CPU results.",
        ))
    }

    fn synchronize(&self) -> GpuResult<()> {
        self.stream.synchronize().map_err(map_cuda_error)
    }
}

impl CudaContext {
    #[allow(dead_code)]
    pub fn device(&self) -> Device {
        self.device
    }

    #[allow(dead_code)]
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    #[allow(dead_code)]
    pub fn alloc_device_buffer<T: DeviceCopy>(&self, len: usize) -> GpuResult<DeviceBuffer<T>> {
        unsafe { DeviceBuffer::uninitialized(len) }.map_err(map_cuda_error)
    }
}

fn map_cuda_error<E>(err: E) -> GpuError
where
    E: fmt::Display,
{
    GpuError::backend(format!("CUDA error: {err}"))
}
