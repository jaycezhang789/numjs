use crate::CoreResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBackendKind {
    #[cfg(feature = "gpu-cuda")]
    Cuda,
    #[cfg(feature = "gpu-rocm")]
    Rocm,
}

impl GpuBackendKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            #[cfg(feature = "gpu-cuda")]
            GpuBackendKind::Cuda => "cuda",
            #[cfg(feature = "gpu-rocm")]
            GpuBackendKind::Rocm => "rocm",
            #[allow(unreachable_patterns)]
            _ => "unknown",
        }
    }
}

pub fn active_backend_kind() -> Option<GpuBackendKind> {
    #[cfg(feature = "gpu-cuda")]
    {
        if cuda::is_available() {
            return Some(GpuBackendKind::Cuda);
        }
    }
    #[cfg(feature = "gpu-rocm")]
    {
        if rocm::is_available() {
            return Some(GpuBackendKind::Rocm);
        }
    }
    None
}

pub fn backend_name() -> Option<&'static str> {
    active_backend_kind().map(|kind| kind.as_str())
}

pub fn matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> CoreResult<Vec<f32>> {
    #[cfg(feature = "gpu-cuda")]
    {
        if cuda::is_available() {
            return cuda::matmul_f32(a, b, m, k, n);
        }
    }
    #[cfg(feature = "gpu-rocm")]
    {
        if rocm::is_available() {
            return rocm::matmul_f32(a, b, m, k, n);
        }
    }
    let _ = (a, b, m, k, n);
    Err("GPU matmul backend not available".into())
}

pub fn reduce_sum_f32(values: &[f32]) -> CoreResult<f32> {
    #[cfg(feature = "gpu-cuda")]
    {
        if cuda::is_available() {
            return cuda::reduce_sum_f32(values);
        }
    }
    #[cfg(feature = "gpu-rocm")]
    {
        if rocm::is_available() {
            return rocm::reduce_sum_f32(values);
        }
    }
    let _ = values;
    Err("GPU reduce backend not available".into())
}

#[cfg(feature = "gpu-cuda")]
mod cuda {
    use super::*;
    use cudarc::driver::{CudaContext, CudaFunction, CudaModule, LaunchConfig};
    use cudarc::nvrtc::compile_ptx;
    use once_cell::sync::Lazy;
    use std::sync::Arc;

    /// Kernel launch uses a fixed block size that matches the shared memory
    /// allocation inside `reduce_sum_f32`.
    const THREADS_PER_BLOCK: u32 = 256;

    const CUDA_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void matmul_f32(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ out,
    int m,
    int k,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * n;
    if (idx >= total) { return; }
    int row = idx / n;
    int col = idx % n;
    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
        acc += lhs[row * k + i] * rhs[i * n + col];
    }
    out[idx] = acc;
}

extern "C" __global__ void reduce_sum_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int length
) {
    // blockDim.x is limited to 256 on the host side so this shared array is safe.
    __shared__ float shared[256];
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int grid_size = block_size * gridDim.x;
    int index = blockIdx.x * blockDim.x + tid;
    float total = 0.0f;
    while (index < length) {
        total += input[index];
        index += grid_size;
    }
    shared[tid] = total;
    __syncthreads();
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}
"#;

    struct CudaState {
        ctx: Arc<CudaContext>,
        module: Arc<CudaModule>,
        matmul: Arc<CudaFunction>,
        reduce: Arc<CudaFunction>,
    }

    static CUDA_STATE: Lazy<Result<CudaState, String>> = Lazy::new(|| {
        let ctx = CudaContext::new(0).map_err(|err| err.to_string())?;
        let ptx = compile_ptx(CUDA_KERNEL_SOURCE).map_err(|err| err.to_string())?;
        let module = ctx.load_module(ptx).map_err(|err| err.to_string())?;
        let matmul = module
            .load_function("matmul_f32")
            .map_err(|err| err.to_string())?;
        let reduce = module
            .load_function("reduce_sum_f32")
            .map_err(|err| err.to_string())?;
        Ok(CudaState {
            ctx,
            module,
            matmul: Arc::new(matmul),
            reduce: Arc::new(reduce),
        })
    });

    fn resolve_state() -> Result<&'static CudaState, String> {
        CUDA_STATE.as_ref().map_err(|err| err.clone())
    }

    pub fn is_available() -> bool {
        CUDA_STATE.is_ok()
    }

    fn launch_config_for_len(len: usize) -> Result<LaunchConfig, String> {
        let threads_per_block = THREADS_PER_BLOCK as usize;
        let blocks = if len == 0 {
            0
        } else {
            ((len - 1) / threads_per_block) + 1
        };
        if blocks > u32::MAX as usize {
            return Err("cuda launch: grid dimension overflow".into());
        }
        Ok(LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }

    pub fn matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> CoreResult<Vec<f32>> {
        let state = resolve_state()?;
        let lhs_elems = m
            .checked_mul(k)
            .ok_or_else(|| "cuda matmul: lhs dimension overflow".to_string())?;
        if a.len() != lhs_elems {
            return Err(
                format!("cuda matmul: lhs length mismatch (expected {lhs_elems}, got {})", a.len())
                    .into(),
            );
        }
        let rhs_elems = k
            .checked_mul(n)
            .ok_or_else(|| "cuda matmul: rhs dimension overflow".to_string())?;
        if b.len() != rhs_elems {
            return Err(
                format!("cuda matmul: rhs length mismatch (expected {rhs_elems}, got {})", b.len())
                    .into(),
            );
        }
        let total_elems = m
            .checked_mul(n)
            .ok_or_else(|| "cuda matmul: output dimension overflow".to_string())?;
        if total_elems > i32::MAX as usize {
            return Err("cuda matmul: output element count exceeds i32::MAX".into());
        }
        if total_elems == 0 {
            return Ok(Vec::new());
        }
        let m_i32 = i32::try_from(m)
            .map_err(|_| "cuda matmul: m dimension exceeds i32::MAX".to_string())?;
        let k_i32 = i32::try_from(k)
            .map_err(|_| "cuda matmul: k dimension exceeds i32::MAX".to_string())?;
        let n_i32 = i32::try_from(n)
            .map_err(|_| "cuda matmul: n dimension exceeds i32::MAX".to_string())?;
        let stream = state.ctx.default_stream();
        let a_dev = stream
            .memcpy_stod(a)
            .map_err(|err| format!("cuda memcpy_stod(lhs): {err}"))?;
        let b_dev = stream
            .memcpy_stod(b)
            .map_err(|err| format!("cuda memcpy_stod(rhs): {err}"))?;
        let mut c_dev = stream
            .alloc_zeros::<f32>(total_elems)
            .map_err(|err| format!("cuda alloc_zeros(out): {err}"))?;
        let cfg = launch_config_for_len(total_elems)?;
        let mut launch = stream.launch_builder(&state.matmul);
        launch.arg(&a_dev);
        launch.arg(&b_dev);
        launch.arg(&mut c_dev);
        launch.arg(&m_i32);
        launch.arg(&k_i32);
        launch.arg(&n_i32);
        unsafe { launch.launch(cfg) }.map_err(|err| format!("cuda matmul launch failed: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("cuda stream synchronize: {err}"))?;
        let host = stream
            .memcpy_dtov(&c_dev)
            .map_err(|err| format!("cuda memcpy_dtov(out): {err}"))?;
        Ok(host)
    }

    pub fn reduce_sum_f32(values: &[f32]) -> CoreResult<f32> {
        if values.is_empty() {
            return Ok(0.0);
        }
        if values.len() > i32::MAX as usize {
            return Err("cuda reduce: input length exceeds i32::MAX".into());
        }
        let state = resolve_state()?;
        let stream = state.ctx.default_stream();
        let input_dev = stream
            .memcpy_stod(values)
            .map_err(|err| format!("cuda memcpy_stod(input): {err}"))?;
        let mut output_dev = stream
            .alloc_zeros::<f32>(1)
            .map_err(|err| format!("cuda alloc_zeros(sum): {err}"))?;
        let cfg = launch_config_for_len(values.len())?;
        let mut launch = stream.launch_builder(&state.reduce);
        launch.arg(&input_dev);
        launch.arg(&mut output_dev);
        let len_i32 = i32::try_from(values.len())
            .map_err(|_| "cuda reduce: input length exceeds i32::MAX".to_string())?;
        launch.arg(&len_i32);
        unsafe { launch.launch(cfg) }.map_err(|err| format!("cuda reduce launch failed: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("cuda stream synchronize: {err}"))?;
        let host = stream
            .memcpy_dtov(&output_dev)
            .map_err(|err| format!("cuda memcpy_dtov(sum): {err}"))?;
        Ok(host[0])
    }
}

#[cfg(feature = "gpu-rocm")]
mod rocm {
    use super::*;

    pub fn is_available() -> bool {
        false
    }

    pub fn matmul_f32(
        _a: &[f32],
        _b: &[f32],
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> CoreResult<Vec<f32>> {
        Err("ROCm backend not yet implemented".into())
    }

    pub fn reduce_sum_f32(_values: &[f32]) -> CoreResult<f32> {
        Err("ROCm backend not yet implemented".into())
    }
}
