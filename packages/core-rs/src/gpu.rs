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

    pub fn matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> CoreResult<Vec<f32>> {
        let state = resolve_state()?;
        if a.len() != m * k {
            return Err("cuda matmul: lhs length mismatch".into());
        }
        if b.len() != k * n {
            return Err("cuda matmul: rhs length mismatch".into());
        }
        let stream = state.ctx.default_stream();
        let a_dev = stream
            .memcpy_stod(a)
            .map_err(|err| format!("cuda memcpy_stod(lhs): {err}"))?;
        let b_dev = stream
            .memcpy_stod(b)
            .map_err(|err| format!("cuda memcpy_stod(rhs): {err}"))?;
        let mut c_dev = stream
            .alloc_zeros::<f32>(m * n)
            .map_err(|err| format!("cuda alloc_zeros(out): {err}"))?;
        let cfg = LaunchConfig::for_num_elems((m * n) as u32);
        let mut launch = stream.launch_builder(&state.matmul);
        launch.arg(&a_dev);
        launch.arg(&b_dev);
        launch.arg(&mut c_dev);
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let n_i32 = n as i32;
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
        let state = resolve_state()?;
        let stream = state.ctx.default_stream();
        let input_dev = stream
            .memcpy_stod(values)
            .map_err(|err| format!("cuda memcpy_stod(input): {err}"))?;
        let mut output_dev = stream
            .alloc_zeros::<f32>(1)
            .map_err(|err| format!("cuda alloc_zeros(sum): {err}"))?;
        let cfg = LaunchConfig::for_num_elems(values.len() as u32);
        let mut launch = stream.launch_builder(&state.reduce);
        launch.arg(&input_dev);
        launch.arg(&mut output_dev);
        let len_i32 = values.len() as i32;
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
