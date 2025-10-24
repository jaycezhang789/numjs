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
    use cudarc::cublas::sys::cublasOperation_t;
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
    use cudarc::driver::{CudaDevice, CudaFunction, LaunchConfig};
    use cudarc::nvrtc::{compile_ptx_with_opts, CompileError, CompileOptions};
    use once_cell::sync::Lazy;
    use std::collections::BTreeSet;
    use std::os::raw::c_int;
    use std::path::PathBuf;
    use std::sync::Arc;

    const MODULE_NAME: &str = "numjs_cuda_core";
    /// Must stay in sync with the CUB block size used in `reduce_sum_f32`.
    const THREADS_PER_BLOCK: u32 = 256;

    const CUDA_KERNEL_SOURCE: &str = r#"
#include <cub/block/block_reduce.cuh>

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
    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage storage;
    float thread_sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < length) {
        thread_sum += input[idx];
        idx += stride;
    }
    float block_sum = BlockReduce(storage).Sum(thread_sum);
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum);
    }
}
"#;

    struct CudaState {
        device: Arc<CudaDevice>,
        blas: Arc<CudaBlas>,
        reduce: CudaFunction,
        matmul_fallback: Option<CudaFunction>,
    }

    static CUDA_STATE: Lazy<Result<CudaState, String>> = Lazy::new(|| {
        let device = CudaDevice::new(0).map_err(|err| err.to_string())?;
        let blas = CudaBlas::new(device.clone()).map_err(|err| err.to_string())?;
        let ptx = compile_cuda_module()?;
        device
            .load_ptx(ptx, MODULE_NAME, &["matmul_f32", "reduce_sum_f32"])
            .map_err(|err| err.to_string())?;
        let reduce = device
            .get_func(MODULE_NAME, "reduce_sum_f32")
            .ok_or_else(|| "cuda reduce kernel not found".to_string())?;
        let matmul_fallback = device.get_func(MODULE_NAME, "matmul_f32");
        Ok(CudaState {
            device,
            blas: Arc::new(blas),
            reduce,
            matmul_fallback,
        })
    });

    fn compile_cuda_module() -> Result<cudarc::nvrtc::Ptx, String> {
        let mut opts = CompileOptions::default();
        opts.include_paths = cuda_include_paths();
        opts.use_fast_math = Some(true);
        match compile_ptx_with_opts(CUDA_KERNEL_SOURCE, opts) {
            Ok(ptx) => Ok(ptx),
            Err(CompileError::CompileError { log, .. }) => {
                Err(format!("nvrtc compile failed: {}", log.to_string_lossy()))
            }
            Err(err) => Err(format!("nvrtc compile failed: {err:?}")),
        }
    }

    fn cuda_include_paths() -> Vec<String> {
        let mut paths = BTreeSet::new();
        for key in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
            if let Ok(base) = std::env::var(key) {
                let candidate = PathBuf::from(&base).join("include");
                if candidate.exists() {
                    paths.insert(candidate.to_string_lossy().into_owned());
                }
            }
        }
        if let Ok(path) = std::env::var("CUDA_INC_PATH") {
            let candidate = PathBuf::from(path);
            if candidate.exists() {
                paths.insert(candidate.to_string_lossy().into_owned());
            }
        }
        paths.into_iter().collect()
    }

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
            return Err(format!(
                "cuda matmul: lhs length mismatch (expected {lhs_elems}, got {})",
                a.len()
            )
            .into());
        }
        let rhs_elems = k
            .checked_mul(n)
            .ok_or_else(|| "cuda matmul: rhs dimension overflow".to_string())?;
        if b.len() != rhs_elems {
            return Err(format!(
                "cuda matmul: rhs length mismatch (expected {rhs_elems}, got {})",
                b.len()
            )
            .into());
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

        let device = &state.device;
        let a_dev = device
            .htod_sync_copy(a)
            .map_err(|err| format!("cuda htod_sync_copy(lhs): {err}"))?;
        let b_dev = device
            .htod_sync_copy(b)
            .map_err(|err| format!("cuda htod_sync_copy(rhs): {err}"))?;
        let mut c_dev = device
            .alloc_zeros::<f32>(total_elems)
            .map_err(|err| format!("cuda alloc_zeros(out): {err}"))?;

        let gemm_cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n_i32 as c_int,
            n: m_i32 as c_int,
            k: k_i32 as c_int,
            alpha: 1.0f32,
            lda: n_i32 as c_int,
            ldb: k_i32 as c_int,
            beta: 0.0f32,
            ldc: n_i32 as c_int,
        };

        let blas = &state.blas;
        if let Err(err) = unsafe { blas.gemm(gemm_cfg, &b_dev, &a_dev, &mut c_dev) }
            .map_err(|err| format!("cuda cublas sgemm failed: {err}"))
        {
            if let Some(kernel) = state.matmul_fallback.as_ref() {
                let cfg = launch_config_for_len(total_elems)?;
                unsafe {
                    kernel
                        .clone()
                        .launch(cfg, (&a_dev, &b_dev, &mut c_dev, m_i32, k_i32, n_i32))
                }
                .map_err(|launch_err| {
                    format!("cuda matmul fallback launch failed: {launch_err}")
                })?;
            } else {
                return Err(err.into());
            }
        }

        device
            .synchronize()
            .map_err(|err| format!("cuda device synchronize: {err}"))?;
        let host = device
            .dtoh_sync_copy(&c_dev)
            .map_err(|err| format!("cuda dtoh(out): {err}"))?;
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
        let device = &state.device;
        let input_dev = device
            .htod_sync_copy(values)
            .map_err(|err| format!("cuda htod_sync_copy(input): {err}"))?;
        let mut output_dev = device
            .alloc_zeros::<f32>(1)
            .map_err(|err| format!("cuda alloc_zeros(sum): {err}"))?;
        let cfg = launch_config_for_len(values.len())?;
        let len_i32 = i32::try_from(values.len())
            .map_err(|_| "cuda reduce: input length exceeds i32::MAX".to_string())?;
        unsafe {
            state
                .reduce
                .clone()
                .launch(cfg, (&input_dev, &mut output_dev, len_i32))
        }
        .map_err(|err| format!("cuda reduce launch failed: {err}"))?;
        device
            .synchronize()
            .map_err(|err| format!("cuda device synchronize: {err}"))?;
        let host = device
            .dtoh_sync_copy(&output_dev)
            .map_err(|err| format!("cuda dtoh(sum): {err}"))?;
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
