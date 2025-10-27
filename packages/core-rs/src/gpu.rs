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
    active_backend_kind().map(|k| k.as_str())
}

#[derive(Clone, Copy, Debug)]
pub enum NanPolicy {
    Ignore,
    Propagate,
}

pub fn reduce_sum_f32(values: &[f32]) -> CoreResult<f32> {
    #[cfg(feature = "gpu-cuda")]
    {
        if cuda::is_available() {
            return cuda::reduce_sum_f32(values);
        }
    }
    Ok(values.iter().copied().sum())
}

pub fn reduce_max_f32(values: &[f32]) -> CoreResult<f32> {
    reduce_max_f32_with_policy(values, NanPolicy::Ignore)
}

pub fn argmax_f32(values: &[f32]) -> CoreResult<usize> {
    argmax_f32_with_policy(values, NanPolicy::Ignore)
}

pub fn reduce_max_f32_with_policy(values: &[f32], policy: NanPolicy) -> CoreResult<f32> {
    if values.is_empty() {
        return Err("reduce_max on empty slice".into());
    }
    #[cfg(feature = "gpu-cuda")]
    {
        if cuda::is_available() {
            return cuda::reduce_max_f32_with_policy(values, policy);
        }
    }
    match policy {
        NanPolicy::Ignore => {
            let mut any_valid = false;
            let mut best = f32::NEG_INFINITY;
            for &v in values {
                if v.is_nan() {
                    continue;
                }
                any_valid = true;
                if v > best {
                    best = v;
                }
            }
            if any_valid {
                Ok(best)
            } else {
                Ok(f32::NAN)
            }
        }
        NanPolicy::Propagate => {
            if values.iter().any(|v| v.is_nan()) {
                Ok(f32::NAN)
            } else {
                Ok(values.iter().copied().fold(f32::NEG_INFINITY, f32::max))
            }
        }
    }
}

pub fn argmax_f32_with_policy(values: &[f32], policy: NanPolicy) -> CoreResult<usize> {
    if values.is_empty() {
        return Err("argmax on empty slice".into());
    }
    #[cfg(feature = "gpu-cuda")]
    {
        if cuda::is_available() {
            return cuda::argmax_f32_with_policy(values, policy);
        }
    }
    match policy {
        NanPolicy::Ignore => {
            let mut best_idx: Option<usize> = None;
            let mut best_val = f32::NEG_INFINITY;
            for (idx, &value) in values.iter().enumerate() {
                if value.is_nan() {
                    continue;
                }
                if best_idx.is_none() || value > best_val {
                    best_idx = Some(idx);
                    best_val = value;
                }
            }
            Ok(best_idx.unwrap_or(0))
        }
        NanPolicy::Propagate => {
            if let Some(idx) = values.iter().position(|v| v.is_nan()) {
                Ok(idx)
            } else {
                let mut best_idx = 0usize;
                let mut best_val = values[0];
                for (idx, &value) in values.iter().enumerate().skip(1) {
                    if value > best_val {
                        best_idx = idx;
                        best_val = value;
                    }
                }
                Ok(best_idx)
            }
        }
    }
}

pub fn matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> CoreResult<Vec<f32>> {
    if a.len() != m.saturating_mul(k) || b.len() != k.saturating_mul(n) {
        return Err("matmul_f32: shape mismatch".into());
    }
    #[cfg(feature = "gpu-cuda")]
    {
        if cuda::is_available() {
            return cuda::matmul_f32(a, b, m, k, n);
        }
    }
    Ok(matmul_cpu(a, b, m, k, n))
}

pub fn matmul_f32_ex(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> CoreResult<Vec<f32>> {
    #[cfg(feature = "gpu-cuda")]
    {
        if cuda::is_available() {
            return cuda::matmul_f32_ex(a, b, m, k, n, trans_a, trans_b);
        }
    }
    Ok(matmul_cpu_ex(a, b, m, k, n, trans_a, trans_b))
}

pub fn matmul_batched_f32(
    a: &[f32],
    b: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> CoreResult<Vec<f32>> {
    #[cfg(feature = "gpu-cuda")]
    {
        if cuda::is_available() {
            return cuda::matmul_batched_f32(a, b, batch, m, k, n, trans_a, trans_b);
        }
    }
    let mut out = vec![0.0f32; batch.saturating_mul(m.saturating_mul(n))];
    let a_each = m.saturating_mul(k);
    let b_each = k.saturating_mul(n);
    let c_each = m.saturating_mul(n);
    for t in 0..batch {
        let a_slice = &a[t * a_each..(t + 1) * a_each];
        let b_slice = &b[t * b_each..(t + 1) * b_each];
        let c_slice = &mut out[t * c_each..(t + 1) * c_each];
        let tmp = matmul_cpu_ex(a_slice, b_slice, m, k, n, trans_a, trans_b);
        c_slice.copy_from_slice(&tmp);
    }
    Ok(out)
}

pub fn matmul_batched_f32_strided(
    a: &[f32],
    b: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
) -> CoreResult<Vec<f32>> {
    #[cfg(feature = "gpu-cuda")]
    {
        if cuda::is_available() {
            return cuda::matmul_batched_f32_strided(
                a, b, batch, m, k, n, trans_a, trans_b, stride_a, stride_b, stride_c,
            );
        }
    }
    let need_a = (m as i64) * (k as i64);
    let need_b = (k as i64) * (n as i64);
    let need_c = (m as i64) * (n as i64);
    if stride_a != need_a || stride_b != need_b || stride_c != need_c {
        return Err("CPU fallback does not support non-contiguous strided batched matmul".into());
    }
    matmul_batched_f32(a, b, batch, m, k, n, trans_a, trans_b)
}

fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            let row = i * n;
            let b_row = p * n;
            for j in 0..n {
                out[row + j] += a_ip * b[b_row + j];
            }
        }
    }
    out
}

fn matmul_cpu_ex(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Vec<f32> {
    let (_ma, ka) = if trans_a { (k, m) } else { (m, k) };
    let (kb, _nb) = if trans_b { (n, k) } else { (k, n) };
    assert_eq!(ka, kb, "inner dim mismatch");
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                let a_val = if trans_a { a[p * m + i] } else { a[i * k + p] };
                let b_val = if trans_b { b[j * k + p] } else { b[p * n + j] };
                acc += a_val * b_val;
            }
            out[i * n + j] = acc;
        }
    }
    out
}

#[cfg(feature = "gpu-cuda")]
mod cuda {
    use super::CoreResult;
    use crate::gpu::NanPolicy;
    use cudarc::cublas::sys::cublasOperation_t;
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
    use cudarc::driver::{
        CudaContext, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig,
        PushKernelArg,
    };
    use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
    use std::{collections::HashSet, env, fs, path::PathBuf, sync::Arc};

    const REDUCE_KERNEL_SRC: &str = r#"
#include <math.h>
#define BLOCK_SIZE 256

extern "C" __global__ void reduce_sum_stage1(const float* __restrict__ input,
                                             float* __restrict__ partial,
                                             int length) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int global = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    float acc = 0.0f;
    for (int idx = global; idx < length; idx += stride) {
        acc += input[idx];
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

extern "C" __global__ void reduce_max_stage1_ignore_nan(const float* __restrict__ input,
                                                        float* __restrict__ partial,
                                                        int length) {
    __shared__ float smax[BLOCK_SIZE];
    int tid = threadIdx.x;
    int global = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    float best = -INFINITY;
    for (int idx = global; idx < length; idx += stride) {
        float v = input[idx];
        if (!isnan(v) && v > best) {
            best = v;
        }
    }
    smax[tid] = best;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other = smax[tid + s];
            if (other > smax[tid]) {
                smax[tid] = other;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial[blockIdx.x] = smax[0];
    }
}

extern "C" __global__ void argmax_stage1_ignore_nan(const float* __restrict__ input,
                                                    float* __restrict__ pmax,
                                                    int* __restrict__ pidx,
                                                    int length) {
    __shared__ float smax[BLOCK_SIZE];
    __shared__ int sarg[BLOCK_SIZE];
    int tid = threadIdx.x;
    int global = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    float best_val = -INFINITY;
    int best_idx = -1;
    for (int idx = global; idx < length; idx += stride) {
        float v = input[idx];
        if (!isnan(v)) {
            if (best_idx < 0 || v > best_val || (v == best_val && idx < best_idx)) {
                best_val = v;
                best_idx = idx;
            }
        }
    }
    smax[tid] = best_val;
    sarg[tid] = best_idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = smax[tid + s];
            int oi = sarg[tid + s];
            float cv = smax[tid];
            int ci = sarg[tid];
            if (oi >= 0 && (ci < 0 || ov > cv || (ov == cv && oi < ci))) {
                smax[tid] = ov;
                sarg[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        pmax[blockIdx.x] = smax[0];
        pidx[blockIdx.x] = sarg[0];
    }
}

extern "C" __global__ void first_nan_index_stage1(const float* __restrict__ input,
                                                  int* __restrict__ pmin,
                                                  int length) {
    __shared__ int smin[BLOCK_SIZE];
    int tid = threadIdx.x;
    int global = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    int best = length;
    for (int idx = global; idx < length; idx += stride) {
        float v = input[idx];
        if (isnan(v) && idx < best) {
            best = idx;
        }
    }
    smin[tid] = best;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int other = smin[tid + s];
            if (other < smin[tid]) {
                smin[tid] = other;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        pmin[blockIdx.x] = smin[0];
    }
}
"#;

    pub fn is_available() -> bool {
        match CudaContext::new(0) {
            Ok(ctx) => {
                let stream = ctx.default_stream();
                CudaBlas::new(stream).is_ok()
            }
            Err(_) => false,
        }
    }

    fn build_cublas_config_row_major(
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Result<GemmConfig<f32>, String> {
        let to_i32 = |x: usize| -> Result<i32, String> {
            if x > i32::MAX as usize {
                Err("gemm dims exceed i32".into())
            } else {
                Ok(x as i32)
            }
        };
        let m_i = to_i32(m)?;
        let n_i = to_i32(n)?;
        let k_i = to_i32(k)?;
        let transa_cublas =
            if trans_b { cublasOperation_t::CUBLAS_OP_T } else { cublasOperation_t::CUBLAS_OP_N };
        let transb_cublas =
            if trans_a { cublasOperation_t::CUBLAS_OP_T } else { cublasOperation_t::CUBLAS_OP_N };
        let lda = if trans_b { k_i } else { n_i };
        let ldb = if trans_a { m_i } else { k_i };
        let ldc = n_i;
        Ok(GemmConfig {
            transa: transa_cublas,
            transb: transb_cublas,
            m: n_i,
            n: m_i,
            k: k_i,
            alpha: 1.0,
            lda,
            ldb,
            beta: 0.0,
            ldc,
        })
    }

    fn load_reduce_module(ctx: &Arc<CudaContext>) -> Result<Arc<CudaModule>, String> {
        let mut opts = CompileOptions::default();
        opts.include_paths = nvrtc_include_paths();
        let ptx =
            compile_ptx_with_opts(REDUCE_KERNEL_SRC, opts).map_err(|e| format!("nvrtc: {:?}", e))?;
        ctx.load_module(ptx)
            .map_err(|e| format!("load module: {:?}", e))
    }

    fn nvrtc_include_paths() -> Vec<String> {
        let mut seen = HashSet::new();
        let mut push_path = |acc: &mut Vec<String>, path: PathBuf| {
            if path.exists() {
                let s = path_to_string(&path);
                if seen.insert(s.clone()) {
                    acc.push(s);
                }
            }
        };

        let mut paths = Vec::new();
        for key in ["CUDA_PATH", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR"] {
            if let Ok(val) = env::var(key) {
                push_path(&mut paths, PathBuf::from(val).join("include"));
            }
        }

        push_path(&mut paths, PathBuf::from("/usr/local/cuda/include"));
        if cfg!(target_os = "windows") {
            let default_root = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
            if default_root.exists() {
                if let Ok(entries) = fs::read_dir(default_root) {
                    for entry in entries.flatten() {
                        let include_dir = entry.path().join("include");
                        push_path(&mut paths, include_dir);
                    }
                }
            }
        }

        paths
    }

    fn path_to_string(path: &PathBuf) -> String {
        path.to_string_lossy().into_owned()
    }

    pub fn matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> CoreResult<Vec<f32>> {
        matmul_f32_ex(a, b, m, k, n, false, false)
    }

    pub fn matmul_f32_ex(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> CoreResult<Vec<f32>> {
        let need_a = if trans_a { k.checked_mul(m) } else { m.checked_mul(k) }.ok_or("size overflow")?;
        let need_b = if trans_b { n.checked_mul(k) } else { k.checked_mul(n) }.ok_or("size overflow")?;
        if a.len() != need_a || b.len() != need_b {
            return Err("matmul_ex: shape mismatch".into());
        }
        let total = m.checked_mul(n).ok_or("size overflow")?;
        if total == 0 {
            return Ok(Vec::new());
        }
        let ctx = CudaContext::new(0).map_err(|e| format!("cuda: {:?}", e))?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("cublas: {:?}", e))?;
        let d_a = stream
            .memcpy_stod(a)
            .map_err(|e| format!("htod a: {:?}", e))?;
        let d_b = stream
            .memcpy_stod(b)
            .map_err(|e| format!("htod b: {:?}", e))?;
        let mut d_c = stream
            .alloc_zeros::<f32>(total)
            .map_err(|e| format!("alloc c: {:?}", e))?;
        let cfg = build_cublas_config_row_major(m, k, n, trans_a, trans_b)?;
        unsafe { blas.gemm(cfg, &d_b, &d_a, &mut d_c) }
            .map_err(|e| format!("sgemm: {:?}", e))?;
        stream
            .memcpy_dtov(&d_c)
            .map_err(|e| format!("dtoh c: {:?}", e))
    }

    pub fn matmul_batched_f32(
        a: &[f32],
        b: &[f32],
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> CoreResult<Vec<f32>> {
        let a_each = if trans_a { k.checked_mul(m) } else { m.checked_mul(k) }.ok_or("size overflow")?;
        let b_each = if trans_b { n.checked_mul(k) } else { k.checked_mul(n) }.ok_or("size overflow")?;
        let c_each = m.checked_mul(n).ok_or("size overflow")?;
        if a.len() != batch.checked_mul(a_each).ok_or("size overflow")?
            || b.len() != batch.checked_mul(b_each).ok_or("size overflow")?
        {
            return Err("matmul_batched_f32: shape mismatch".into());
        }
        matmul_batched_f32_strided(
            a,
            b,
            batch,
            m,
            k,
            n,
            trans_a,
            trans_b,
            a_each as i64,
            b_each as i64,
            c_each as i64,
        )
    }

    pub fn matmul_batched_f32_strided(
        a: &[f32],
        b: &[f32],
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
    ) -> CoreResult<Vec<f32>> {
        let a_need = stride_a.checked_mul(batch as i64).ok_or("stride overflow")?;
        let b_need = stride_b.checked_mul(batch as i64).ok_or("stride overflow")?;
        let c_need = stride_c.checked_mul(batch as i64).ok_or("stride overflow")? as usize;
        if (a.len() as i64) < a_need || (b.len() as i64) < b_need {
            return Err("matmul_batched_f32_strided: buffer too small for given stride".into());
        }
        let ctx = CudaContext::new(0).map_err(|e| format!("cuda: {:?}", e))?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("cublas: {:?}", e))?;
        let d_a = stream
            .memcpy_stod(a)
            .map_err(|e| format!("htod a: {:?}", e))?;
        let d_b = stream
            .memcpy_stod(b)
            .map_err(|e| format!("htod b: {:?}", e))?;
        let mut d_c = stream
            .alloc_zeros::<f32>(c_need)
            .map_err(|e| format!("alloc c: {:?}", e))?;
        let gemm_cfg = build_cublas_config_row_major(m, k, n, trans_a, trans_b)?;
        let cfg = StridedBatchedConfig::<f32> {
            gemm: gemm_cfg,
            batch_size: batch as i32,
            stride_a: stride_b,
            stride_b: stride_a,
            stride_c,
        };
        unsafe { blas.gemm_strided_batched(cfg, &d_b, &d_a, &mut d_c) }
            .map_err(|e| format!("sgemm_strided_batched: {:?}", e))?;
        stream
            .memcpy_dtov(&d_c)
            .map_err(|e| format!("dtoh c: {:?}", e))
    }

    pub fn matmul_f32_ex_device<A: DevicePtr<f32>, B: DevicePtr<f32>, Cc: DevicePtrMut<f32>>(
        stream: Arc<CudaStream>,
        a: &A,
        b: &B,
        c: &mut Cc,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> CoreResult<()> {
        let (ap, _) = a.device_ptr(&stream);
        let (bp, _) = b.device_ptr(&stream);
        let (cp, _) = c.device_ptr_mut(&stream);
        if cp == ap || cp == bp {
            return Err("matmul_f32_ex_device: output aliases input".into());
        }
        let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("cublas: {:?}", e))?;
        let cfg = build_cublas_config_row_major(m, k, n, trans_a, trans_b)?;
        unsafe { blas.gemm(cfg, b, a, c) }.map_err(|e| format!("sgemm: {:?}", e))
    }

    pub fn matmul_batched_f32_strided_device<
        A: DevicePtr<f32>,
        B: DevicePtr<f32>,
        Cc: DevicePtrMut<f32>,
    >(
        stream: Arc<CudaStream>,
        a: &A,
        b: &B,
        c: &mut Cc,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
    ) -> CoreResult<()> {
        let (ap, _) = a.device_ptr(&stream);
        let (bp, _) = b.device_ptr(&stream);
        let (cp, _) = c.device_ptr_mut(&stream);
        if cp == ap || cp == bp {
            return Err("matmul_batched_f32_strided_device: output aliases input".into());
        }
        let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("cublas: {:?}", e))?;
        let gemm_cfg = build_cublas_config_row_major(m, k, n, trans_a, trans_b)?;
        let cfg = StridedBatchedConfig::<f32> {
            gemm: gemm_cfg,
            batch_size: batch as i32,
            stride_a: stride_b,
            stride_b: stride_a,
            stride_c,
        };
        unsafe { blas.gemm_strided_batched(cfg, b, a, c) }.map_err(|e| format!("sgemm_strided_batched: {:?}", e))
    }

    pub fn reduce_sum_f32(values: &[f32]) -> CoreResult<f32> {
        let n = values.len();
        if n == 0 {
            return Ok(0.0);
        }
        let ctx = CudaContext::new(0).map_err(|e| format!("cuda: {:?}", e))?;
        let stream = ctx.default_stream();
        let module = load_reduce_module(&ctx)?;
        let func = module
            .load_function("reduce_sum_stage1")
            .map_err(|e| format!("get func: {:?}", e))?;
        let block: u32 = 256;
        let blocks = ((n as u32 + block - 1) / block).max(1);
        let d_in = stream
            .memcpy_stod(values)
            .map_err(|e| format!("htod: {:?}", e))?;
        let mut tmp: CudaSlice<f32> = stream
            .alloc_zeros(blocks as usize)
            .map_err(|e| format!("alloc tmp: {:?}", e))?;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&d_in)
                .arg(&mut tmp)
                .arg(&((n as i32)))
                .launch(cfg)
        }
        .map_err(|e| format!("launch: {:?}", e))?;
        let host = stream
            .memcpy_dtov(&tmp)
            .map_err(|e| format!("dtoh: {:?}", e))?;
        Ok(host.into_iter().sum())
    }

    pub fn reduce_max_f32_with_policy(values: &[f32], policy: NanPolicy) -> CoreResult<f32> {
        let n = values.len();
        if n == 0 {
            return Err("reduce_max on empty slice".into());
        }
        let ctx = CudaContext::new(0).map_err(|e| format!("cuda: {:?}", e))?;
        let stream = ctx.default_stream();
        let module = load_reduce_module(&ctx)?;
        let k_max = module
            .load_function("reduce_max_stage1_ignore_nan")
            .map_err(|e| format!("get func: {:?}", e))?;
        let k_nan = module
            .load_function("first_nan_index_stage1")
            .map_err(|e| format!("get func: {:?}", e))?;
        let block: u32 = 256;
        let d_in = stream
            .memcpy_stod(values)
            .map_err(|e| format!("htod: {:?}", e))?;
        if matches!(policy, NanPolicy::Propagate) {
            let blocks = ((n as u32 + block - 1) / block).max(1);
            let mut tmp_nan = stream
                .alloc_zeros::<i32>(blocks as usize)
                .map_err(|e| format!("alloc tmp_nan: {:?}", e))?;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&k_nan)
                    .arg(&d_in)
                    .arg(&mut tmp_nan)
                    .arg(&((n as i32)))
                    .launch(cfg)
            }
            .map_err(|e| format!("launch first_nan: {:?}", e))?;
            let host = stream
                .memcpy_dtov(&tmp_nan)
                .map_err(|e| format!("dtoh: {:?}", e))?;
            let mut min_idx = n as i32;
            for v in host {
                if v < min_idx {
                    min_idx = v;
                }
            }
            if min_idx < n as i32 {
                return Ok(f32::NAN);
            }
        }
        let blocks = ((n as u32 + block - 1) / block).max(1);
        let mut tmp = stream
            .alloc_zeros::<f32>(blocks as usize)
            .map_err(|e| format!("alloc tmp: {:?}", e))?;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&k_max)
                .arg(&d_in)
                .arg(&mut tmp)
                .arg(&((n as i32)))
                .launch(cfg)
        }
        .map_err(|e| format!("launch max: {:?}", e))?;
        let host = stream
            .memcpy_dtov(&tmp)
            .map_err(|e| format!("dtoh: {:?}", e))?;
        let mut max_val = f32::NEG_INFINITY;
        for v in host {
            if v > max_val {
                max_val = v;
            }
        }
        if matches!(policy, NanPolicy::Ignore) && !values.iter().any(|v| !v.is_nan()) {
            Ok(f32::NAN)
        } else {
            Ok(max_val)
        }
    }

    pub fn argmax_f32_with_policy(values: &[f32], policy: NanPolicy) -> CoreResult<usize> {
        let n = values.len();
        if n == 0 {
            return Err("argmax on empty slice".into());
        }
        let ctx = CudaContext::new(0).map_err(|e| format!("cuda: {:?}", e))?;
        let stream = ctx.default_stream();
        let module = load_reduce_module(&ctx)?;
        let k_argmax = module
            .load_function("argmax_stage1_ignore_nan")
            .map_err(|e| format!("get func: {:?}", e))?;
        let k_nan = module
            .load_function("first_nan_index_stage1")
            .map_err(|e| format!("get func: {:?}", e))?;
        let block: u32 = 256;
        let d_in = stream
            .memcpy_stod(values)
            .map_err(|e| format!("htod: {:?}", e))?;
        if matches!(policy, NanPolicy::Propagate) {
            let blocks = ((n as u32 + block - 1) / block).max(1);
            let mut tmp_nan = stream
                .alloc_zeros::<i32>(blocks as usize)
                .map_err(|e| format!("alloc tmp_nan: {:?}", e))?;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&k_nan)
                    .arg(&d_in)
                    .arg(&mut tmp_nan)
                    .arg(&((n as i32)))
                    .launch(cfg)
            }
            .map_err(|e| format!("launch first_nan: {:?}", e))?;
            let host = stream
                .memcpy_dtov(&tmp_nan)
                .map_err(|e| format!("dtoh: {:?}", e))?;
            let mut min_idx = n as i32;
            for v in host {
                if v < min_idx {
                    min_idx = v;
                }
            }
            if min_idx < n as i32 {
                return Ok(min_idx as usize);
            }
        }
        let blocks = ((n as u32 + block - 1) / block).max(1);
        let mut pmax = stream
            .alloc_zeros::<f32>(blocks as usize)
            .map_err(|e| format!("alloc pmax: {:?}", e))?;
        let mut pidx = stream
            .alloc_zeros::<i32>(blocks as usize)
            .map_err(|e| format!("alloc pidx: {:?}", e))?;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&k_argmax)
                .arg(&d_in)
                .arg(&mut pmax)
                .arg(&mut pidx)
                .arg(&((n as i32)))
                .launch(cfg)
        }
        .map_err(|e| format!("launch argmax: {:?}", e))?;
        let hv = stream
            .memcpy_dtov(&pmax)
            .map_err(|e| format!("dtoh: {:?}", e))?;
        let hi = stream
            .memcpy_dtov(&pidx)
            .map_err(|e| format!("dtoh: {:?}", e))?;
        let mut best_val = f32::NEG_INFINITY;
        let mut best_idx: i32 = -1;
        for (val, idx) in hv.into_iter().zip(hi.into_iter()) {
            if idx >= 0 && (best_idx < 0 || val > best_val || (val == best_val && idx < best_idx)) {
                best_val = val;
                best_idx = idx;
            }
        }
        Ok(if best_idx < 0 { 0usize } else { best_idx as usize })
    }
}

#[cfg(feature = "gpu-rocm")]
mod rocm {
    pub fn is_available() -> bool {
        false
    }
    pub fn matmul_f32(
        _a: &[f32],
        _b: &[f32],
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> Result<Vec<f32>, String> {
        Err("rocm not implemented".into())
    }
    pub fn matmul_f32_ex(
        _a: &[f32],
        _b: &[f32],
        _m: usize,
        _k: usize,
        _n: usize,
        _ta: bool,
        _tb: bool,
    ) -> Result<Vec<f32>, String> {
        Err("rocm not implemented".into())
    }
    pub fn matmul_batched_f32(
        _a: &[f32],
        _b: &[f32],
        _batch: usize,
        _m: usize,
        _k: usize,
        _n: usize,
        _ta: bool,
        _tb: bool,
    ) -> Result<Vec<f32>, String> {
        Err("rocm not implemented".into())
    }
    pub fn matmul_batched_f32_strided(
        _a: &[f32],
        _b: &[f32],
        _batch: usize,
        _m: usize,
        _k: usize,
        _n: usize,
        _ta: bool,
        _tb: bool,
        _sa: i64,
        _sb: i64,
        _sc: i64,
    ) -> Result<Vec<f32>, String> {
        Err("rocm not implemented".into())
    }
}

#[cfg(feature = "gpu-cuda")]
pub use cuda::{matmul_batched_f32_strided_device, matmul_f32_ex_device};

#[cfg(all(test, feature = "gpu-cuda"))]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext;

    fn assert_close(lhs: &[f32], rhs: &[f32]) {
        assert_eq!(lhs.len(), rhs.len(), "length mismatch");
        for (a, b) in lhs.iter().zip(rhs.iter()) {
            assert!((a - b).abs() < 1e-4, "values differ: {a} vs {b}");
        }
    }

    fn cuda_available() -> bool {
        super::cuda::is_available()
    }

    #[test]
    fn matmul_ex_matches_cpu_all_transposes() {
        if !cuda_available() {
            eprintln!("skipping matmul_ex_matches_cpu_all_transposes: no CUDA device");
            return;
        }
        let m = 2;
        let k = 3;
        let n = 2;
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 + 1.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 + 0.5) * 0.5).collect();
        let cases = [
            (false, false),
            (true, false),
            (false, true),
            (true, true),
        ];
        for (trans_a, trans_b) in cases {
            let gpu = super::cuda::matmul_f32_ex(&a, &b, m, k, n, trans_a, trans_b)
                .expect("gpu matmul");
            let cpu = super::matmul_cpu_ex(&a, &b, m, k, n, trans_a, trans_b);
            assert_close(&gpu, &cpu);
        }
    }

    #[test]
    fn matmul_device_matches_cpu() {
        if !cuda_available() {
            eprintln!("skipping matmul_device_matches_cpu: no CUDA device");
            return;
        }
        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let m = 3;
        let k = 2;
        let n = 2;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 + 0.75) * 0.3).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.25) * 0.2).collect();
        let d_a = stream.memcpy_stod(&a).expect("d_a");
        let d_b = stream.memcpy_stod(&b).expect("d_b");
        let mut d_c = stream
            .alloc_zeros::<f32>(m * n)
            .expect("alloc d_c");
        super::matmul_f32_ex_device(
            stream.clone(),
            &d_a,
            &d_b,
            &mut d_c,
            m,
            k,
            n,
            false,
            false,
        )
        .expect("device matmul");
        let gpu = stream.memcpy_dtov(&d_c).expect("dtoh c");
        let cpu = super::matmul_cpu_ex(&a, &b, m, k, n, false, false);
        assert_close(&gpu, &cpu);
    }

    #[test]
    fn matmul_device_strided_matches_cpu() {
        if !cuda_available() {
            eprintln!("skipping matmul_device_strided_matches_cpu: no CUDA device");
            return;
        }
        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let batch = 2;
        let m = 2;
        let k = 2;
        let n = 2;
        let stride_a = (m * k + 2) as i64;
        let stride_b = (k * n + 3) as i64;
        let stride_c = (m * n + 4) as i64;
        let mut host_a = vec![0.0f32; stride_a as usize * batch];
        let mut host_b = vec![0.0f32; stride_b as usize * batch];
        for b in 0..batch {
            for i in 0..(m * k) {
                host_a[b * stride_a as usize + i] = (b * 10 + i) as f32 + 0.1;
            }
            for i in 0..(k * n) {
                host_b[b * stride_b as usize + i] = (b * 5 + i) as f32 - 0.2;
            }
        }
        let d_a = stream.memcpy_stod(&host_a).expect("d_a");
        let d_b = stream.memcpy_stod(&host_b).expect("d_b");
        let mut d_c = stream
            .alloc_zeros::<f32>(stride_c as usize * batch)
            .expect("alloc d_c");
        super::matmul_batched_f32_strided_device(
            stream.clone(),
            &d_a,
            &d_b,
            &mut d_c,
            batch,
            m,
            k,
            n,
            false,
            false,
            stride_a,
            stride_b,
            stride_c,
        )
        .expect("device batched strided matmul");
        let gpu = stream.memcpy_dtov(&d_c).expect("dtoh c");
        for b in 0..batch {
            let a_block = &host_a[b * stride_a as usize..][..m * k];
            let b_block = &host_b[b * stride_b as usize..][..k * n];
            let expected = super::matmul_cpu_ex(a_block, b_block, m, k, n, false, false);
            let gpu_block = &gpu[b * stride_c as usize..][..m * n];
            assert_close(gpu_block, &expected);
        }
    }

    #[test]
    fn reduce_max_nan_policy_behaviour() {
        if !cuda_available() {
            eprintln!("skipping reduce_max_nan_policy_behaviour: no CUDA device");
            return;
        }
        let values = vec![1.0f32, f32::NAN, 3.5, -2.0];
        let res_ignore =
            super::reduce_max_f32_with_policy(&values, NanPolicy::Ignore).expect("ignore");
        assert!((res_ignore - 3.5).abs() < 1e-5);
        let res_prop =
            super::reduce_max_f32_with_policy(&values, NanPolicy::Propagate).expect("prop");
        assert!(res_prop.is_nan());
        let nan_only = vec![f32::NAN, f32::NAN];
        let res_ignore =
            super::reduce_max_f32_with_policy(&nan_only, NanPolicy::Ignore).expect("nan only");
        assert!(res_ignore.is_nan());
    }

    #[test]
    fn argmax_nan_policy_behaviour() {
        if !cuda_available() {
            eprintln!("skipping argmax_nan_policy_behaviour: no CUDA device");
            return;
        }
        let values = vec![0.1f32, f32::NAN, 0.5, 0.3];
        let res_ignore =
            super::argmax_f32_with_policy(&values, NanPolicy::Ignore).expect("ignore");
        assert_eq!(res_ignore, 2);
        let res_prop =
            super::argmax_f32_with_policy(&values, NanPolicy::Propagate).expect("prop");
        assert_eq!(res_prop, 1);
        let nan_only = vec![f32::NAN, f32::NAN];
        let res_ignore =
            super::argmax_f32_with_policy(&nan_only, NanPolicy::Ignore).expect("nan only");
        assert_eq!(res_ignore, 0);
    }
}
