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
pub enum NanPolicy { Ignore, Propagate }

pub fn reduce_sum_f32(values: &[f32]) -> CoreResult<f32> {
    let mut acc = 0.0f32;
    for &v in values { acc += v; }
    Ok(acc)
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
    match policy {
        NanPolicy::Ignore => {
            let mut found_any = false;
            let mut max_val = f32::NEG_INFINITY;
            for &v in values {
                if v.is_nan() { continue; }
                found_any = true;
                if v > max_val { max_val = v; }
            }
            if !found_any { Ok(f32::NAN) } else { Ok(max_val) }
        }
        NanPolicy::Propagate => {
            for &v in values { if v.is_nan() { return Ok(f32::NAN); } }
            let mut max_val = f32::NEG_INFINITY;
            for &v in values { if v > max_val { max_val = v; } }
            Ok(max_val)
        }
    }
}

pub fn argmax_f32_with_policy(values: &[f32], policy: NanPolicy) -> CoreResult<usize> {
    if values.is_empty() {
        return Err("argmax on empty slice".into());
    }
    match policy {
        NanPolicy::Ignore => {
            let mut best_idx: Option<usize> = None;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in values.iter().enumerate() {
                if v.is_nan() { continue; }
                if best_idx.is_none() || v > best_val { best_idx = Some(i); best_val = v; }
            }
            Ok(best_idx.unwrap_or(0))
        }
        NanPolicy::Propagate => {
            for (i, &v) in values.iter().enumerate() {
                if v.is_nan() { return Ok(i); }
            }
            let mut best_idx = 0usize;
            let mut best_val = values[0];
            for (i, &v) in values.iter().enumerate().skip(1) {
                if v > best_val { best_idx = i; best_val = v; }
            }
            Ok(best_idx)
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
    a: &[f32], b: &[f32], m: usize, k: usize, n: usize, trans_a: bool, trans_b: bool,
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
    a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize, trans_a: bool, trans_b: bool,
) -> CoreResult<Vec<f32>> {
    #[cfg(feature = "gpu-cuda")]
    {
        if cuda::is_available() {
            return cuda::matmul_batched_f32(a, b, batch, m, k, n, trans_a, trans_b);
        }
    }
    // CPU fallback
    let mut out = vec![0.0f32; batch.saturating_mul(m.saturating_mul(n))];
    let a_each = m.saturating_mul(k);
    let b_each = k.saturating_mul(n);
    let c_each = m.saturating_mul(n);
    for t in 0..batch {
        let a_slice = &a[t * a_each..(t + 1) * a_each];
        let b_slice = &b[t * b_each..(t + 1) * b_each];
        let c = &mut out[t * c_each..(t + 1) * c_each];
        let tmp = matmul_cpu_ex(a_slice, b_slice, m, k, n, trans_a, trans_b);
        c.copy_from_slice(&tmp);
    }
    Ok(out)
}

pub fn matmul_batched_f32_strided(
    a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize,
    trans_a: bool, trans_b: bool, stride_a: i64, stride_b: i64, stride_c: i64,
) -> CoreResult<Vec<f32>> {
    #[cfg(feature = "gpu-cuda")]
    {
        if cuda::is_available() {
            return cuda::matmul_batched_f32_strided(
                a, b, batch, m, k, n, trans_a, trans_b, stride_a, stride_b, stride_c,
            );
        }
    }
    // CPU fallback: treat strides as contiguous if they match logical sizes; otherwise error for now.
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

fn matmul_cpu_ex(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, trans_a: bool, trans_b: bool) -> Vec<f32> {
    let (ma, ka) = if trans_a { (k, m) } else { (m, k) };
    let (kb, nb) = if trans_b { (n, k) } else { (k, n) };
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
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    pub fn is_available() -> bool {
        CudaContext::new(0).is_ok()
    }

    pub fn matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> CoreResult<Vec<f32>> {
        let total = m.checked_mul(n).ok_or_else(|| "size overflow".to_string())?; if total == 0 { return Ok(Vec::new()); } if (total as u64) > (u32::MAX as u64) { return Err("problem too large to launch".into()); }
        let ptx = compile_ptx(KERNEL_SRC).map_err(|e| format!("nvrtc: {:?}", e))?;
        let ctx = CudaContext::new(0).map_err(|e| format!("cuda: {:?}", e))?;
        let module = ctx.load_module(ptx).map_err(|e| format!("load_module: {:?}", e))?;
        let func = module.load_function("matmul_f32").map_err(|e| format!("load_function: {:?}", e))?;
        let stream = ctx.default_stream();
        let d_a = stream.memcpy_stod(a).map_err(|e| format!("htod a: {:?}", e))?;
        let d_b = stream.memcpy_stod(b).map_err(|e| format!("htod b: {:?}", e))?;
        let mut d_c = stream.alloc_zeros::<f32>(total).map_err(|e| format!("alloc c: {:?}", e))?;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        unsafe { stream.launch_builder(&func)
            .arg(&d_a)
            .arg(&d_b)
            .arg(&mut d_c)
            .arg(&(m as i32))
            .arg(&(k as i32))
            .arg(&(n as i32))
            .launch(cfg) }
            .map_err(|e| format!("launch: {:?}", e))?;
        Ok(stream.memcpy_dtov(&d_c).map_err(|e| format!("dtoh c: {:?}", e))?)
    }

    pub fn matmul_f32_ex(
        a: &[f32], b: &[f32], m: usize, k: usize, n: usize, trans_a: bool, trans_b: bool,
    ) -> CoreResult<Vec<f32>> {
        if !trans_a && !trans_b {
            return matmul_f32(a, b, m, k, n);
        }
        // Fallback to CPU for transpose cases to keep implementation simple for now.
        Ok(super::matmul_cpu_ex(a, b, m, k, n, trans_a, trans_b))
    }

    pub fn matmul_batched_f32(
        a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize, trans_a: bool, trans_b: bool,
    ) -> CoreResult<Vec<f32>> {
        // Simple loop over batches using single-matmul implementation.
        let mut out = vec![0.0f32; batch.saturating_mul(m.saturating_mul(n))];
        let a_each = m.saturating_mul(k);
        let b_each = k.saturating_mul(n);
        let c_each = m.saturating_mul(n);
        for t in 0..batch {
            let a_slice = &a[t * a_each..(t + 1) * a_each];
            let b_slice = &b[t * b_each..(t + 1) * b_each];
            let tmp = matmul_f32_ex(a_slice, b_slice, m, k, n, trans_a, trans_b)?;
            out[t * c_each..(t + 1) * c_each].copy_from_slice(&tmp);
        }
        Ok(out)
    }

    pub fn matmul_batched_f32_strided(
        a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize,
        trans_a: bool, trans_b: bool, stride_a: i64, stride_b: i64, stride_c: i64,
    ) -> CoreResult<Vec<f32>> {
        let need_a = (m as i64) * (k as i64);
        let need_b = (k as i64) * (n as i64);
        let need_c = (m as i64) * (n as i64);
        if stride_a != need_a || stride_b != need_b || stride_c != need_c {
            return Err("strided batched matmul: non-contiguous strides not yet supported".into());
        }
        matmul_batched_f32(a, b, batch, m, k, n, trans_a, trans_b)
    }

    const KERNEL_SRC: &str = r#"
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
    if (idx >= total) return;
    int row = idx / n;
    int col = idx % n;
    float acc = 0.0f;
    for (int p = 0; p < k; ++p) {
        acc += lhs[row * k + p] * rhs[p * n + col];
    }
    out[idx] = acc;
}
"#;
}

#[cfg(feature = "gpu-rocm")]
mod rocm {
    pub fn is_available() -> bool { false }
    // Placeholders to satisfy cfg branches if feature is enabled by user in the future.
    // Real ROCm implementation would mirror CUDA module.
    pub fn matmul_f32(_a:&[f32], _b:&[f32], _m:usize,_k:usize,_n:usize) -> Result<Vec<f32>, String> { Err("rocm not implemented".into()) }
    pub fn matmul_f32_ex(_a:&[f32], _b:&[f32], _m:usize,_k:usize,_n:usize,_ta:bool,_tb:bool) -> Result<Vec<f32>, String> { Err("rocm not implemented".into()) }
    pub fn matmul_batched_f32(_a:&[f32], _b:&[f32], _batch:usize,_m:usize,_k:usize,_n:usize,_ta:bool,_tb:bool) -> Result<Vec<f32>, String> { Err("rocm not implemented".into()) }
    pub fn matmul_batched_f32_strided(_a:&[f32], _b:&[f32], _batch:usize,_m:usize,_k:usize,_n:usize,_ta:bool,_tb:bool,_sa:i64,_sb:i64,_sc:i64) -> Result<Vec<f32>, String> { Err("rocm not implemented".into()) }
}





