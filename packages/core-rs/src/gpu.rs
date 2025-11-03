use crate::CoreResult;
#[cfg(feature = "gpu-cuda")]
use cudarc::driver::{CudaSlice, CudaStream};
#[cfg(feature = "gpu-cuda")]
use std::sync::Arc;

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

#[derive(Clone, Copy, Debug)]
pub enum SumPrecisionPolicy {
    Default,
    Float64,
    Kahan,
}

impl Default for SumPrecisionPolicy {
    fn default() -> Self {
        SumPrecisionPolicy::Default
    }
}

fn sum_f32_as_f64(values: &[f32]) -> f32 {
    values.iter().fold(0.0f64, |acc, &v| acc + v as f64) as f32
}

fn sum_f32_kahan(values: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut c = 0.0f32;
    for &value in values {
        let y = value - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

#[derive(Clone, Copy, Debug)]
pub enum MatmulTensorCorePolicy {
    /// Favors precision by staying on the FP32 execution path (default)
    Accuracy,
    /// Favors throughput and enables TF32 Tensor Core on supported hardware (cuBLASLt)
    Performance,
    /// Forces FP16 Tensor Core usage (host converts inputs before execution)
    Float16,
    /// Forces BF16 Tensor Core usage (host converts inputs before execution)
    BFloat16,
}

impl Default for MatmulTensorCorePolicy {
    fn default() -> Self {
        MatmulTensorCorePolicy::Accuracy
    }
}

pub fn reduce_sum_f32(values: &[f32]) -> CoreResult<f32> {
    reduce_sum_f32_with_policy(values, SumPrecisionPolicy::Default)
}

pub fn reduce_sum_f32_with_policy(values: &[f32], policy: SumPrecisionPolicy) -> CoreResult<f32> {
    #[cfg(feature = "gpu-cuda")]
    {
        if cuda::is_available() {
            return cuda::reduce_sum_f32_with_policy(values, policy);
        }
    }
    #[cfg(feature = "gpu-rocm")]
    {
        if rocm::is_available() {
            match rocm::reduce_sum_f32_with_policy(values, policy) {
                Ok(total) => return Ok(total),
                Err(err) => {
                    if cfg!(debug_assertions) {
                        eprintln!("[num-rs][rocm] reduce_sum fallback: {err}");
                    }
                }
            }
        }
    }
    let result = match policy {
        SumPrecisionPolicy::Default => values.iter().copied().sum(),
        SumPrecisionPolicy::Float64 => sum_f32_as_f64(values),
        SumPrecisionPolicy::Kahan => sum_f32_kahan(values),
    };
    Ok(result)
}

#[cfg(feature = "gpu-cuda")]
pub fn reduce_sum_f32_device(
    stream: Arc<CudaStream>,
    values: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
) -> CoreResult<()> {
    cuda::reduce_sum_f32_device(stream, values, out)
}

#[cfg(feature = "gpu-cuda")]
pub fn reduce_sum_f32_device_with_policy(
    stream: Arc<CudaStream>,
    values: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    policy: SumPrecisionPolicy,
) -> CoreResult<()> {
    if out.len() != 1 {
        return Err("reduce_sum_f32_device: output slice must have length 1".into());
    }
    cuda::reduce_sum_f32_device_with_policy(stream, values, out, policy)
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

#[cfg(feature = "gpu-cuda")]
pub fn reduce_max_f32_device(
    stream: Arc<CudaStream>,
    values: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
) -> CoreResult<()> {
    cuda::reduce_max_f32_device(stream, values, out)
}

#[cfg(feature = "gpu-cuda")]
pub fn reduce_max_f32_device_with_policy(
    stream: Arc<CudaStream>,
    values: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    policy: NanPolicy,
) -> CoreResult<()> {
    if out.len() != 1 {
        return Err("reduce_max_f32_device: output slice must have length 1".into());
    }
    cuda::reduce_max_f32_device_with_policy(stream, values, out, policy)
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

#[cfg(feature = "gpu-cuda")]
pub fn argmax_f32_device(
    stream: Arc<CudaStream>,
    values: &CudaSlice<f32>,
    out: &mut CudaSlice<i32>,
) -> CoreResult<()> {
    cuda::argmax_f32_device(stream, values, out)
}

#[cfg(feature = "gpu-cuda")]
pub fn argmax_f32_device_with_policy(
    stream: Arc<CudaStream>,
    values: &CudaSlice<f32>,
    out: &mut CudaSlice<i32>,
    policy: NanPolicy,
) -> CoreResult<()> {
    if out.len() != 1 {
        return Err("argmax_f32_device: output slice must have length 1".into());
    }
    cuda::argmax_f32_device_with_policy(stream, values, out, policy)
}

pub fn matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> CoreResult<Vec<f32>> {
    matmul_f32_with_policy(a, b, m, k, n, MatmulTensorCorePolicy::Accuracy)
}

pub fn matmul_f32_with_policy(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    policy: MatmulTensorCorePolicy,
) -> CoreResult<Vec<f32>> {
    if a.len() != m.saturating_mul(k) || b.len() != k.saturating_mul(n) {
        return Err("matmul_f32: shape mismatch".into());
    }
    #[cfg(feature = "gpu-cuda")]
    {
        if cuda::is_available() {
            return cuda::matmul_f32_with_policy(a, b, m, k, n, policy);
        }
    }
    if !matches!(policy, MatmulTensorCorePolicy::Accuracy) {
        return Err("matmul_f32_with_policy: non-Accuracy policies require CUDA backend".into());
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
    matmul_f32_ex_with_policy(
        a,
        b,
        m,
        k,
        n,
        trans_a,
        trans_b,
        MatmulTensorCorePolicy::Accuracy,
    )
}

pub fn matmul_f32_ex_with_policy(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
    policy: MatmulTensorCorePolicy,
) -> CoreResult<Vec<f32>> {
    #[cfg(feature = "gpu-cuda")]
    {
        if cuda::is_available() {
            return cuda::matmul_f32_ex_with_policy(a, b, m, k, n, trans_a, trans_b, policy);
        }
    }
    #[cfg(feature = "gpu-rocm")]
    {
        if rocm::is_available() {
            match rocm::matmul_f32_ex_with_policy(a, b, m, k, n, trans_a, trans_b, policy) {
                Ok(result) => return Ok(result),
                Err(err) => {
                    if cfg!(debug_assertions) {
                        eprintln!("[num-rs][rocm] matmul_ex fallback: {err}");
                    }
                }
            }
        }
    }
    if !matches!(policy, MatmulTensorCorePolicy::Accuracy) {
        return Err("matmul_f32_ex_with_policy: non-Accuracy policies require CUDA backend".into());
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
    matmul_batched_f32_with_policy(
        a,
        b,
        batch,
        m,
        k,
        n,
        trans_a,
        trans_b,
        MatmulTensorCorePolicy::Accuracy,
    )
}

pub fn matmul_batched_f32_with_policy(
    a: &[f32],
    b: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
    policy: MatmulTensorCorePolicy,
) -> CoreResult<Vec<f32>> {
    #[cfg(feature = "gpu-cuda")]
    {
        if cuda::is_available() {
            return cuda::matmul_batched_f32_with_policy(
                a, b, batch, m, k, n, trans_a, trans_b, policy,
            );
        }
    }
    #[cfg(feature = "gpu-rocm")]
    {
        if rocm::is_available() && matches!(policy, MatmulTensorCorePolicy::Accuracy) {
            match rocm::matmul_batched_f32(a, b, batch, m, k, n, trans_a, trans_b) {
                Ok(result) => return Ok(result),
                Err(err) => {
                    if cfg!(debug_assertions) {
                        eprintln!("[num-rs][rocm] matmul_batched fallback: {err}");
                    }
                }
            }
        }
    }
    if !matches!(policy, MatmulTensorCorePolicy::Accuracy) {
        if cfg!(debug_assertions) {
            eprintln!(
                "[num-rs] matmul_batched_f32_with_policy policy {:?} falling back to CPU Accuracy implementation",
                policy
            );
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
    matmul_batched_f32_strided_with_policy(
        a,
        b,
        batch,
        m,
        k,
        n,
        trans_a,
        trans_b,
        stride_a,
        stride_b,
        stride_c,
        MatmulTensorCorePolicy::Accuracy,
    )
}

pub fn matmul_batched_f32_strided_with_policy(
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
    policy: MatmulTensorCorePolicy,
) -> CoreResult<Vec<f32>> {
    #[cfg(feature = "gpu-cuda")]
    {
        if cuda::is_available() {
            return cuda::matmul_batched_f32_strided_with_policy(
                a, b, batch, m, k, n, trans_a, trans_b, stride_a, stride_b, stride_c, policy,
            );
        }
    }
    #[cfg(feature = "gpu-rocm")]
    {
        if rocm::is_available() && matches!(policy, MatmulTensorCorePolicy::Accuracy) {
            match rocm::matmul_batched_f32_strided(
                a, b, batch, m, k, n, trans_a, trans_b, stride_a, stride_b, stride_c,
            ) {
                Ok(result) => return Ok(result),
                Err(err) => {
                    if cfg!(debug_assertions) {
                        eprintln!("[num-rs][rocm] matmul_batched_strided fallback: {err}");
                    }
                }
            }
        }
    }
    if !matches!(policy, MatmulTensorCorePolicy::Accuracy) {
        if cfg!(debug_assertions) {
            eprintln!(
                "[num-rs] matmul_batched_f32_strided_with_policy policy {:?} falling back to CPU Accuracy implementation",
                policy
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
    use crate::gpu::{MatmulTensorCorePolicy, NanPolicy, SumPrecisionPolicy};
    use cudarc::cublas::sys::cublasOperation_t;
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
    use cudarc::cublaslt::safe::{CudaBlasLT, Matmul, MatmulConfig};
    use cudarc::driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
        LaunchConfig, PushKernelArg,
    };
    use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
    use half::{bf16, f16};
    use std::{
        collections::{HashMap, HashSet},
        env, fs,
        path::PathBuf,
        sync::{Arc, Mutex, OnceLock},
    };

    static GLOBAL_CONTEXT: OnceLock<Result<Arc<CudaContext>, String>> = OnceLock::new();
    static MODULE_CACHE: OnceLock<Mutex<HashMap<usize, HashMap<&'static str, Arc<CudaModule>>>>> =
        OnceLock::new();

    fn write_scalar_f32(
        stream: &Arc<CudaStream>,
        out: &mut CudaSlice<f32>,
        value: f32,
    ) -> Result<(), String> {
        if out.len() == 0 {
            return Err("write_scalar_f32: output slice is empty".into());
        }
        let host = [value];
        stream
            .memcpy_htod(&host, out)
            .map_err(|e| format!("memcpy_htod scalar f32: {:?}", e))
    }

    fn write_scalar_i32(
        stream: &Arc<CudaStream>,
        out: &mut CudaSlice<i32>,
        value: i32,
    ) -> Result<(), String> {
        if out.len() == 0 {
            return Err("write_scalar_i32: output slice is empty".into());
        }
        let host = [value];
        stream
            .memcpy_htod(&host, out)
            .map_err(|e| format!("memcpy_htod scalar i32: {:?}", e))
    }

    fn extract_optional_index(idx: Option<i32>, len: usize) -> Option<usize> {
        match idx.unwrap_or(len as i32) {
            v if v >= 0 && (v as usize) < len => Some(v as usize),
            _ => None,
        }
    }

    fn first_index_with_stage(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        values: &CudaSlice<f32>,
        stage1_name: &str,
        label: &str,
    ) -> Result<Option<usize>, String> {
        if values.len() == 0 {
            return Ok(None);
        }
        let stage1 = module
            .load_function(stage1_name)
            .map_err(|e| format!("load {stage1_name}: {:?}", e))?;
        let reduce = module
            .load_function("min_index_stage_reduce")
            .map_err(|e| format!("load min_index_stage_reduce: {:?}", e))?;
        let block_dim_stage1 = select_block_dim(&stage1, smem_per_thread_i32)?;
        let shared_stage1 = shared_mem_bytes(block_dim_stage1, std::mem::size_of::<i32>())?;
        let mut blocks = blocks_for_len("index reduce stage1", values.len(), block_dim_stage1)?;
        let mut current = stream
            .alloc_zeros::<i32>(blocks as usize)
            .map_err(|e| format!("alloc index tmp: {:?}", e))?;
        let len_i32 = usize_to_i32("index reduce length", values.len())?;
        let cfg_stage1 = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block_dim_stage1, 1, 1),
            shared_mem_bytes: shared_stage1,
        };
        unsafe {
            stream
                .launch_builder(&stage1)
                .arg(values)
                .arg(&mut current)
                .arg(&len_i32)
                .launch(cfg_stage1)
        }
        .map_err(|e| format!("launch {stage1_name}: {:?}", e))?;
        if blocks == 1 {
            let host = stream
                .memcpy_dtov(&current)
                .map_err(|e| format!("{label}: {:?}", e))?;
            return Ok(extract_optional_index(
                host.into_iter().next(),
                values.len(),
            ));
        }
        let block_dim_reduce = select_block_dim(&reduce, smem_per_thread_i32)?;
        let shared_reduce = shared_mem_bytes(block_dim_reduce, std::mem::size_of::<i32>())?;
        let mut current_len = blocks as usize;
        loop {
            blocks = blocks_for_len("index reduce partial", current_len, block_dim_reduce)?;
            let mut next = stream
                .alloc_zeros::<i32>(blocks as usize)
                .map_err(|e| format!("alloc index next: {:?}", e))?;
            let len_i32 = usize_to_i32("index reduce partial length", current_len)?;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (block_dim_reduce, 1, 1),
                shared_mem_bytes: shared_reduce,
            };
            unsafe {
                stream
                    .launch_builder(&reduce)
                    .arg(&current)
                    .arg(&mut next)
                    .arg(&len_i32)
                    .launch(cfg)
            }
            .map_err(|e| format!("launch min_index_stage_reduce: {:?}", e))?;
            if blocks == 1 {
                let host = stream
                    .memcpy_dtov(&next)
                    .map_err(|e| format!("{label}: {:?}", e))?;
                return Ok(extract_optional_index(
                    host.into_iter().next(),
                    values.len(),
                ));
            }
            current_len = blocks as usize;
            current = next;
        }
    }

    fn first_nan_index(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        values: &CudaSlice<f32>,
    ) -> Result<Option<usize>, String> {
        first_index_with_stage(
            stream,
            module,
            values,
            "first_nan_index_stage1",
            "dtoh first_nan index",
        )
    }

    fn first_valid_index(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        values: &CudaSlice<f32>,
    ) -> Result<Option<usize>, String> {
        first_index_with_stage(
            stream,
            module,
            values,
            "first_valid_index_stage1",
            "dtoh first_valid index",
        )
    }

    fn reduce_sum_device_internal(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        values: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), String> {
        let func = module
            .load_function("reduce_sum_stage1")
            .map_err(|e| format!("load reduce_sum_stage1: {:?}", e))?;
        let block_dim = select_block_dim(&func, smem_per_thread_f32)?;
        let shared = shared_mem_bytes(block_dim, std::mem::size_of::<f32>())?;
        let mut current_len = values.len();
        let mut storage: Option<CudaSlice<f32>> = None;
        loop {
            let current_slice = storage.as_ref().unwrap_or(values);
            let len_i32 = usize_to_i32("reduce_sum length", current_len)?;
            let blocks = blocks_for_len("reduce_sum length", current_len, block_dim)?;
            let mut next = stream
                .alloc_zeros::<f32>(blocks as usize)
                .map_err(|e| format!("alloc reduce_sum tmp: {:?}", e))?;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: shared,
            };
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(current_slice)
                    .arg(&mut next)
                    .arg(&len_i32)
                    .launch(cfg)
            }
            .map_err(|e| format!("launch reduce_sum: {:?}", e))?;
            if blocks == 1 {
                stream
                    .memcpy_dtod(&next, out)
                    .map_err(|e| format!("dtod reduce_sum result: {:?}", e))?;
                break;
            }
            current_len = blocks as usize;
            storage = Some(next);
        }
        Ok(())
    }

    fn reduce_sum_device_internal_f64(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        values: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), String> {
        let func_from_f32 = module
            .load_function("reduce_sum_stage1_f64_from_f32")
            .map_err(|e| format!("load reduce_sum_stage1_f64_from_f32: {:?}", e))?;
        let func_f64 = module
            .load_function("reduce_sum_stage1_f64")
            .map_err(|e| format!("load reduce_sum_stage1_f64: {:?}", e))?;
        let block_dim_f32 = select_block_dim(&func_from_f32, smem_per_thread_f64)?;
        let shared_f32 = shared_mem_bytes(block_dim_f32, std::mem::size_of::<f64>())?;
        let block_dim_f64 = select_block_dim(&func_f64, smem_per_thread_f64)?;
        let shared_f64 = shared_mem_bytes(block_dim_f64, std::mem::size_of::<f64>())?;

        let mut current_len = values.len();
        let mut storage: Option<CudaSlice<f64>> = None;
        let mut use_f32_kernel = true;

        loop {
            let (func, block_dim, shared_bytes) = if use_f32_kernel {
                (&func_from_f32, block_dim_f32, shared_f32)
            } else {
                (&func_f64, block_dim_f64, shared_f64)
            };

            let len_i32 = usize_to_i32("reduce_sum_f64 length", current_len)?;
            let blocks = blocks_for_len("reduce_sum_f64 length", current_len, block_dim)?;
            let mut next = stream
                .alloc_zeros::<f64>(blocks as usize)
                .map_err(|e| format!("alloc reduce_sum_f64 tmp: {:?}", e))?;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: shared_bytes,
            };
            unsafe {
                let mut launch = stream.launch_builder(func);
                if use_f32_kernel {
                    launch.arg(values).arg(&mut next).arg(&len_i32).launch(cfg)
                } else {
                    let current_slice = storage
                        .as_ref()
                        .expect("double storage should exist before second pass");
                    launch
                        .arg(current_slice)
                        .arg(&mut next)
                        .arg(&len_i32)
                        .launch(cfg)
                }
            }
            .map_err(|e| format!("launch reduce_sum_f64: {:?}", e))?;

            if blocks == 1 {
                let host = stream
                    .memcpy_dtov(&next)
                    .map_err(|e| format!("dtoh reduce_sum_f64: {:?}", e))?;
                let value = host.into_iter().next().unwrap_or(0.0) as f32;
                write_scalar_f32(stream, out, value)?;
                break;
            }

            storage = Some(next);
            current_len = blocks as usize;
            use_f32_kernel = false;
        }
        Ok(())
    }

    fn reduce_sum_device_internal_kahan(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        values: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), String> {
        let func_from_f32 = module
            .load_function("reduce_sum_stage1_kahan_from_f32")
            .map_err(|e| format!("load reduce_sum_stage1_kahan_from_f32: {:?}", e))?;
        let func_f64 = module
            .load_function("reduce_sum_stage1_kahan")
            .map_err(|e| format!("load reduce_sum_stage1_kahan: {:?}", e))?;
        let block_dim_f32 = select_block_dim(&func_from_f32, smem_per_thread_2xf64)?;
        let shared_f32 = shared_mem_bytes(block_dim_f32, 2 * std::mem::size_of::<f64>())?;
        let block_dim_f64 = select_block_dim(&func_f64, smem_per_thread_2xf64)?;
        let shared_f64 = shared_mem_bytes(block_dim_f64, 2 * std::mem::size_of::<f64>())?;

        let mut current_len = values.len();
        let mut storage: Option<CudaSlice<f64>> = None;
        let mut use_f32_kernel = true;

        loop {
            let (func, block_dim, shared_bytes) = if use_f32_kernel {
                (&func_from_f32, block_dim_f32, shared_f32)
            } else {
                (&func_f64, block_dim_f64, shared_f64)
            };
            let len_i32 = usize_to_i32("reduce_sum_kahan length", current_len)?;
            let blocks = blocks_for_len("reduce_sum_kahan length", current_len, block_dim)?;
            let mut next = stream
                .alloc_zeros::<f64>(blocks as usize)
                .map_err(|e| format!("alloc reduce_sum_kahan tmp: {:?}", e))?;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: shared_bytes,
            };
            unsafe {
                let mut launch = stream.launch_builder(func);
                if use_f32_kernel {
                    launch.arg(values).arg(&mut next).arg(&len_i32).launch(cfg)
                } else {
                    let current_slice = storage
                        .as_ref()
                        .expect("double storage should exist before second pass");
                    launch
                        .arg(current_slice)
                        .arg(&mut next)
                        .arg(&len_i32)
                        .launch(cfg)
                }
            }
            .map_err(|e| format!("launch reduce_sum_kahan: {:?}", e))?;

            if blocks == 1 {
                let host = stream
                    .memcpy_dtov(&next)
                    .map_err(|e| format!("dtoh reduce_sum_kahan: {:?}", e))?;
                let value = host.into_iter().next().unwrap_or(0.0) as f32;
                write_scalar_f32(stream, out, value)?;
                break;
            }

            storage = Some(next);
            current_len = blocks as usize;
            use_f32_kernel = false;
        }
        Ok(())
    }

    fn reduce_max_device_ignore_nan(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        values: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), String> {
        let func = module
            .load_function("reduce_max_stage1_ignore_nan")
            .map_err(|e| format!("load reduce_max_stage1_ignore_nan: {:?}", e))?;
        let block_dim = select_block_dim(&func, smem_per_thread_f32)?;
        let shared = shared_mem_bytes(block_dim, std::mem::size_of::<f32>())?;
        let mut current_len = values.len();
        let mut storage: Option<CudaSlice<f32>> = None;
        loop {
            let current_slice = storage.as_ref().unwrap_or(values);
            let len_i32 = usize_to_i32("reduce_max length", current_len)?;
            let blocks = blocks_for_len("reduce_max length", current_len, block_dim)?;
            let mut next = stream
                .alloc_zeros::<f32>(blocks as usize)
                .map_err(|e| format!("alloc reduce_max tmp: {:?}", e))?;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: shared,
            };
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(current_slice)
                    .arg(&mut next)
                    .arg(&len_i32)
                    .launch(cfg)
            }
            .map_err(|e| format!("launch reduce_max: {:?}", e))?;
            if blocks == 1 {
                stream
                    .memcpy_dtod(&next, out)
                    .map_err(|e| format!("dtod reduce_max result: {:?}", e))?;
                break;
            }
            current_len = blocks as usize;
            storage = Some(next);
        }
        Ok(())
    }

    fn argmax_device_ignore_nan(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        values: &CudaSlice<f32>,
        out: &mut CudaSlice<i32>,
    ) -> Result<(), String> {
        let stage1 = module
            .load_function("argmax_stage1_ignore_nan")
            .map_err(|e| format!("load argmax_stage1_ignore_nan: {:?}", e))?;
        let block_dim_stage1 = select_block_dim(&stage1, smem_per_thread_f32_i32)?;
        let shared_stage1 = shared_mem_bytes(
            block_dim_stage1,
            std::mem::size_of::<f32>() + std::mem::size_of::<i32>(),
        )?;
        let len_i32 = usize_to_i32("argmax length", values.len())?;
        let mut blocks = blocks_for_len("argmax length", values.len(), block_dim_stage1)?;
        let mut vals = stream
            .alloc_zeros::<f32>(blocks as usize)
            .map_err(|e| format!("alloc argmax partial values: {:?}", e))?;
        let mut idxs = stream
            .alloc_zeros::<i32>(blocks as usize)
            .map_err(|e| format!("alloc argmax partial idx: {:?}", e))?;
        let cfg_stage1 = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block_dim_stage1, 1, 1),
            shared_mem_bytes: shared_stage1,
        };
        unsafe {
            stream
                .launch_builder(&stage1)
                .arg(values)
                .arg(&mut vals)
                .arg(&mut idxs)
                .arg(&len_i32)
                .launch(cfg_stage1)
        }
        .map_err(|e| format!("launch argmax_stage1_ignore_nan: {:?}", e))?;
        if blocks == 1 {
            stream
                .memcpy_dtod(&idxs, out)
                .map_err(|e| format!("dtod argmax result: {:?}", e))?;
            return Ok(());
        }
        let reduce = module
            .load_function("argmax_stage_reduce")
            .map_err(|e| format!("load argmax_stage_reduce: {:?}", e))?;
        let block_dim_reduce = select_block_dim(&reduce, smem_per_thread_f32_i32)?;
        let shared_reduce = shared_mem_bytes(
            block_dim_reduce,
            std::mem::size_of::<f32>() + std::mem::size_of::<i32>(),
        )?;
        let mut current_len = blocks as usize;
        loop {
            blocks = blocks_for_len("argmax partial length", current_len, block_dim_reduce)?;
            let mut next_vals = stream
                .alloc_zeros::<f32>(blocks as usize)
                .map_err(|e| format!("alloc argmax next values: {:?}", e))?;
            let mut next_idxs = stream
                .alloc_zeros::<i32>(blocks as usize)
                .map_err(|e| format!("alloc argmax next idx: {:?}", e))?;
            let len_i32 = usize_to_i32("argmax partial length", current_len)?;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (block_dim_reduce, 1, 1),
                shared_mem_bytes: shared_reduce,
            };
            unsafe {
                stream
                    .launch_builder(&reduce)
                    .arg(&vals)
                    .arg(&idxs)
                    .arg(&mut next_vals)
                    .arg(&mut next_idxs)
                    .arg(&len_i32)
                    .launch(cfg)
            }
            .map_err(|e| format!("launch argmax_stage_reduce: {:?}", e))?;
            if blocks == 1 {
                stream
                    .memcpy_dtod(&next_idxs, out)
                    .map_err(|e| format!("dtod argmax result: {:?}", e))?;
                break;
            }
            current_len = blocks as usize;
            vals = next_vals;
            idxs = next_idxs;
        }
        Ok(())
    }

    const REDUCE_KERNEL_SRC: &str = r#"
#include <math.h>

extern "C" __global__ void reduce_sum_stage1(const float* __restrict__ input,
                                             float* __restrict__ partial,
                                             int length) {
    extern __shared__ float sdata[];
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

extern "C" __global__ void reduce_sum_stage1_f64_from_f32(const float* __restrict__ input,
                                                          double* __restrict__ partial,
                                                          int length) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int global = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    double acc = 0.0;
    for (int idx = global; idx < length; idx += stride) {
        acc += double(input[idx]);
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

extern "C" __global__ void reduce_sum_stage1_f64(const double* __restrict__ input,
                                                 double* __restrict__ partial,
                                                 int length) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int global = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    double acc = 0.0;
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

__device__ __forceinline__ void kahan_add(double value, double& sum, double& comp) {
    double y = value - comp;
    double t = sum + y;
    comp = (t - sum) - y;
    sum = t;
}

extern "C" __global__ void reduce_sum_stage1_kahan_from_f32(const float* __restrict__ input,
                                                            double* __restrict__ partial,
                                                            int length) {
    extern __shared__ double shared[];
    double* ssum = shared;
    double* scomp = shared + blockDim.x;

    int tid = threadIdx.x;
    int global = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    double sum = 0.0;
    double comp = 0.0;
    for (int idx = global; idx < length; idx += stride) {
        double v = double(input[idx]);
        kahan_add(v, sum, comp);
    }
    ssum[tid] = sum;
    scomp[tid] = comp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            double peer = ssum[tid + s] + scomp[tid + s];
            kahan_add(peer, ssum[tid], scomp[tid]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial[blockIdx.x] = ssum[0] + scomp[0];
    }
}

extern "C" __global__ void reduce_sum_stage1_kahan(const double* __restrict__ input,
                                                   double* __restrict__ partial,
                                                   int length) {
    extern __shared__ double shared[];
    double* ssum = shared;
    double* scomp = shared + blockDim.x;

    int tid = threadIdx.x;
    int global = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    double sum = 0.0;
    double comp = 0.0;
    for (int idx = global; idx < length; idx += stride) {
        kahan_add(input[idx], sum, comp);
    }
    ssum[tid] = sum;
    scomp[tid] = comp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            double peer = ssum[tid + s] + scomp[tid + s];
            kahan_add(peer, ssum[tid], scomp[tid]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial[blockIdx.x] = ssum[0] + scomp[0];
    }
}

extern "C" __global__ void reduce_max_stage1_ignore_nan(const float* __restrict__ input,
                                                        float* __restrict__ partial,
                                                        int length) {
    extern __shared__ float smax[];
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
    extern __shared__ unsigned char smem[];
    float* smax = reinterpret_cast<float*>(smem);
    int* sarg = reinterpret_cast<int*>(smem + blockDim.x * sizeof(float));
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

extern "C" __global__ void argmax_stage_reduce(const float* __restrict__ values_in,
                                               const int* __restrict__ idx_in,
                                               float* __restrict__ values_out,
                                               int* __restrict__ idx_out,
                                               int length) {
    extern __shared__ unsigned char smem[];
    float* smax = reinterpret_cast<float*>(smem);
    int* sidx = reinterpret_cast<int*>(smem + blockDim.x * sizeof(float));
    int tid = threadIdx.x;
    int global = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    float best_val = -INFINITY;
    int best_idx = -1;
    for (int idx = global; idx < length; idx += stride) {
        float v = values_in[idx];
        int i = idx_in[idx];
        if (i >= 0) {
            if (best_idx < 0 || v > best_val || (v == best_val && i < best_idx)) {
                best_val = v;
                best_idx = i;
            }
        }
    }
    smax[tid] = best_val;
    sidx[tid] = best_idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = smax[tid + s];
            int oi = sidx[tid + s];
            float cv = smax[tid];
            int ci = sidx[tid];
            if (oi >= 0 && (ci < 0 || ov > cv || (ov == cv && oi < ci))) {
                smax[tid] = ov;
                sidx[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        values_out[blockIdx.x] = smax[0];
        idx_out[blockIdx.x] = sidx[0];
    }
}

extern "C" __global__ void first_nan_index_stage1(const float* __restrict__ input,
                                                  int* __restrict__ pmin,
                                                  int length) {
    extern __shared__ int smin[];
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

extern "C" __global__ void min_index_stage_reduce(const int* __restrict__ input,
                                                  int* __restrict__ partial,
                                                  int length) {
    extern __shared__ int smin[];
    int tid = threadIdx.x;
    int global = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    int best = length;
    for (int idx = global; idx < length; idx += stride) {
        int v = input[idx];
        if (v < best) {
            best = v;
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
        partial[blockIdx.x] = smin[0];
    }
}
"#;

    const MATMUL_KERNEL_SRC: &str = r#"
#include <cuda_runtime.h>
#include <stdint.h>
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

__device__ __forceinline__ float load_a(const float* A,
                                        int row,
                                        int col,
                                        int m,
                                        int k,
                                        int lda,
                                        bool trans_a) {
    if (trans_a) {
        if (col < k && row < m) {
            return A[col * lda + row];
        }
    } else {
        if (row < m && col < k) {
            return A[row * lda + col];
        }
    }
    return 0.0f;
}

__device__ __forceinline__ float load_b(const float* B,
                                        int row,
                                        int col,
                                        int k,
                                        int n,
                                        int ldb,
                                        bool trans_b) {
    if (trans_b) {
        if (col < n && row < k) {
            return B[col * ldb + row];
        }
    } else {
        if (row < k && col < n) {
            return B[row * ldb + col];
        }
    }
    return 0.0f;
}

extern "C" __global__ void matmul_tiled(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int m,
                                        int k,
                                        int n,
                                        int lda,
                                        int ldb,
                                        int ldc,
                                        int trans_a,
                                        int trans_b) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;
    float acc = 0.0f;
    int tiles = (k + TILE_K - 1) / TILE_K;
    for (int t = 0; t < tiles; ++t) {
        int a_col = t * TILE_K + threadIdx.x;
        int b_row = t * TILE_K + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            load_a(A, row, a_col, m, k, lda, trans_a != 0);
        Bs[threadIdx.y][threadIdx.x] =
            load_b(B, b_row, col, k, n, ldb, trans_b != 0);
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < m && col < n) {
        C[row * ldc + col] = acc;
    }
}

extern "C" __global__ void matmul_tiled_strided(const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float* __restrict__ C,
                                                int m,
                                                int k,
                                                int n,
                                                int lda,
                                                int ldb,
                                                int ldc,
                                                long long stride_a,
                                                long long stride_b,
                                                long long stride_c,
                                                int trans_a,
                                                int trans_b,
                                                int batch) {
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch) {
        return;
    }
    const float* Abase = A + batch_idx * stride_a;
    const float* Bbase = B + batch_idx * stride_b;
    float* Cbase = C + batch_idx * stride_c;
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;
    float acc = 0.0f;
    int tiles = (k + TILE_K - 1) / TILE_K;
    for (int t = 0; t < tiles; ++t) {
        int a_col = t * TILE_K + threadIdx.x;
        int b_row = t * TILE_K + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            load_a(Abase, row, a_col, m, k, lda, trans_a != 0);
        Bs[threadIdx.y][threadIdx.x] =
            load_b(Bbase, b_row, col, k, n, ldb, trans_b != 0);
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < m && col < n) {
        Cbase[row * ldc + col] = acc;
    }
}

extern "C" __global__ void matmul_tiled_64(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int m,
                                           int k,
                                           int n,
                                           int lda,
                                           int ldb,
                                           int ldc,
                                           int trans_a,
                                           int trans_b) {
    const int TILE_M = 64;
    const int TILE_N = 64;
    const int TILE_K = 16;
    const int VEC = 4;
    const int THREADS_X = TILE_N / VEC;
    const int THREADS_Y = TILE_M / VEC;
    if (threadIdx.x >= THREADS_X || threadIdx.y >= THREADS_Y) {
        return;
    }
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];
    const int lane_x = threadIdx.x;
    const int lane_y = threadIdx.y;
    int row_idx[VEC];
    int col_idx[VEC];
#pragma unroll
    for (int v = 0; v < VEC; ++v) {
        row_idx[v] = lane_y + v * THREADS_Y;
        col_idx[v] = lane_x * VEC + v;
    }
    float acc[VEC][VEC];
#pragma unroll
    for (int r = 0; r < VEC; ++r) {
#pragma unroll
        for (int c = 0; c < VEC; ++c) {
            acc[r][c] = 0.0f;
        }
    }
    int tiles = (k + TILE_K - 1) / TILE_K;
    for (int tile = 0; tile < tiles; ++tile) {
        int k_col = tile * TILE_K + lane_x;
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
            int row_local = row_idx[v];
            int global_row = blockIdx.y * TILE_M + row_local;
            float val = 0.0f;
            if (global_row < m && k_col < k) {
                if (trans_a) {
                    val = A[k_col * lda + global_row];
                } else {
                    val = A[global_row * lda + k_col];
                }
            }
            As[row_local][lane_x] = val;
        }
        int k_row = lane_y;
        int global_k = tile * TILE_K + k_row;
        int col_base = lane_x * VEC;
        int global_col_base = blockIdx.x * TILE_N + col_base;
        float vals[VEC];
        if (global_k < k) {
            int can_vector = 0;
            if (!trans_b) {
                if (global_col_base + VEC - 1 < n) {
                    can_vector = 1;
                }
            } else {
                if (global_k + VEC - 1 < k) {
                    can_vector = 1;
                }
            }
            if (can_vector) {
                if (!trans_b) {
                    const float* ptr = B + global_k * ldb + global_col_base;
                    if ((((uintptr_t)ptr) & 15) == 0) {
                        float4 vec = *reinterpret_cast<const float4*>(ptr);
                        vals[0] = vec.x;
                        vals[1] = vec.y;
                        vals[2] = vec.z;
                        vals[3] = vec.w;
                    } else {
                        vals[0] = ptr[0];
                        vals[1] = ptr[1];
                        vals[2] = ptr[2];
                        vals[3] = ptr[3];
                    }
                } else {
                    const float* ptr = B + global_col_base * ldb + global_k;
                    if ((((uintptr_t)ptr) & 15) == 0) {
                        float4 vec = *reinterpret_cast<const float4*>(ptr);
                        vals[0] = vec.x;
                        vals[1] = vec.y;
                        vals[2] = vec.z;
                        vals[3] = vec.w;
                    } else {
                        vals[0] = ptr[0];
                        vals[1] = ptr[1];
                        vals[2] = ptr[2];
                        vals[3] = ptr[3];
                    }
                }
            } else {
#pragma unroll
                for (int v = 0; v < VEC; ++v) {
                    int global_col = global_col_base + v;
                    vals[v] = load_b(B, global_k, global_col, k, n, ldb, trans_b != 0);
                }
            }
        } else {
#pragma unroll
            for (int v = 0; v < VEC; ++v) {
                vals[v] = 0.0f;
            }
        }
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
            int col_local = col_base + v;
            Bs[k_row][col_local] = vals[v];
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float b_vals[VEC];
#pragma unroll
            for (int v = 0; v < VEC; ++v) {
                b_vals[v] = Bs[kk][col_idx[v]];
            }
#pragma unroll
            for (int r = 0; r < VEC; ++r) {
                float a_val = As[row_idx[r]][kk];
#pragma unroll
                for (int c = 0; c < VEC; ++c) {
                    acc[r][c] += a_val * b_vals[c];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < VEC; ++r) {
        int row_local = row_idx[r];
        int global_row = blockIdx.y * TILE_M + row_local;
        if (global_row < m) {
#pragma unroll
            for (int c = 0; c < VEC; ++c) {
                int col_local = col_idx[c];
                int global_col = blockIdx.x * TILE_N + col_local;
                if (global_col < n) {
                    C[global_row * ldc + global_col] = acc[r][c];
                }
            }
        }
    }
}

extern "C" __global__ void matmul_tiled_64_strided(const float* __restrict__ A,
                                                   const float* __restrict__ B,
                                                   float* __restrict__ C,
                                                   int m,
                                                   int k,
                                                   int n,
                                                   int lda,
                                                   int ldb,
                                                   int ldc,
                                                   long long stride_a,
                                                   long long stride_b,
                                                   long long stride_c,
                                                   int trans_a,
                                                   int trans_b,
                                                   int batch) {
    const int TILE_M = 64;
    const int TILE_N = 64;
    const int TILE_K = 16;
    const int VEC = 4;
    const int THREADS_X = TILE_N / VEC;
    const int THREADS_Y = TILE_M / VEC;
    if (threadIdx.x >= THREADS_X || threadIdx.y >= THREADS_Y) {
        return;
    }
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch) {
        return;
    }
    const float* Abase = A + batch_idx * stride_a;
    const float* Bbase = B + batch_idx * stride_b;
    float* Cbase = C + batch_idx * stride_c;
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];
    const int lane_x = threadIdx.x;
    const int lane_y = threadIdx.y;
    int row_idx[VEC];
    int col_idx[VEC];
#pragma unroll
    for (int v = 0; v < VEC; ++v) {
        row_idx[v] = lane_y + v * THREADS_Y;
        col_idx[v] = lane_x * VEC + v;
    }
    float acc[VEC][VEC];
#pragma unroll
    for (int r = 0; r < VEC; ++r) {
#pragma unroll
        for (int c = 0; c < VEC; ++c) {
            acc[r][c] = 0.0f;
        }
    }
    int tiles = (k + TILE_K - 1) / TILE_K;
    for (int tile = 0; tile < tiles; ++tile) {
        int k_col = tile * TILE_K + lane_x;
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
            int row_local = row_idx[v];
            int global_row = blockIdx.y * TILE_M + row_local;
            float val = 0.0f;
            if (global_row < m && k_col < k) {
                if (trans_a) {
                    val = Abase[k_col * lda + global_row];
                } else {
                    val = Abase[global_row * lda + k_col];
                }
            }
            As[row_local][lane_x] = val;
        }
        int k_row = lane_y;
        int global_k = tile * TILE_K + k_row;
        int col_base = lane_x * VEC;
        int global_col_base = blockIdx.x * TILE_N + col_base;
        float vals[VEC];
        if (global_k < k) {
            int can_vector = 0;
            if (!trans_b) {
                if (global_col_base + VEC - 1 < n) {
                    can_vector = 1;
                }
            } else {
                if (global_k + VEC - 1 < k) {
                    can_vector = 1;
                }
            }
            if (can_vector) {
                if (!trans_b) {
                    const float* ptr = Bbase + global_k * ldb + global_col_base;
                    if ((((uintptr_t)ptr) & 15) == 0) {
                        float4 vec = *reinterpret_cast<const float4*>(ptr);
                        vals[0] = vec.x;
                        vals[1] = vec.y;
                        vals[2] = vec.z;
                        vals[3] = vec.w;
                    } else {
                        vals[0] = ptr[0];
                        vals[1] = ptr[1];
                        vals[2] = ptr[2];
                        vals[3] = ptr[3];
                    }
                } else {
                    const float* ptr = Bbase + global_col_base * ldb + global_k;
                    if ((((uintptr_t)ptr) & 15) == 0) {
                        float4 vec = *reinterpret_cast<const float4*>(ptr);
                        vals[0] = vec.x;
                        vals[1] = vec.y;
                        vals[2] = vec.z;
                        vals[3] = vec.w;
                    } else {
                        vals[0] = ptr[0];
                        vals[1] = ptr[1];
                        vals[2] = ptr[2];
                        vals[3] = ptr[3];
                    }
                }
            } else {
#pragma unroll
                for (int v = 0; v < VEC; ++v) {
                    int global_col = global_col_base + v;
                    vals[v] = load_b(Bbase, global_k, global_col, k, n, ldb, trans_b != 0);
                }
            }
        } else {
#pragma unroll
            for (int v = 0; v < VEC; ++v) {
                vals[v] = 0.0f;
            }
        }
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
            int col_local = col_base + v;
            Bs[k_row][col_local] = vals[v];
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float b_vals[VEC];
#pragma unroll
            for (int v = 0; v < VEC; ++v) {
                b_vals[v] = Bs[kk][col_idx[v]];
            }
#pragma unroll
            for (int r = 0; r < VEC; ++r) {
                float a_val = As[row_idx[r]][kk];
#pragma unroll
                for (int c = 0; c < VEC; ++c) {
                    acc[r][c] += a_val * b_vals[c];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < VEC; ++r) {
        int row_local = row_idx[r];
        int global_row = blockIdx.y * TILE_M + row_local;
        if (global_row < m) {
#pragma unroll
            for (int c = 0; c < VEC; ++c) {
                int col_local = col_idx[c];
                int global_col = blockIdx.x * TILE_N + col_local;
                if (global_col < n) {
                    Cbase[global_row * ldc + global_col] = acc[r][c];
                }
            }
        }
    }
}

extern "C" __global__ void matmul_tiled_128(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int m,
                                            int k,
                                            int n,
                                            int lda,
                                            int ldb,
                                            int ldc,
                                            int trans_a,
                                            int trans_b) {
    const int TILE_M = 128;
    const int TILE_N = 128;
    const int TILE_K = 32;
    const int VEC = 4;
    const int THREADS_X = TILE_N / VEC;
    const int THREADS_Y = TILE_M / VEC;
    if (threadIdx.x >= THREADS_X || threadIdx.y >= THREADS_Y) {
        return;
    }
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];
    const int lane_x = threadIdx.x;
    const int lane_y = threadIdx.y;
    int row_idx[VEC];
    int col_idx[VEC];
#pragma unroll
    for (int v = 0; v < VEC; ++v) {
        row_idx[v] = lane_y + v * THREADS_Y;
        col_idx[v] = lane_x * VEC + v;
    }
    float acc[VEC][VEC];
#pragma unroll
    for (int r = 0; r < VEC; ++r) {
#pragma unroll
        for (int c = 0; c < VEC; ++c) {
            acc[r][c] = 0.0f;
        }
    }
    int tiles = (k + TILE_K - 1) / TILE_K;
    for (int tile = 0; tile < tiles; ++tile) {
        int k_col = tile * TILE_K + lane_x;
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
            int row_local = row_idx[v];
            int global_row = blockIdx.y * TILE_M + row_local;
            float val = 0.0f;
            if (global_row < m && k_col < k) {
                if (trans_a) {
                    val = A[k_col * lda + global_row];
                } else {
                    val = A[global_row * lda + k_col];
                }
            }
            As[row_local][lane_x] = val;
        }
        int k_row = lane_y;
        int global_k = tile * TILE_K + k_row;
        int col_base = lane_x * VEC;
        int global_col_base = blockIdx.x * TILE_N + col_base;
        float vals[VEC];
        if (global_k < k) {
            int can_vector = 0;
            if (!trans_b) {
                if (global_col_base + VEC - 1 < n) {
                    can_vector = 1;
                }
            } else {
                if (global_k + VEC - 1 < k) {
                    can_vector = 1;
                }
            }
            if (can_vector) {
                if (!trans_b) {
                    const float* ptr = B + global_k * ldb + global_col_base;
                    if ((((uintptr_t)ptr) & 15) == 0) {
                        float4 vec = *reinterpret_cast<const float4*>(ptr);
                        vals[0] = vec.x;
                        vals[1] = vec.y;
                        vals[2] = vec.z;
                        vals[3] = vec.w;
                    } else {
                        vals[0] = ptr[0];
                        vals[1] = ptr[1];
                        vals[2] = ptr[2];
                        vals[3] = ptr[3];
                    }
                } else {
                    const float* ptr = B + global_col_base * ldb + global_k;
                    if ((((uintptr_t)ptr) & 15) == 0) {
                        float4 vec = *reinterpret_cast<const float4*>(ptr);
                        vals[0] = vec.x;
                        vals[1] = vec.y;
                        vals[2] = vec.z;
                        vals[3] = vec.w;
                    } else {
                        vals[0] = ptr[0];
                        vals[1] = ptr[1];
                        vals[2] = ptr[2];
                        vals[3] = ptr[3];
                    }
                }
            } else {
#pragma unroll
                for (int v = 0; v < VEC; ++v) {
                    int global_col = global_col_base + v;
                    vals[v] = load_b(B, global_k, global_col, k, n, ldb, trans_b != 0);
                }
            }
        } else {
#pragma unroll
            for (int v = 0; v < VEC; ++v) {
                vals[v] = 0.0f;
            }
        }
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
            int col_local = col_base + v;
            Bs[k_row][col_local] = vals[v];
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float b_vals[VEC];
#pragma unroll
            for (int v = 0; v < VEC; ++v) {
                b_vals[v] = Bs[kk][col_idx[v]];
            }
#pragma unroll
            for (int r = 0; r < VEC; ++r) {
                float a_val = As[row_idx[r]][kk];
#pragma unroll
                for (int c = 0; c < VEC; ++c) {
                    acc[r][c] += a_val * b_vals[c];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < VEC; ++r) {
        int row_local = row_idx[r];
        int global_row = blockIdx.y * TILE_M + row_local;
        if (global_row < m) {
#pragma unroll
            for (int c = 0; c < VEC; ++c) {
                int col_local = col_idx[c];
                int global_col = blockIdx.x * TILE_N + col_local;
                if (global_col < n) {
                    C[global_row * ldc + global_col] = acc[r][c];
                }
            }
        }
    }
}

extern "C" __global__ void matmul_tiled_128_strided(const float* __restrict__ A,
                                                    const float* __restrict__ B,
                                                    float* __restrict__ C,
                                                    int m,
                                                    int k,
                                                    int n,
                                                    int lda,
                                                    int ldb,
                                                    int ldc,
                                                    long long stride_a,
                                                    long long stride_b,
                                                    long long stride_c,
                                                    int trans_a,
                                                    int trans_b,
                                                    int batch) {
    const int TILE_M = 128;
    const int TILE_N = 128;
    const int TILE_K = 32;
    const int VEC = 4;
    const int THREADS_X = TILE_N / VEC;
    const int THREADS_Y = TILE_M / VEC;
    if (threadIdx.x >= THREADS_X || threadIdx.y >= THREADS_Y) {
        return;
    }
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch) {
        return;
    }
    const float* Abase = A + batch_idx * stride_a;
    const float* Bbase = B + batch_idx * stride_b;
    float* Cbase = C + batch_idx * stride_c;
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];
    const int lane_x = threadIdx.x;
    const int lane_y = threadIdx.y;
    int row_idx[VEC];
    int col_idx[VEC];
#pragma unroll
    for (int v = 0; v < VEC; ++v) {
        row_idx[v] = lane_y + v * THREADS_Y;
        col_idx[v] = lane_x * VEC + v;
    }
    float acc[VEC][VEC];
#pragma unroll
    for (int r = 0; r < VEC; ++r) {
#pragma unroll
        for (int c = 0; c < VEC; ++c) {
            acc[r][c] = 0.0f;
        }
    }
    int tiles = (k + TILE_K - 1) / TILE_K;
    for (int tile = 0; tile < tiles; ++tile) {
        int k_col = tile * TILE_K + lane_x;
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
            int row_local = row_idx[v];
            int global_row = blockIdx.y * TILE_M + row_local;
            float val = 0.0f;
            if (global_row < m && k_col < k) {
                if (trans_a) {
                    val = Abase[k_col * lda + global_row];
                } else {
                    val = Abase[global_row * lda + k_col];
                }
            }
            As[row_local][lane_x] = val;
        }
        int k_row = lane_y;
        int global_k = tile * TILE_K + k_row;
        int col_base = lane_x * VEC;
        int global_col_base = blockIdx.x * TILE_N + col_base;
        float vals[VEC];
        if (global_k < k) {
            int can_vector = 0;
            if (!trans_b) {
                if (global_col_base + VEC - 1 < n) {
                    can_vector = 1;
                }
            } else {
                if (global_k + VEC - 1 < k) {
                    can_vector = 1;
                }
            }
            if (can_vector) {
                if (!trans_b) {
                    const float* ptr = Bbase + global_k * ldb + global_col_base;
                    if ((((uintptr_t)ptr) & 15) == 0) {
                        float4 vec = *reinterpret_cast<const float4*>(ptr);
                        vals[0] = vec.x;
                        vals[1] = vec.y;
                        vals[2] = vec.z;
                        vals[3] = vec.w;
                    } else {
                        vals[0] = ptr[0];
                        vals[1] = ptr[1];
                        vals[2] = ptr[2];
                        vals[3] = ptr[3];
                    }
                } else {
                    const float* ptr = Bbase + global_col_base * ldb + global_k;
                    if ((((uintptr_t)ptr) & 15) == 0) {
                        float4 vec = *reinterpret_cast<const float4*>(ptr);
                        vals[0] = vec.x;
                        vals[1] = vec.y;
                        vals[2] = vec.z;
                        vals[3] = vec.w;
                    } else {
                        vals[0] = ptr[0];
                        vals[1] = ptr[1];
                        vals[2] = ptr[2];
                        vals[3] = ptr[3];
                    }
                }
            } else {
#pragma unroll
                for (int v = 0; v < VEC; ++v) {
                    int global_col = global_col_base + v;
                    vals[v] = load_b(Bbase, global_k, global_col, k, n, ldb, trans_b != 0);
                }
            }
        } else {
#pragma unroll
            for (int v = 0; v < VEC; ++v) {
                vals[v] = 0.0f;
            }
        }
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
            int col_local = col_base + v;
            Bs[k_row][col_local] = vals[v];
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float b_vals[VEC];
#pragma unroll
            for (int v = 0; v < VEC; ++v) {
                b_vals[v] = Bs[kk][col_idx[v]];
            }
#pragma unroll
            for (int r = 0; r < VEC; ++r) {
                float a_val = As[row_idx[r]][kk];
#pragma unroll
                for (int c = 0; c < VEC; ++c) {
                    acc[r][c] += a_val * b_vals[c];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < VEC; ++r) {
        int row_local = row_idx[r];
        int global_row = blockIdx.y * TILE_M + row_local;
        if (global_row < m) {
#pragma unroll
            for (int c = 0; c < VEC; ++c) {
                int col_local = col_idx[c];
                int global_col = blockIdx.x * TILE_N + col_local;
                if (global_col < n) {
                    Cbase[global_row * ldc + global_col] = acc[r][c];
                }
            }
        }
    }
}
"#;

    pub fn is_available() -> bool {
        if let Ok(ctx) = global_context() {
            let stream = ctx.default_stream();
            CudaBlas::new(stream).is_ok()
        } else {
            false
        }
    }

    fn build_cublas_config_row_major(
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Result<GemmConfig<f32>, String> {
        let m_i = usize_to_i32("gemm m", m)?;
        let n_i = usize_to_i32("gemm n", n)?;
        let k_i = usize_to_i32("gemm k", k)?;
        let transa_cublas = if trans_b {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let transb_cublas = if trans_a {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
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
        load_module_cached(ctx, "reduce", REDUCE_KERNEL_SRC)
    }

    fn load_matmul_module(ctx: &Arc<CudaContext>) -> Result<Arc<CudaModule>, String> {
        load_module_cached(ctx, "matmul", MATMUL_KERNEL_SRC)
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

    fn global_context() -> Result<Arc<CudaContext>, String> {
        let entry = GLOBAL_CONTEXT
            .get_or_init(|| CudaContext::new(0).map_err(|e| format!("cuda: {:?}", e)));
        match entry {
            Ok(ctx) => Ok(Arc::clone(ctx)),
            Err(err) => Err(err.clone()),
        }
    }

    fn module_cache() -> &'static Mutex<HashMap<usize, HashMap<&'static str, Arc<CudaModule>>>> {
        MODULE_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn load_module_cached(
        ctx: &Arc<CudaContext>,
        key: &'static str,
        source: &str,
    ) -> Result<Arc<CudaModule>, String> {
        let ctx_key = Arc::as_ptr(ctx) as usize;
        if let Ok(cache) = module_cache().lock() {
            if let Some(mods) = cache.get(&ctx_key) {
                if let Some(module) = mods.get(key) {
                    return Ok(module.clone());
                }
            }
        }

        let mut opts = CompileOptions::default();
        opts.include_paths = nvrtc_include_paths();
        let ptx = compile_ptx_with_opts(source, opts).map_err(|e| format!("nvrtc: {:?}", e))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("load module: {:?}", e))?;

        let mut cache = module_cache()
            .lock()
            .map_err(|_| "module cache poisoned".to_string())?;
        let entry = cache.entry(ctx_key).or_insert_with(HashMap::new);
        entry.entry(key).or_insert_with(|| module.clone());

        Ok(module)
    }

    fn path_to_string(path: &PathBuf) -> String {
        path.to_string_lossy().into_owned()
    }

    fn build_cublaslt_config_row_major(
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Result<MatmulConfig, String> {
        let gemm = build_cublas_config_row_major(m, k, n, trans_a, trans_b)?;
        Ok(MatmulConfig {
            transa: gemm.transa == cublasOperation_t::CUBLAS_OP_T,
            transb: gemm.transb == cublasOperation_t::CUBLAS_OP_T,
            transc: false,
            m: gemm.m as u64,
            n: gemm.n as u64,
            k: gemm.k as u64,
            alpha: gemm.alpha,
            lda: gemm.lda as i64,
            ldb: gemm.ldb as i64,
            beta: gemm.beta,
            ldc: gemm.ldc as i64,
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        })
    }

    fn build_cublaslt_config_row_major_strided(
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
        batch: usize,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
    ) -> Result<MatmulConfig, String> {
        let mut cfg = build_cublaslt_config_row_major(m, k, n, trans_a, trans_b)?;
        cfg.stride_a = Some(stride_b);
        cfg.stride_b = Some(stride_a);
        cfg.stride_c = Some(stride_c);
        cfg.batch_size = Some(usize_to_i32("cublasLt batch size", batch)?);
        Ok(cfg)
    }

    fn ensure_no_alias<T, A, B, Cc>(
        stream: &Arc<CudaStream>,
        a: &A,
        b: &B,
        c: &mut Cc,
        label: &str,
    ) -> Result<(), String>
    where
        A: DevicePtr<T> + ?Sized,
        B: DevicePtr<T> + ?Sized,
        Cc: DevicePtrMut<T> + ?Sized,
    {
        let (ap, _) = a.device_ptr(stream);
        let (bp, _) = b.device_ptr(stream);
        let (cp, _) = c.device_ptr_mut(stream);
        if cp == ap || cp == bp {
            Err(format!("{label}: output aliases input"))
        } else {
            Ok(())
        }
    }

    fn launch_cublaslt_f32<I, Cc>(
        stream: Arc<CudaStream>,
        a: &I,
        b: &I,
        c: &mut Cc,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Result<(), String>
    where
        I: DevicePtr<f32>,
        Cc: DevicePtrMut<f32>,
    {
        let cfg = build_cublaslt_config_row_major(m, k, n, trans_a, trans_b)?;
        let blas_lt = CudaBlasLT::new(stream.clone()).map_err(|e| format!("cublasLt: {:?}", e))?;
        unsafe {
            <CudaBlasLT as Matmul<f32>>::matmul(&blas_lt, cfg, b, a, c, None, None)
                .map_err(|e| format!("cublasLt matmul: {:?}", e))
        }
    }

    fn launch_cublaslt_f32_strided<I, Cc>(
        stream: Arc<CudaStream>,
        a: &I,
        b: &I,
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
    ) -> Result<(), String>
    where
        I: DevicePtr<f32>,
        Cc: DevicePtrMut<f32>,
    {
        let cfg = build_cublaslt_config_row_major_strided(
            m, k, n, trans_a, trans_b, batch, stride_a, stride_b, stride_c,
        )?;
        let blas_lt = CudaBlasLT::new(stream.clone()).map_err(|e| format!("cublasLt: {:?}", e))?;
        unsafe {
            <CudaBlasLT as Matmul<f32>>::matmul(&blas_lt, cfg, b, a, c, None, None)
                .map_err(|e| format!("cublasLt matmul: {:?}", e))
        }
    }

    fn launch_cublaslt_half<T, I, Cc>(
        stream: Arc<CudaStream>,
        a: &I,
        b: &I,
        c: &mut Cc,
        cfg: MatmulConfig,
    ) -> Result<(), String>
    where
        T: 'static,
        I: DevicePtr<T>,
        Cc: DevicePtrMut<T>,
        CudaBlasLT: Matmul<T>,
    {
        let blas_lt = CudaBlasLT::new(stream.clone()).map_err(|e| format!("cublasLt: {:?}", e))?;
        unsafe {
            <CudaBlasLT as Matmul<T>>::matmul(&blas_lt, cfg, b, a, c, None, None)
                .map_err(|e| format!("cublasLt matmul: {:?}", e))
        }
    }

    fn usize_to_i32(label: &str, value: usize) -> Result<i32, String> {
        if value > i32::MAX as usize {
            Err(format!("{label}: {value} exceeds i32::MAX"))
        } else {
            Ok(value as i32)
        }
    }

    fn usize_to_i64(label: &str, value: usize) -> Result<i64, String> {
        if value > i64::MAX as usize {
            Err(format!("{label}: {value} exceeds i64::MAX"))
        } else {
            Ok(value as i64)
        }
    }

    fn blocks_for_len(label: &str, len: usize, block_dim: u32) -> Result<u32, String> {
        if len == 0 {
            return Ok(1);
        }
        let block_dim = block_dim.max(1) as usize;
        let adjusted = len
            .checked_add(block_dim - 1)
            .ok_or_else(|| format!("{label}: length overflow computing grid dim"))?;
        let blocks = adjusted / block_dim;
        let blocks = blocks.max(1);
        if blocks > u32::MAX as usize {
            return Err(format!("{label}: grid_dim.x exceeds u32::MAX"));
        }
        Ok(blocks as u32)
    }

    fn checked_stride_product(label: &str, stride: i64, count: usize) -> Result<usize, String> {
        if stride < 0 {
            return Err(format!("{label}: negative stride {stride} not supported"));
        }
        let product = (stride as u128)
            .checked_mul(count as u128)
            .ok_or_else(|| format!("{label}: stride overflow"))?;
        if product > usize::MAX as u128 {
            return Err(format!("{label}: stride product exceeds usize::MAX"));
        }
        Ok(product as usize)
    }

    extern "C" fn smem_per_thread_f32(block_size: i32) -> usize {
        (block_size as usize) * std::mem::size_of::<f32>()
    }

    extern "C" fn smem_per_thread_f32_i32(block_size: i32) -> usize {
        (block_size as usize) * (std::mem::size_of::<f32>() + std::mem::size_of::<i32>())
    }

    extern "C" fn smem_per_thread_i32(block_size: i32) -> usize {
        (block_size as usize) * std::mem::size_of::<i32>()
    }

    extern "C" fn smem_per_thread_f64(block_size: i32) -> usize {
        (block_size as usize) * std::mem::size_of::<f64>()
    }

    extern "C" fn smem_per_thread_2xf64(block_size: i32) -> usize {
        (block_size as usize) * 2 * std::mem::size_of::<f64>()
    }

    fn select_block_dim(
        func: &CudaFunction,
        smem_fn: extern "C" fn(i32) -> usize,
    ) -> Result<u32, String> {
        let block = match func.occupancy_max_potential_block_size(smem_fn, 0, 0, None) {
            Ok((block, _)) if block > 0 => block as u32,
            _ => 256,
        };
        Ok(block.clamp(64, 1024))
    }

    fn shared_mem_bytes(block_dim: u32, bytes_per_thread: usize) -> Result<u32, String> {
        let total = (block_dim as usize)
            .checked_mul(bytes_per_thread)
            .ok_or_else(|| "shared memory size overflow".to_string())?;
        if total > u32::MAX as usize {
            return Err("shared memory size exceeds u32::MAX".into());
        }
        Ok(total as u32)
    }

    fn grid_dim_for(label: &str, len: usize, tile: u32) -> Result<u32, String> {
        if len == 0 {
            return Ok(0);
        }
        let tile = tile.max(1) as usize;
        let adjusted = len
            .checked_add(tile - 1)
            .ok_or_else(|| format!("{label}: overflow computing grid dimension"))?;
        let blocks = adjusted / tile;
        if blocks > u32::MAX as usize {
            return Err(format!("{label}: grid_dim exceeds u32::MAX"));
        }
        Ok(blocks as u32)
    }

    struct MatmulTileConfig {
        kernel: &'static str,
        kernel_strided: &'static str,
        tile_m: u32,
        tile_n: u32,
        block_x: u32,
        block_y: u32,
    }

    fn select_matmul_tile_config(m: usize, n: usize, k: usize) -> MatmulTileConfig {
        if m >= 128 && n >= 128 && k >= 32 {
            MatmulTileConfig {
                kernel: "matmul_tiled_128",
                kernel_strided: "matmul_tiled_128_strided",
                tile_m: 128,
                tile_n: 128,
                block_x: 32,
                block_y: 32,
            }
        } else if m >= 64 && n >= 64 && k >= 16 {
            MatmulTileConfig {
                kernel: "matmul_tiled_64",
                kernel_strided: "matmul_tiled_64_strided",
                tile_m: 64,
                tile_n: 64,
                block_x: 16,
                block_y: 16,
            }
        } else {
            MatmulTileConfig {
                kernel: "matmul_tiled",
                kernel_strided: "matmul_tiled_strided",
                tile_m: 16,
                tile_n: 16,
                block_x: 16,
                block_y: 16,
            }
        }
    }

    fn leading_dims(
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Result<(i32, i32, i32), String> {
        let lda = if trans_a { m } else { k };
        let ldb = if trans_b { k } else { n };
        let ldc = n;
        Ok((
            usize_to_i32("matmul lda", lda)?,
            usize_to_i32("matmul ldb", ldb)?,
            usize_to_i32("matmul ldc", ldc)?,
        ))
    }

    fn launch_matmul_tiled(
        stream: Arc<CudaStream>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Result<(), String> {
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        let tile_cfg = select_matmul_tile_config(m, n, k);
        let ctx = stream.context().clone();
        let module = load_matmul_module(&ctx)?;
        let func = module
            .load_function(tile_cfg.kernel)
            .map_err(|e| format!("load {}: {:?}", tile_cfg.kernel, e))?;
        let grid_x = grid_dim_for("matmul grid x", n, tile_cfg.tile_n)?;
        let grid_y = grid_dim_for("matmul grid y", m, tile_cfg.tile_m)?;
        if grid_x == 0 || grid_y == 0 {
            return Ok(());
        }
        let (lda, ldb, ldc) = leading_dims(m, k, n, trans_a, trans_b)?;
        let m_i = usize_to_i32("matmul m", m)?;
        let k_i = usize_to_i32("matmul k", k)?;
        let n_i = usize_to_i32("matmul n", n)?;
        let trans_a_i = if trans_a { 1 } else { 0 };
        let trans_b_i = if trans_b { 1 } else { 0 };
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (tile_cfg.block_x, tile_cfg.block_y, 1),
            shared_mem_bytes: 0,
        };
        let (a_ptr, _sync_a) = a.device_ptr(&stream);
        let (b_ptr, _sync_b) = b.device_ptr(&stream);
        let (c_ptr, _sync_c) = c.device_ptr_mut(&stream);
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&a_ptr)
                .arg(&b_ptr)
                .arg(&c_ptr)
                .arg(&m_i)
                .arg(&k_i)
                .arg(&n_i)
                .arg(&lda)
                .arg(&ldb)
                .arg(&ldc)
                .arg(&trans_a_i)
                .arg(&trans_b_i)
                .launch(cfg)
        }
        .map(|_| ())
        .map_err(|e| format!("launch {}: {:?}", tile_cfg.kernel, e))
    }

    fn launch_matmul_tiled_strided(
        stream: Arc<CudaStream>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
    ) -> Result<(), String> {
        if batch == 0 || m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        let tile_cfg = select_matmul_tile_config(m, n, k);
        let ctx = stream.context().clone();
        let module = load_matmul_module(&ctx)?;
        let func = module
            .load_function(tile_cfg.kernel_strided)
            .map_err(|e| format!("load {}: {:?}", tile_cfg.kernel_strided, e))?;
        let grid_x = grid_dim_for("matmul grid x", n, tile_cfg.tile_n)?;
        let grid_y = grid_dim_for("matmul grid y", m, tile_cfg.tile_m)?;
        if grid_x == 0 || grid_y == 0 {
            return Ok(());
        }
        let grid_z = usize_to_i32("matmul batch size", batch)?;
        if grid_z <= 0 {
            return Ok(());
        }
        let (lda, ldb, ldc) = leading_dims(m, k, n, trans_a, trans_b)?;
        let m_i = usize_to_i32("matmul m", m)?;
        let k_i = usize_to_i32("matmul k", k)?;
        let n_i = usize_to_i32("matmul n", n)?;
        let trans_a_i = if trans_a { 1 } else { 0 };
        let trans_b_i = if trans_b { 1 } else { 0 };
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, grid_z as u32),
            block_dim: (tile_cfg.block_x, tile_cfg.block_y, 1),
            shared_mem_bytes: 0,
        };
        let (a_ptr, _sync_a) = a.device_ptr(&stream);
        let (b_ptr, _sync_b) = b.device_ptr(&stream);
        let (c_ptr, _sync_c) = c.device_ptr_mut(&stream);
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&a_ptr)
                .arg(&b_ptr)
                .arg(&c_ptr)
                .arg(&m_i)
                .arg(&k_i)
                .arg(&n_i)
                .arg(&lda)
                .arg(&ldb)
                .arg(&ldc)
                .arg(&stride_a)
                .arg(&stride_b)
                .arg(&stride_c)
                .arg(&trans_a_i)
                .arg(&trans_b_i)
                .arg(&grid_z)
                .launch(cfg)
        }
        .map(|_| ())
        .map_err(|e| format!("launch {}: {:?}", tile_cfg.kernel_strided, e))
    }

    #[allow(dead_code)]
    pub fn matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> CoreResult<Vec<f32>> {
        matmul_f32_ex_with_policy(
            a,
            b,
            m,
            k,
            n,
            false,
            false,
            MatmulTensorCorePolicy::Accuracy,
        )
    }

    pub fn matmul_f32_with_policy(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        policy: MatmulTensorCorePolicy,
    ) -> CoreResult<Vec<f32>> {
        matmul_f32_ex_with_policy(a, b, m, k, n, false, false, policy)
    }

    #[allow(dead_code)]
    pub fn matmul_f32_ex(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> CoreResult<Vec<f32>> {
        matmul_f32_ex_with_policy(
            a,
            b,
            m,
            k,
            n,
            trans_a,
            trans_b,
            MatmulTensorCorePolicy::Accuracy,
        )
    }

    pub fn matmul_f32_ex_with_policy(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
        policy: MatmulTensorCorePolicy,
    ) -> CoreResult<Vec<f32>> {
        let need_a = if trans_a {
            k.checked_mul(m)
        } else {
            m.checked_mul(k)
        }
        .ok_or("size overflow")?;
        let need_b = if trans_b {
            n.checked_mul(k)
        } else {
            k.checked_mul(n)
        }
        .ok_or("size overflow")?;
        if a.len() != need_a || b.len() != need_b {
            return Err("matmul_ex: shape mismatch".into());
        }
        let total = m.checked_mul(n).ok_or("size overflow")?;
        if total == 0 {
            return Ok(Vec::new());
        }
        let ctx = global_context()?;
        let stream = ctx.default_stream();
        match policy {
            MatmulTensorCorePolicy::Accuracy => {
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
                match CudaBlas::new(stream.clone()) {
                    Ok(blas) => {
                        match unsafe { blas.gemm(cfg, &d_b, &d_a, &mut d_c) } {
                            Ok(()) => stream
                                .memcpy_dtov(&d_c)
                                .map_err(|e| format!("dtoh c: {:?}", e)),
                            Err(err) => {
                                if cfg!(debug_assertions) {
                                    eprintln!("cuBLAS sgemm failed ({err:?}), falling back to tiled kernel");
                                }
                                launch_matmul_tiled(
                                    stream.clone(),
                                    &d_a,
                                    &d_b,
                                    &mut d_c,
                                    m,
                                    k,
                                    n,
                                    trans_a,
                                    trans_b,
                                )?;
                                stream
                                    .memcpy_dtov(&d_c)
                                    .map_err(|e| format!("dtoh c: {:?}", e))
                            }
                        }
                    }
                    Err(err) => {
                        if cfg!(debug_assertions) {
                            eprintln!("cuBLAS unavailable ({err:?}), using tiled kernel fallback");
                        }
                        launch_matmul_tiled(
                            stream.clone(),
                            &d_a,
                            &d_b,
                            &mut d_c,
                            m,
                            k,
                            n,
                            trans_a,
                            trans_b,
                        )?;
                        stream
                            .memcpy_dtov(&d_c)
                            .map_err(|e| format!("dtoh c: {:?}", e))
                    }
                }
            }
            MatmulTensorCorePolicy::Performance => {
                let d_a = stream
                    .memcpy_stod(a)
                    .map_err(|e| format!("htod a: {:?}", e))?;
                let d_b = stream
                    .memcpy_stod(b)
                    .map_err(|e| format!("htod b: {:?}", e))?;
                let mut d_c = stream
                    .alloc_zeros::<f32>(total)
                    .map_err(|e| format!("alloc c: {:?}", e))?;
                match launch_cublaslt_f32(
                    stream.clone(),
                    &d_a,
                    &d_b,
                    &mut d_c,
                    m,
                    k,
                    n,
                    trans_a,
                    trans_b,
                ) {
                    Ok(()) => stream
                        .memcpy_dtov(&d_c)
                        .map_err(|e| format!("dtoh c: {:?}", e)),
                    Err(err) => {
                        // 鍥為€€鍒颁紶缁?cuBLAS
                        let cfg = build_cublas_config_row_major(m, k, n, trans_a, trans_b)?;
                        if cfg!(debug_assertions) {
                            eprintln!(
                                "cublasLt fast path failed ({err}), attempting cuBLAS fallback"
                            );
                        }
                        match CudaBlas::new(stream.clone()) {
                            Ok(blas) => match unsafe { blas.gemm(cfg, &d_b, &d_a, &mut d_c) } {
                                Ok(()) => stream
                                    .memcpy_dtov(&d_c)
                                    .map_err(|e| format!("dtoh c: {:?}", e)),
                                Err(gemm_err) => {
                                    if cfg!(debug_assertions) {
                                        eprintln!(
                                            "cuBLAS fallback failed ({gemm_err:?}), using tiled kernel"
                                        );
                                    }
                                    launch_matmul_tiled(
                                        stream.clone(),
                                        &d_a,
                                        &d_b,
                                        &mut d_c,
                                        m,
                                        k,
                                        n,
                                        trans_a,
                                        trans_b,
                                    )?;
                                    stream
                                        .memcpy_dtov(&d_c)
                                        .map_err(|e| format!("dtoh c: {:?}", e))
                                }
                            },
                            Err(blas_err) => {
                                if cfg!(debug_assertions) {
                                    eprintln!(
                                        "cuBLAS unavailable ({blas_err:?}), using tiled kernel fallback"
                                    );
                                }
                                launch_matmul_tiled(
                                    stream.clone(),
                                    &d_a,
                                    &d_b,
                                    &mut d_c,
                                    m,
                                    k,
                                    n,
                                    trans_a,
                                    trans_b,
                                )?;
                                stream
                                    .memcpy_dtov(&d_c)
                                    .map_err(|e| format!("dtoh c: {:?}", e))
                            }
                        }
                    }
                }
            }
            MatmulTensorCorePolicy::Float16 => {
                let cfg = build_cublaslt_config_row_major(m, k, n, trans_a, trans_b)?;
                let a_half: Vec<f16> = a.iter().map(|&v| f16::from_f32(v)).collect();
                let b_half: Vec<f16> = b.iter().map(|&v| f16::from_f32(v)).collect();
                let d_a = stream
                    .memcpy_stod(&a_half)
                    .map_err(|e| format!("htod a[f16]: {:?}", e))?;
                let d_b = stream
                    .memcpy_stod(&b_half)
                    .map_err(|e| format!("htod b[f16]: {:?}", e))?;
                let mut d_c = stream
                    .alloc_zeros::<f16>(total)
                    .map_err(|e| format!("alloc c[f16]: {:?}", e))?;
                launch_cublaslt_half::<f16, _, _>(stream.clone(), &d_a, &d_b, &mut d_c, cfg)?;
                let host_half = stream
                    .memcpy_dtov(&d_c)
                    .map_err(|e| format!("dtoh c[f16]: {:?}", e))?;
                Ok(host_half.into_iter().map(|v| f32::from(v)).collect())
            }
            MatmulTensorCorePolicy::BFloat16 => {
                let cfg = build_cublaslt_config_row_major(m, k, n, trans_a, trans_b)?;
                let a_half: Vec<bf16> = a.iter().map(|&v| bf16::from_f32(v)).collect();
                let b_half: Vec<bf16> = b.iter().map(|&v| bf16::from_f32(v)).collect();
                let d_a = stream
                    .memcpy_stod(&a_half)
                    .map_err(|e| format!("htod a[bf16]: {:?}", e))?;
                let d_b = stream
                    .memcpy_stod(&b_half)
                    .map_err(|e| format!("htod b[bf16]: {:?}", e))?;
                let mut d_c = stream
                    .alloc_zeros::<bf16>(total)
                    .map_err(|e| format!("alloc c[bf16]: {:?}", e))?;
                launch_cublaslt_half::<bf16, _, _>(stream.clone(), &d_a, &d_b, &mut d_c, cfg)?;
                let host_half = stream
                    .memcpy_dtov(&d_c)
                    .map_err(|e| format!("dtoh c[bf16]: {:?}", e))?;
                Ok(host_half.into_iter().map(|v| f32::from(v)).collect())
            }
        }
    }

    #[allow(dead_code)]
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
        matmul_batched_f32_with_policy(
            a,
            b,
            batch,
            m,
            k,
            n,
            trans_a,
            trans_b,
            MatmulTensorCorePolicy::Accuracy,
        )
    }

    pub fn matmul_batched_f32_with_policy(
        a: &[f32],
        b: &[f32],
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
        policy: MatmulTensorCorePolicy,
    ) -> CoreResult<Vec<f32>> {
        let a_each = if trans_a {
            k.checked_mul(m)
        } else {
            m.checked_mul(k)
        }
        .ok_or("size overflow")?;
        let b_each = if trans_b {
            n.checked_mul(k)
        } else {
            k.checked_mul(n)
        }
        .ok_or("size overflow")?;
        let c_each = m.checked_mul(n).ok_or("size overflow")?;
        if a.len() != batch.checked_mul(a_each).ok_or("size overflow")?
            || b.len() != batch.checked_mul(b_each).ok_or("size overflow")?
        {
            return Err("matmul_batched_f32: shape mismatch".into());
        }
        matmul_batched_f32_strided_with_policy(
            a,
            b,
            batch,
            m,
            k,
            n,
            trans_a,
            trans_b,
            usize_to_i64("matmul_batched_f32 stride_a", a_each)?,
            usize_to_i64("matmul_batched_f32 stride_b", b_each)?,
            usize_to_i64("matmul_batched_f32 stride_c", c_each)?,
            policy,
        )
    }

    #[allow(dead_code)]
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
        matmul_batched_f32_strided_with_policy(
            a,
            b,
            batch,
            m,
            k,
            n,
            trans_a,
            trans_b,
            stride_a,
            stride_b,
            stride_c,
            MatmulTensorCorePolicy::Accuracy,
        )
    }

    pub fn matmul_batched_f32_strided_with_policy(
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
        policy: MatmulTensorCorePolicy,
    ) -> CoreResult<Vec<f32>> {
        let batch_i32 = usize_to_i32("matmul_batched_f32_strided batch", batch)?;
        let a_need =
            checked_stride_product("matmul_batched_f32_strided stride_a", stride_a, batch)?;
        let b_need =
            checked_stride_product("matmul_batched_f32_strided stride_b", stride_b, batch)?;
        let c_need =
            checked_stride_product("matmul_batched_f32_strided stride_c", stride_c, batch)?;
        if a.len() < a_need || b.len() < b_need {
            return Err("matmul_batched_f32_strided: buffer too small for given stride".into());
        }
        let ctx = global_context()?;
        let stream = ctx.default_stream();
        match policy {
            MatmulTensorCorePolicy::Accuracy => {
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
                    batch_size: batch_i32,
                    stride_a: stride_b,
                    stride_b: stride_a,
                    stride_c,
                };
                match CudaBlas::new(stream.clone()) {
                    Ok(blas) => {
                        match unsafe { blas.gemm_strided_batched(cfg, &d_b, &d_a, &mut d_c) } {
                            Ok(()) => stream
                                .memcpy_dtov(&d_c)
                                .map_err(|e| format!("dtoh c: {:?}", e)),
                            Err(err) => {
                                if cfg!(debug_assertions) {
                                    eprintln!(
                                    "cuBLAS strided batched sgemm failed ({err:?}), using tiled kernel fallback"
                                );
                                }
                                launch_matmul_tiled_strided(
                                    stream.clone(),
                                    &d_a,
                                    &d_b,
                                    &mut d_c,
                                    batch,
                                    m,
                                    k,
                                    n,
                                    trans_a,
                                    trans_b,
                                    stride_a,
                                    stride_b,
                                    stride_c,
                                )?;
                                stream
                                    .memcpy_dtov(&d_c)
                                    .map_err(|e| format!("dtoh c: {:?}", e))
                            }
                        }
                    }
                    Err(err) => {
                        if cfg!(debug_assertions) {
                            eprintln!(
                                "cuBLAS unavailable ({err:?}) for strided batched, using tiled kernel fallback"
                            );
                        }
                        launch_matmul_tiled_strided(
                            stream.clone(),
                            &d_a,
                            &d_b,
                            &mut d_c,
                            batch,
                            m,
                            k,
                            n,
                            trans_a,
                            trans_b,
                            stride_a,
                            stride_b,
                            stride_c,
                        )?;
                        stream
                            .memcpy_dtov(&d_c)
                            .map_err(|e| format!("dtoh c: {:?}", e))
                    }
                }
            }
            MatmulTensorCorePolicy::Performance => {
                let d_a = stream
                    .memcpy_stod(a)
                    .map_err(|e| format!("htod a: {:?}", e))?;
                let d_b = stream
                    .memcpy_stod(b)
                    .map_err(|e| format!("htod b: {:?}", e))?;
                let mut d_c = stream
                    .alloc_zeros::<f32>(c_need)
                    .map_err(|e| format!("alloc c: {:?}", e))?;
                match launch_cublaslt_f32_strided(
                    stream.clone(),
                    &d_a,
                    &d_b,
                    &mut d_c,
                    batch,
                    m,
                    k,
                    n,
                    trans_a,
                    trans_b,
                    stride_a,
                    stride_b,
                    stride_c,
                ) {
                    Ok(()) => stream
                        .memcpy_dtov(&d_c)
                        .map_err(|e| format!("dtoh c: {:?}", e)),
                    Err(err) => {
                        if cfg!(debug_assertions) {
                            eprintln!(
                                "cublasLt strided fast path failed ({err}), attempting cuBLAS fallback"
                            );
                        }
                        let gemm_cfg = build_cublas_config_row_major(m, k, n, trans_a, trans_b)?;
                        let cfg = StridedBatchedConfig::<f32> {
                            gemm: gemm_cfg,
                            batch_size: batch_i32,
                            stride_a: stride_b,
                            stride_b: stride_a,
                            stride_c,
                        };
                        match CudaBlas::new(stream.clone()) {
                            Ok(blas) => match unsafe {
                                blas.gemm_strided_batched(cfg, &d_b, &d_a, &mut d_c)
                            } {
                                Ok(()) => stream
                                    .memcpy_dtov(&d_c)
                                    .map_err(|e| format!("dtoh c: {:?}", e)),
                                Err(gemm_err) => {
                                    if cfg!(debug_assertions) {
                                        eprintln!(
                                            "cuBLAS strided fallback failed ({gemm_err:?}), using tiled kernel"
                                        );
                                    }
                                    launch_matmul_tiled_strided(
                                        stream.clone(),
                                        &d_a,
                                        &d_b,
                                        &mut d_c,
                                        batch,
                                        m,
                                        k,
                                        n,
                                        trans_a,
                                        trans_b,
                                        stride_a,
                                        stride_b,
                                        stride_c,
                                    )?;
                                    stream
                                        .memcpy_dtov(&d_c)
                                        .map_err(|e| format!("dtoh c: {:?}", e))
                                }
                            },
                            Err(blas_err) => {
                                if cfg!(debug_assertions) {
                                    eprintln!(
                                        "cuBLAS unavailable ({blas_err:?}) for strided fallback, using tiled kernel"
                                    );
                                }
                                launch_matmul_tiled_strided(
                                    stream.clone(),
                                    &d_a,
                                    &d_b,
                                    &mut d_c,
                                    batch,
                                    m,
                                    k,
                                    n,
                                    trans_a,
                                    trans_b,
                                    stride_a,
                                    stride_b,
                                    stride_c,
                                )?;
                                stream
                                    .memcpy_dtov(&d_c)
                                    .map_err(|e| format!("dtoh c: {:?}", e))
                            }
                        }
                    }
                }
            }
            MatmulTensorCorePolicy::Float16 => {
                let cfg = build_cublaslt_config_row_major_strided(
                    m, k, n, trans_a, trans_b, batch, stride_a, stride_b, stride_c,
                )?;
                let a_half: Vec<f16> = a.iter().map(|&v| f16::from_f32(v)).collect();
                let b_half: Vec<f16> = b.iter().map(|&v| f16::from_f32(v)).collect();
                let d_a = stream
                    .memcpy_stod(&a_half)
                    .map_err(|e| format!("htod a[f16]: {:?}", e))?;
                let d_b = stream
                    .memcpy_stod(&b_half)
                    .map_err(|e| format!("htod b[f16]: {:?}", e))?;
                let mut d_c = stream
                    .alloc_zeros::<f16>(c_need)
                    .map_err(|e| format!("alloc c[f16]: {:?}", e))?;
                launch_cublaslt_half::<f16, _, _>(stream.clone(), &d_a, &d_b, &mut d_c, cfg)?;
                let host_half = stream
                    .memcpy_dtov(&d_c)
                    .map_err(|e| format!("dtoh c[f16]: {:?}", e))?;
                Ok(host_half.into_iter().map(|v| f32::from(v)).collect())
            }
            MatmulTensorCorePolicy::BFloat16 => {
                let cfg = build_cublaslt_config_row_major_strided(
                    m, k, n, trans_a, trans_b, batch, stride_a, stride_b, stride_c,
                )?;
                let a_half: Vec<bf16> = a.iter().map(|&v| bf16::from_f32(v)).collect();
                let b_half: Vec<bf16> = b.iter().map(|&v| bf16::from_f32(v)).collect();
                let d_a = stream
                    .memcpy_stod(&a_half)
                    .map_err(|e| format!("htod a[bf16]: {:?}", e))?;
                let d_b = stream
                    .memcpy_stod(&b_half)
                    .map_err(|e| format!("htod b[bf16]: {:?}", e))?;
                let mut d_c = stream
                    .alloc_zeros::<bf16>(c_need)
                    .map_err(|e| format!("alloc c[bf16]: {:?}", e))?;
                launch_cublaslt_half::<bf16, _, _>(stream.clone(), &d_a, &d_b, &mut d_c, cfg)?;
                let host_half = stream
                    .memcpy_dtov(&d_c)
                    .map_err(|e| format!("dtoh c[bf16]: {:?}", e))?;
                Ok(host_half.into_iter().map(|v| f32::from(v)).collect())
            }
        }
    }

    pub fn matmul_f32_ex_device<I: DevicePtr<f32>, Cc: DevicePtrMut<f32>>(
        stream: Arc<CudaStream>,
        a: &I,
        b: &I,
        c: &mut Cc,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> CoreResult<()> {
        matmul_f32_ex_device_with_policy(
            stream,
            a,
            b,
            c,
            m,
            k,
            n,
            trans_a,
            trans_b,
            MatmulTensorCorePolicy::Accuracy,
        )
    }

    pub fn matmul_f32_ex_device_with_policy<I: DevicePtr<f32>, Cc: DevicePtrMut<f32>>(
        stream: Arc<CudaStream>,
        a: &I,
        b: &I,
        c: &mut Cc,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
        policy: MatmulTensorCorePolicy,
    ) -> CoreResult<()> {
        ensure_no_alias(&stream, a, b, c, "matmul_f32_ex_device")?;
        match policy {
            MatmulTensorCorePolicy::Accuracy => {
                let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("cublas: {:?}", e))?;
                let cfg = build_cublas_config_row_major(m, k, n, trans_a, trans_b)?;
                unsafe { blas.gemm(cfg, b, a, c) }.map_err(|e| format!("sgemm: {:?}", e))
            }
            MatmulTensorCorePolicy::Performance => {
                launch_cublaslt_f32(stream, a, b, c, m, k, n, trans_a, trans_b)
                    .map_err(|e| format!("cublasLt matmul: {e}"))
            }
            MatmulTensorCorePolicy::Float16 | MatmulTensorCorePolicy::BFloat16 => Err(
                "matmul_f32_ex_device_with_policy: Float16/BFloat16 require host-side conversion"
                    .into(),
            ),
        }
    }

    pub fn matmul_batched_f32_strided_device<I: DevicePtr<f32>, Cc: DevicePtrMut<f32>>(
        stream: Arc<CudaStream>,
        a: &I,
        b: &I,
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
        matmul_batched_f32_strided_device_with_policy(
            stream,
            a,
            b,
            c,
            batch,
            m,
            k,
            n,
            trans_a,
            trans_b,
            stride_a,
            stride_b,
            stride_c,
            MatmulTensorCorePolicy::Accuracy,
        )
    }

    pub fn matmul_batched_f32_strided_device_with_policy<
        I: DevicePtr<f32>,
        Cc: DevicePtrMut<f32>,
    >(
        stream: Arc<CudaStream>,
        a: &I,
        b: &I,
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
        policy: MatmulTensorCorePolicy,
    ) -> CoreResult<()> {
        ensure_no_alias(&stream, a, b, c, "matmul_batched_f32_strided_device")?;
        let batch_i32 = usize_to_i32("matmul_batched_f32_strided_device batch", batch)?;
        match policy {
            MatmulTensorCorePolicy::Accuracy => {
                let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("cublas: {:?}", e))?;
                let gemm_cfg = build_cublas_config_row_major(m, k, n, trans_a, trans_b)?;
                let cfg = StridedBatchedConfig::<f32> {
                    gemm: gemm_cfg,
                    batch_size: batch_i32,
                    stride_a: stride_b,
                    stride_b: stride_a,
                    stride_c,
                };
                unsafe { blas.gemm_strided_batched(cfg, b, a, c) }
                    .map_err(|e| format!("sgemm_strided_batched: {:?}", e))
            }
            MatmulTensorCorePolicy::Performance => launch_cublaslt_f32_strided(
                stream,
                a,
                b,
                c,
                batch,
                m,
                k,
                n,
                trans_a,
                trans_b,
                stride_a,
                stride_b,
                stride_c,
            )
            .map_err(|e| format!("cublasLt sgemm_strided_batched: {e}")),
            MatmulTensorCorePolicy::Float16 | MatmulTensorCorePolicy::BFloat16 => Err(
                "matmul_batched_f32_strided_device_with_policy: Float16/BFloat16 require host-side conversion"
                    .into(),
            ),
        }
    }

    pub fn reduce_sum_f32(values: &[f32]) -> CoreResult<f32> {
        if values.is_empty() {
            return Ok(0.0);
        }
        let ctx = global_context()?;
        let stream = ctx.default_stream();
        let device_values = stream
            .memcpy_stod(values)
            .map_err(|e| format!("reduce_sum htod: {:?}", e))?;
        let mut device_out = stream
            .alloc_zeros::<f32>(1)
            .map_err(|e| format!("reduce_sum alloc: {:?}", e))?;
        reduce_sum_f32_device(stream.clone(), &device_values, &mut device_out)?;
        let host = stream
            .memcpy_dtov(&device_out)
            .map_err(|e| format!("reduce_sum dtoh: {:?}", e))?;
        Ok(host.into_iter().next().unwrap_or(0.0))
    }

    pub fn reduce_sum_f32_with_policy(
        values: &[f32],
        policy: SumPrecisionPolicy,
    ) -> CoreResult<f32> {
        match policy {
            SumPrecisionPolicy::Default => reduce_sum_f32(values),
            SumPrecisionPolicy::Float64 => Ok(super::sum_f32_as_f64(values)),
            SumPrecisionPolicy::Kahan => Ok(super::sum_f32_kahan(values)),
        }
    }

    pub fn reduce_sum_f32_device(
        stream: Arc<CudaStream>,
        values: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> CoreResult<()> {
        reduce_sum_f32_device_with_policy(stream, values, out, SumPrecisionPolicy::Default)
    }

    pub fn reduce_sum_f32_device_with_policy(
        stream: Arc<CudaStream>,
        values: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        policy: SumPrecisionPolicy,
    ) -> CoreResult<()> {
        if out.len() == 0 {
            return Err("reduce_sum_f32_device: output slice is empty".into());
        }
        if values.len() == 0 {
            write_scalar_f32(&stream, out, 0.0)?;
            return Ok(());
        }
        let ctx = stream.context().clone();
        let module = load_reduce_module(&ctx)?;
        match policy {
            SumPrecisionPolicy::Default => {
                reduce_sum_device_internal(&stream, &module, values, out)
            }
            SumPrecisionPolicy::Float64 => {
                reduce_sum_device_internal_f64(&stream, &module, values, out)
            }
            SumPrecisionPolicy::Kahan => {
                reduce_sum_device_internal_kahan(&stream, &module, values, out)
            }
        }
    }

    pub fn reduce_max_f32_with_policy(values: &[f32], policy: NanPolicy) -> CoreResult<f32> {
        if values.is_empty() {
            return Err("reduce_max on empty slice".into());
        }
        if matches!(policy, NanPolicy::Ignore) && !values.iter().any(|v| !v.is_nan()) {
            return Ok(f32::NAN);
        }
        if matches!(policy, NanPolicy::Propagate) && values.iter().any(|v| v.is_nan()) {
            return Ok(f32::NAN);
        }
        let ctx = global_context()?;
        let stream = ctx.default_stream();
        let device_values = stream
            .memcpy_stod(values)
            .map_err(|e| format!("reduce_max htod: {:?}", e))?;
        let mut device_out = stream
            .alloc_zeros::<f32>(1)
            .map_err(|e| format!("reduce_max alloc: {:?}", e))?;
        reduce_max_f32_device_with_policy(stream.clone(), &device_values, &mut device_out, policy)?;
        let host = stream
            .memcpy_dtov(&device_out)
            .map_err(|e| format!("reduce_max dtoh: {:?}", e))?;
        Ok(host.into_iter().next().unwrap_or(f32::NEG_INFINITY))
    }

    pub fn reduce_max_f32_device(
        stream: Arc<CudaStream>,
        values: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> CoreResult<()> {
        reduce_max_f32_device_with_policy(stream, values, out, NanPolicy::Ignore)
    }

    pub fn reduce_max_f32_device_with_policy(
        stream: Arc<CudaStream>,
        values: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        policy: NanPolicy,
    ) -> CoreResult<()> {
        if values.len() == 0 {
            return Err("reduce_max_f32_device: empty input".into());
        }
        if out.len() == 0 {
            return Err("reduce_max_f32_device: output slice is empty".into());
        }
        let ctx = stream.context().clone();
        let module = load_reduce_module(&ctx)?;
        match policy {
            NanPolicy::Ignore => {
                if first_valid_index(&stream, &module, values)?.is_none() {
                    write_scalar_f32(&stream, out, f32::NAN)?;
                    Ok(())
                } else {
                    reduce_max_device_ignore_nan(&stream, &module, values, out)
                }
            }
            NanPolicy::Propagate => {
                if first_nan_index(&stream, &module, values)?.is_some() {
                    write_scalar_f32(&stream, out, f32::NAN)?;
                    Ok(())
                } else {
                    reduce_max_device_ignore_nan(&stream, &module, values, out)
                }
            }
        }
    }

    pub fn argmax_f32_with_policy(values: &[f32], policy: NanPolicy) -> CoreResult<usize> {
        if values.is_empty() {
            return Err("argmax on empty slice".into());
        }
        match policy {
            NanPolicy::Ignore => {
                if !values.iter().any(|v| !v.is_nan()) {
                    return Ok(0);
                }
            }
            NanPolicy::Propagate => {
                if let Some(idx) = values.iter().position(|v| v.is_nan()) {
                    return Ok(idx);
                }
            }
        }
        let ctx = global_context()?;
        let stream = ctx.default_stream();
        let device_values = stream
            .memcpy_stod(values)
            .map_err(|e| format!("argmax htod: {:?}", e))?;
        let mut device_out = stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| format!("argmax alloc: {:?}", e))?;
        argmax_f32_device_with_policy(stream.clone(), &device_values, &mut device_out, policy)?;
        let host = stream
            .memcpy_dtov(&device_out)
            .map_err(|e| format!("argmax dtoh: {:?}", e))?;
        let best = host.into_iter().next().unwrap_or(-1);
        if best < 0 {
            Ok(0)
        } else {
            Ok(best as usize)
        }
    }

    pub fn argmax_f32_device(
        stream: Arc<CudaStream>,
        values: &CudaSlice<f32>,
        out: &mut CudaSlice<i32>,
    ) -> CoreResult<()> {
        argmax_f32_device_with_policy(stream, values, out, NanPolicy::Ignore)
    }

    pub fn argmax_f32_device_with_policy(
        stream: Arc<CudaStream>,
        values: &CudaSlice<f32>,
        out: &mut CudaSlice<i32>,
        policy: NanPolicy,
    ) -> CoreResult<()> {
        if values.len() == 0 {
            return Err("argmax_f32_device: empty input".into());
        }
        if out.len() == 0 {
            return Err("argmax_f32_device: output slice is empty".into());
        }
        let ctx = stream.context().clone();
        let module = load_reduce_module(&ctx)?;
        match policy {
            NanPolicy::Ignore => {
                if first_valid_index(&stream, &module, values)?.is_none() {
                    write_scalar_i32(&stream, out, 0)?;
                    Ok(())
                } else {
                    argmax_device_ignore_nan(&stream, &module, values, out)
                }
            }
            NanPolicy::Propagate => {
                if let Some(idx) = first_nan_index(&stream, &module, values)? {
                    let idx_i32 = usize_to_i32("argmax nan index", idx)?;
                    write_scalar_i32(&stream, out, idx_i32)?;
                    Ok(())
                } else {
                    argmax_device_ignore_nan(&stream, &module, values, out)
                }
            }
        }
    }
}

#[cfg(all(feature = "gpu-rocm", target_os = "linux"))]
mod rocm {
    use super::{MatmulTensorCorePolicy, SumPrecisionPolicy};
    use crate::error;
    use hip_sys::hipblas::*;
    use hip_sys::hiprt::*;
    use once_cell::sync::OnceCell;
    use std::ffi::CStr;
    use std::mem::size_of;
    use std::os::raw::c_int;
    use std::ptr;

    type RocmResult<T> = std::result::Result<T, String>;

    struct RocmContext {
        device_id: c_int,
        stream: hipStream_t,
        handle: hipblasHandle_t,
    }

    impl Drop for RocmContext {
        fn drop(&mut self) {
            unsafe {
                hipblasDestroy(self.handle);
                hipStreamDestroy(self.stream);
            }
        }
    }

    static CONTEXT: OnceCell<RocmResult<RocmContext>> = OnceCell::new();

    pub fn is_available() -> bool {
        global_context().is_ok()
    }

    pub fn matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> RocmResult<Vec<f32>> {
        matmul_f32_with_policy(a, b, m, k, n, MatmulTensorCorePolicy::Accuracy)
    }

    pub fn matmul_f32_with_policy(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        policy: MatmulTensorCorePolicy,
    ) -> RocmResult<Vec<f32>> {
        gemm(a, b, m, k, n, false, false, policy)
    }

    pub fn matmul_f32_ex(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> RocmResult<Vec<f32>> {
        matmul_f32_ex_with_policy(
            a,
            b,
            m,
            k,
            n,
            trans_a,
            trans_b,
            MatmulTensorCorePolicy::Accuracy,
        )
    }

    pub fn matmul_f32_ex_with_policy(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
        policy: MatmulTensorCorePolicy,
    ) -> RocmResult<Vec<f32>> {
        gemm(a, b, m, k, n, trans_a, trans_b, policy)
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
    ) -> RocmResult<Vec<f32>> {
        let stride_a = (if trans_a {
            k.saturating_mul(m)
        } else {
            m.saturating_mul(k)
        }) as i64;
        let stride_b = (if trans_b {
            n.saturating_mul(k)
        } else {
            k.saturating_mul(n)
        }) as i64;
        let stride_c = (m.saturating_mul(n)) as i64;
        matmul_batched_f32_strided(
            a, b, batch, m, k, n, trans_a, trans_b, stride_a, stride_b, stride_c,
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
    ) -> RocmResult<Vec<f32>> {
        if stride_a <= 0 || stride_b <= 0 || stride_c == 0 {
            return Err("batched matmul: strides must be non-zero".into());
        }
        if batch == 0 || m == 0 || n == 0 || k == 0 {
            return Ok(vec![0.0; batch.saturating_mul(m.saturating_mul(n))]);
        }
        let ctx = global_context()?;
        hip_check(unsafe { hipSetDevice(ctx.device_id) }, "hipSetDevice")?;

        let (a_rows, a_cols) = if trans_a { (k, m) } else { (m, k) };
        let (b_rows, b_cols) = if trans_b { (n, k) } else { (k, n) };

        let a_col = to_column_major_batched_strided(a, a_rows, a_cols, batch, stride_a)?;
        let b_col = to_column_major_batched_strided(b, b_rows, b_cols, batch, stride_b)?;

        let output_stride = stride_c.unsigned_abs() as usize;
        let per_c = m
            .checked_mul(n)
            .ok_or_else(|| "batched matmul: size overflow".to_string())?;
        let required_c = output_stride
            .checked_mul(batch.saturating_sub(1))
            .and_then(|offset| offset.checked_add(per_c))
            .ok_or_else(|| "batched matmul: stride_c overflow".to_string())?;

        let d_a = DeviceBuffer::from_slice(&a_col)?;
        let d_b = DeviceBuffer::from_slice(&b_col)?;
        let d_c = DeviceBuffer::alloc(size_of::<f32>() * per_c * batch)?;

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let op_a = if trans_a {
            hipblasOperation_t::HIPBLAS_OP_T
        } else {
            hipblasOperation_t::HIPBLAS_OP_N
        };
        let op_b = if trans_b {
            hipblasOperation_t::HIPBLAS_OP_T
        } else {
            hipblasOperation_t::HIPBLAS_OP_N
        };
        let stride_a_col = (a_rows as hipblasStride) * (a_cols as hipblasStride);
        let stride_b_col = (b_rows as hipblasStride) * (b_cols as hipblasStride);
        let stride_c_col = (m as hipblasStride) * (n as hipblasStride);

        hipblas_check(
            unsafe {
                hipblasSgemmStridedBatched(
                    ctx.handle,
                    op_a,
                    op_b,
                    m as c_int,
                    n as c_int,
                    k as c_int,
                    &alpha as *const f32,
                    d_a.as_ptr::<f32>(),
                    a_rows as c_int,
                    stride_a_col,
                    d_b.as_ptr::<f32>(),
                    b_rows as c_int,
                    stride_b_col,
                    &beta as *const f32,
                    d_c.as_mut_ptr::<f32>(),
                    m as c_int,
                    stride_c_col,
                    batch as c_int,
                )
            },
            "hipblasSgemmStridedBatched",
        )?;

        hip_check(
            unsafe { hipStreamSynchronize(ctx.stream) },
            "hipStreamSynchronize",
        )?;

        let mut dense_c = vec![0.0f32; per_c * batch];
        if per_c > 0 {
            hip_check(
                unsafe {
                    hipMemcpy(
                        dense_c.as_mut_ptr() as *mut _,
                        d_c.ptr,
                        size_of::<f32>() * per_c * batch,
                        hipMemcpyKind::hipMemcpyDeviceToHost,
                    )
                },
                "hipMemcpy(C batched)",
            )?;
        }

        let dense_row_major = from_column_major_batched(&dense_c, m, n, batch);

        if stride_c == per_c as i64 {
            return Ok(dense_row_major);
        }

        let mut result = vec![0.0f32; required_c];
        for batch_idx in 0..batch {
            let src = &dense_row_major[batch_idx * per_c..(batch_idx + 1) * per_c];
            let dst_offset = output_stride * batch_idx;
            let dst = &mut result[dst_offset..dst_offset + per_c];
            dst.copy_from_slice(src);
        }

        Ok(result)
    }

    pub fn reduce_sum_f32(values: &[f32]) -> RocmResult<f32> {
        reduce_sum_f32_with_policy(values, SumPrecisionPolicy::Default)
    }

    pub fn reduce_sum_f32_with_policy(
        values: &[f32],
        policy: SumPrecisionPolicy,
    ) -> RocmResult<f32> {
        if !matches!(policy, SumPrecisionPolicy::Default) {
            return Err(
                "ROCm reduce_sum currently supports only SumPrecisionPolicy::Default".into(),
            );
        }
        if values.is_empty() {
            return Ok(0.0);
        }
        let ctx = global_context()?;
        hip_check(unsafe { hipSetDevice(ctx.device_id) }, "hipSetDevice")?;

        let d_values = DeviceBuffer::from_slice(values)?;
        let ones = vec![1.0f32; values.len()];
        let d_ones = DeviceBuffer::from_slice(&ones)?;

        let mut result = 0.0f32;
        hipblas_check(
            unsafe {
                hipblasSdot(
                    ctx.handle,
                    values.len() as c_int,
                    d_values.as_ptr(),
                    1,
                    d_ones.as_ptr(),
                    1,
                    &mut result as *mut f32,
                )
            },
            "hipblasSdot",
        )?;

        hip_check(
            unsafe { hipStreamSynchronize(ctx.stream) },
            "hipStreamSynchronize",
        )?;

        Ok(result)
    }

    fn global_context() -> RocmResult<&'static RocmContext> {
        match CONTEXT.get_or_init(init_context) {
            Ok(ctx) => Ok(ctx),
            Err(err) => Err(err.clone()),
        }
    }

    fn init_context() -> RocmResult<RocmContext> {
        unsafe {
            hip_check(hipInit(0), "hipInit")?;
        }
        let mut count: c_int = 0;
        hip_check(
            unsafe { hipGetDeviceCount(&mut count as *mut c_int) },
            "hipGetDeviceCount",
        )?;
        if count <= 0 {
            return Err("hipGetDeviceCount returned zero devices".into());
        }
        let device_id = 0;
        hip_check(unsafe { hipSetDevice(device_id) }, "hipSetDevice")?;

        let mut stream: hipStream_t = ptr::null_mut();
        hip_check(
            unsafe { hipStreamCreate(&mut stream as *mut hipStream_t) },
            "hipStreamCreate",
        )?;

        let mut handle: hipblasHandle_t = ptr::null_mut();
        hipblas_check(
            unsafe { hipblasCreate(&mut handle as *mut hipblasHandle_t) },
            "hipblasCreate",
        )?;
        hipblas_check(
            unsafe { hipblasSetStream(handle, stream) },
            "hipblasSetStream",
        )?;
        hipblas_check(
            unsafe {
                hipblasSetPointerMode(handle, hipblasPointerMode_t::HIPBLAS_POINTER_MODE_HOST)
            },
            "hipblasSetPointerMode",
        )?;

        Ok(RocmContext {
            device_id,
            stream,
            handle,
        })
    }

    fn hip_error_to_string(code: hipError_t) -> String {
        unsafe {
            let ptr = hipGetErrorString(code);
            if ptr.is_null() {
                format!("hip error {:?}", code as i32)
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        }
    }

    fn hip_check(code: hipError_t, context: &str) -> RocmResult<()> {
        if code == hipError_t::hipSuccess {
            return Ok(());
        }
        let raw = format!("{context}: {}", hip_error_to_string(code));
        let formatted = match code {
            hipError_t::hipErrorNoDevice
            | hipError_t::hipErrorInvalidDevice
            | hipError_t::hipErrorInvalidDevicePointer
            | hipError_t::hipErrorInsufficientDriver
            | hipError_t::hipErrorDeviceAlreadyInUse => error::gpu_unavailable(raw),
            hipError_t::hipErrorInvalidValue => error::numeric_issue(raw),
            _ => error::gpu_error(raw),
        };
        Err(formatted)
    }

    fn hipblas_status_to_string(status: hipblasStatus_t) -> String {
        unsafe {
            let ptr = hipblasStatusToString(status);
            if ptr.is_null() {
                format!("hipblas status {:?}", status as i32)
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        }
    }

    fn hipblas_check(status: hipblasStatus_t, context: &str) -> RocmResult<()> {
        if status == hipblasStatus_t::HIPBLAS_STATUS_SUCCESS {
            return Ok(());
        }
        let raw = format!("{context}: {}", hipblas_status_to_string(status));
        let formatted = match status {
            hipblasStatus_t::HIPBLAS_STATUS_NOT_SUPPORTED => error::gpu_not_implemented(raw),
            hipblasStatus_t::HIPBLAS_STATUS_ALLOC_FAILED
            | hipblasStatus_t::HIPBLAS_STATUS_INTERNAL_ERROR => error::gpu_error(raw),
            hipblasStatus_t::HIPBLAS_STATUS_INVALID_VALUE => error::numeric_issue(raw),
            hipblasStatus_t::HIPBLAS_STATUS_EXECUTION_FAILED => error::gpu_error(raw),
            _ => error::gpu_error(raw),
        };
        Err(formatted)
    }

    struct DeviceBuffer {
        ptr: *mut ::libc::c_void,
        _size: usize,
    }

    impl DeviceBuffer {
        fn alloc(size: usize) -> RocmResult<Self> {
            if size == 0 {
                return Ok(Self {
                    ptr: ptr::null_mut(),
                    _size: 0,
                });
            }
            let mut ptr = ptr::null_mut();
            hip_check(
                unsafe { hipMalloc(&mut ptr as *mut *mut ::libc::c_void, size) },
                "hipMalloc",
            )?;
            Ok(Self { ptr, _size: size })
        }

        fn from_slice(data: &[f32]) -> RocmResult<Self> {
            let size = size_of::<f32>() * data.len();
            let buffer = Self::alloc(size)?;
            if size > 0 {
                hip_check(
                    unsafe {
                        hipMemcpy(
                            buffer.ptr,
                            data.as_ptr() as *const _,
                            size,
                            hipMemcpyKind::hipMemcpyHostToDevice,
                        )
                    },
                    "hipMemcpy(host->device)",
                )?;
            }
            Ok(buffer)
        }

        fn as_ptr<T>(&self) -> *const T {
            self.ptr as *const T
        }

        fn as_mut_ptr<T>(&self) -> *mut T {
            self.ptr as *mut T
        }
    }

    impl Drop for DeviceBuffer {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe {
                    hipFree(self.ptr);
                }
            }
        }
    }

    fn gemm(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
        policy: MatmulTensorCorePolicy,
    ) -> RocmResult<Vec<f32>> {
        let effective_policy = match policy {
            MatmulTensorCorePolicy::Accuracy => MatmulTensorCorePolicy::Accuracy,
            MatmulTensorCorePolicy::Performance
            | MatmulTensorCorePolicy::Float16
            | MatmulTensorCorePolicy::BFloat16 => {
                if cfg!(debug_assertions) {
                    eprintln!(
                        "[num-rs][rocm] policy {:?} not natively supported; executing FP32 GEMM fallback",
                        policy
                    );
                }
                MatmulTensorCorePolicy::Accuracy
            }
        };

        let (a_rows, a_cols) = if trans_a { (k, m) } else { (m, k) };
        let (b_rows, b_cols) = if trans_b { (n, k) } else { (k, n) };

        let expected_a = a_rows
            .checked_mul(a_cols)
            .ok_or_else(|| "gemm: A size overflow".to_string())?;
        let expected_b = b_rows
            .checked_mul(b_cols)
            .ok_or_else(|| "gemm: B size overflow".to_string())?;
        if a.len() != expected_a || b.len() != expected_b {
            return Err("gemm: shape mismatch".into());
        }

        let total = m
            .checked_mul(n)
            .ok_or_else(|| "gemm: result size overflow".to_string())?;
        if total == 0 {
            return Ok(Vec::new());
        }

        let ctx = global_context()?;
        hip_check(unsafe { hipSetDevice(ctx.device_id) }, "hipSetDevice")?;

        let mut a_col = vec![0.0f32; expected_a];
        let mut b_col = vec![0.0f32; expected_b];
        convert_row_major_to_column_major_into(a, a_rows, a_cols, &mut a_col);
        convert_row_major_to_column_major_into(b, b_rows, b_cols, &mut b_col);

        let d_a = DeviceBuffer::from_slice(&a_col)?;
        let d_b = DeviceBuffer::from_slice(&b_col)?;
        let d_c = DeviceBuffer::alloc(size_of::<f32>() * total)?;

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let op_a = if trans_a {
            hipblasOperation_t::HIPBLAS_OP_T
        } else {
            hipblasOperation_t::HIPBLAS_OP_N
        };
        let op_b = if trans_b {
            hipblasOperation_t::HIPBLAS_OP_T
        } else {
            hipblasOperation_t::HIPBLAS_OP_N
        };

        hipblas_check(
            unsafe {
                hipblasSgemm(
                    ctx.handle,
                    op_a,
                    op_b,
                    m as c_int,
                    n as c_int,
                    k as c_int,
                    &alpha as *const f32,
                    d_a.as_ptr::<f32>(),
                    a_rows as c_int,
                    d_b.as_ptr::<f32>(),
                    b_rows as c_int,
                    &beta as *const f32,
                    d_c.as_mut_ptr::<f32>(),
                    m as c_int,
                )
            },
            match effective_policy {
                MatmulTensorCorePolicy::Accuracy => "hipblasSgemm",
                MatmulTensorCorePolicy::Performance
                | MatmulTensorCorePolicy::Float16
                | MatmulTensorCorePolicy::BFloat16 => "hipblasSgemm (fallback)",
            },
        )?;

        hip_check(
            unsafe { hipStreamSynchronize(ctx.stream) },
            "hipStreamSynchronize",
        )?;

        let mut c_col = vec![0.0f32; total];
        if total > 0 {
            hip_check(
                unsafe {
                    hipMemcpy(
                        c_col.as_mut_ptr() as *mut _,
                        d_c.ptr,
                        size_of::<f32>() * total,
                        hipMemcpyKind::hipMemcpyDeviceToHost,
                    )
                },
                "hipMemcpy(C)",
            )?;
        }

        Ok(from_column_major(&c_col, m, n))
    }

    fn to_column_major(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        convert_row_major_to_column_major_into(data, rows, cols, &mut out);
        out
    }

    fn from_column_major(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        convert_column_major_to_row_major_into(data, rows, cols, &mut out);
        out
    }

    fn to_column_major_batched_strided(
        data: &[f32],
        rows: usize,
        cols: usize,
        batch: usize,
        stride: i64,
    ) -> RocmResult<Vec<f32>> {
        if stride <= 0 {
            return Err("batched strided matmul: stride must be positive".into());
        }
        let stride = stride as usize;
        let per_matrix = rows
            .checked_mul(cols)
            .ok_or_else(|| "batched strided matmul: size overflow".to_string())?;
        if per_matrix == 0 {
            return Ok(vec![0.0f32; batch * per_matrix]);
        }
        let required = stride
            .checked_mul(batch.saturating_sub(1))
            .and_then(|offset| offset.checked_add(per_matrix))
            .ok_or_else(|| "batched strided matmul: stride overflow".to_string())?;
        if data.len() < required {
            return Err("batched strided matmul: input shorter than stride requires".into());
        }
        let mut out = vec![0.0f32; batch * per_matrix];
        for b in 0..batch {
            let start = b
                .checked_mul(stride)
                .ok_or_else(|| "batched strided matmul: stride overflow".to_string())?;
            let src = &data[start..start + per_matrix];
            let dst = &mut out[b * per_matrix..(b + 1) * per_matrix];
            convert_row_major_to_column_major_into(src, rows, cols, dst);
        }
        Ok(out)
    }

    fn from_column_major_batched(data: &[f32], rows: usize, cols: usize, batch: usize) -> Vec<f32> {
        let per_matrix = rows * cols;
        let mut out = vec![0.0f32; batch * per_matrix];
        for b in 0..batch {
            let src = &data[b * per_matrix..(b + 1) * per_matrix];
            let dst = &mut out[b * per_matrix..(b + 1) * per_matrix];
            convert_column_major_to_row_major_into(src, rows, cols, dst);
        }
        out
    }

    fn convert_row_major_to_column_major_into(
        data: &[f32],
        rows: usize,
        cols: usize,
        out: &mut [f32],
    ) {
        for row in 0..rows {
            for col in 0..cols {
                out[col * rows + row] = data[row * cols + col];
            }
        }
    }

    fn convert_column_major_to_row_major_into(
        data: &[f32],
        rows: usize,
        cols: usize,
        out: &mut [f32],
    ) {
        for row in 0..rows {
            for col in 0..cols {
                out[row * cols + col] = data[col * rows + row];
            }
        }
    }
}

#[cfg(any(not(feature = "gpu-rocm"), not(target_os = "linux")))]
#[allow(dead_code)]
mod rocm {
    use super::{MatmulTensorCorePolicy, SumPrecisionPolicy};

    type RocmResult<T> = std::result::Result<T, String>;

    pub fn is_available() -> bool {
        false
    }

    pub fn matmul_f32(
        _a: &[f32],
        _b: &[f32],
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> RocmResult<Vec<f32>> {
        Err("ROCm backend is not available on this platform".into())
    }

    pub fn matmul_f32_with_policy(
        _a: &[f32],
        _b: &[f32],
        _m: usize,
        _k: usize,
        _n: usize,
        _policy: MatmulTensorCorePolicy,
    ) -> RocmResult<Vec<f32>> {
        Err("ROCm backend is not available on this platform".into())
    }

    pub fn matmul_f32_ex(
        _a: &[f32],
        _b: &[f32],
        _m: usize,
        _k: usize,
        _n: usize,
        _ta: bool,
        _tb: bool,
    ) -> RocmResult<Vec<f32>> {
        Err("ROCm backend is not available on this platform".into())
    }

    pub fn matmul_f32_ex_with_policy(
        _a: &[f32],
        _b: &[f32],
        _m: usize,
        _k: usize,
        _n: usize,
        _ta: bool,
        _tb: bool,
        _policy: MatmulTensorCorePolicy,
    ) -> RocmResult<Vec<f32>> {
        Err("ROCm backend is not available on this platform".into())
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
    ) -> RocmResult<Vec<f32>> {
        Err("ROCm backend is not available on this platform".into())
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
    ) -> RocmResult<Vec<f32>> {
        Err("ROCm backend is not available on this platform".into())
    }

    pub fn reduce_sum_f32(_values: &[f32]) -> RocmResult<f32> {
        Err("ROCm backend is not available on this platform".into())
    }

    pub fn reduce_sum_f32_with_policy(
        _values: &[f32],
        _policy: SumPrecisionPolicy,
    ) -> RocmResult<f32> {
        Err("ROCm backend is not available on this platform".into())
    }
}

#[cfg(feature = "gpu-cuda")]
pub use cuda::{
    matmul_batched_f32_strided_device, matmul_batched_f32_strided_device_with_policy,
    matmul_f32_ex_device, matmul_f32_ex_device_with_policy,
};

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
        let cases = [(false, false), (true, false), (false, true), (true, true)];
        for (trans_a, trans_b) in cases {
            let gpu =
                super::cuda::matmul_f32_ex(&a, &b, m, k, n, trans_a, trans_b).expect("gpu matmul");
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
        let mut d_c = stream.alloc_zeros::<f32>(m * n).expect("alloc d_c");
        super::matmul_f32_ex_device(stream.clone(), &d_a, &d_b, &mut d_c, m, k, n, false, false)
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
    fn matmul_tensor_policies_match_cpu() {
        if !cuda_available() {
            eprintln!("skipping matmul_tensor_policies_match_cpu: no CUDA device");
            return;
        }
        let m = 3;
        let k = 4;
        let n = 2;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.03).cos()).collect();
        let cpu = super::matmul_cpu_ex(&a, &b, m, k, n, false, false);
        let precise = super::cuda::matmul_f32_ex_with_policy(
            &a,
            &b,
            m,
            k,
            n,
            false,
            false,
            MatmulTensorCorePolicy::Accuracy,
        )
        .expect("accuracy matmul");
        assert_close(&precise, &cpu);

        let fast = super::cuda::matmul_f32_ex_with_policy(
            &a,
            &b,
            m,
            k,
            n,
            false,
            false,
            MatmulTensorCorePolicy::Performance,
        )
        .expect("performance matmul");
        assert_close(&fast, &cpu);

        if let Ok(res) = super::cuda::matmul_f32_ex_with_policy(
            &a,
            &b,
            m,
            k,
            n,
            false,
            false,
            MatmulTensorCorePolicy::Float16,
        ) {
            for (lhs, rhs) in res.iter().zip(cpu.iter()) {
                assert!((lhs - rhs).abs() < 5e-2, "float16 mismatch: {lhs} vs {rhs}");
            }
        } else {
            eprintln!("skipping float16 tensor core check");
        }

        if let Ok(res) = super::cuda::matmul_f32_ex_with_policy(
            &a,
            &b,
            m,
            k,
            n,
            false,
            false,
            MatmulTensorCorePolicy::BFloat16,
        ) {
            for (lhs, rhs) in res.iter().zip(cpu.iter()) {
                assert!(
                    (lhs - rhs).abs() < 5e-2,
                    "bfloat16 mismatch: {lhs} vs {rhs}"
                );
            }
        } else {
            eprintln!("skipping bfloat16 tensor core check");
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
        let res_ignore = super::argmax_f32_with_policy(&values, NanPolicy::Ignore).expect("ignore");
        assert_eq!(res_ignore, 2);
        let res_prop = super::argmax_f32_with_policy(&values, NanPolicy::Propagate).expect("prop");
        assert_eq!(res_prop, 1);
        let nan_only = vec![f32::NAN, f32::NAN];
        let res_ignore =
            super::argmax_f32_with_policy(&nan_only, NanPolicy::Ignore).expect("nan only");
        assert_eq!(res_ignore, 0);
    }
}
