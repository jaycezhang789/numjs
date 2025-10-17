#![cfg(any(feature = "cpu-openblas", feature = "cpu-blis", feature = "cpu-mkl"))]

use crate::buffer::MatrixBuffer;
use crate::dtype::DType;
use crate::error;
use crate::threading;
use crate::CoreResult;
use std::env;
use std::sync::OnceLock;

#[cfg(all(feature = "cpu-openblas", feature = "cpu-blis"))]
compile_error!("Enable only one of cpu-openblas or cpu-blis at a time");

#[cfg(all(feature = "cpu-openblas", feature = "cpu-mkl"))]
compile_error!("Enable only one of cpu-openblas or cpu-mkl at a time");

#[cfg(all(feature = "cpu-blis", feature = "cpu-mkl"))]
compile_error!("Enable only one of cpu-blis or cpu-mkl at a time");

#[derive(Copy, Clone)]
enum CpuBackend {
    #[cfg(feature = "cpu-openblas")]
    OpenBlas,
    #[cfg(feature = "cpu-blis")]
    Blis,
    #[cfg(feature = "cpu-mkl")]
    Mkl,
}

#[cfg(feature = "cpu-openblas")]
const BACKEND: CpuBackend = CpuBackend::OpenBlas;

#[cfg(feature = "cpu-blis")]
const BACKEND: CpuBackend = CpuBackend::Blis;

#[cfg(feature = "cpu-mkl")]
const BACKEND: CpuBackend = CpuBackend::Mkl;

#[derive(Copy, Clone)]
enum InstructionSet {
    Avx512,
    Avx2,
    Baseline,
}

static INIT_CPU_BACKEND: OnceLock<()> = OnceLock::new();

fn ensure_backend_configured() {
    INIT_CPU_BACKEND.get_or_init(configure_backend);
}

fn configure_backend() {
    threading::ensure_rayon_pool();
    configure_thread_overrides();
    let instruction = detect_instruction_set();
    match BACKEND {
        #[cfg(feature = "cpu-openblas")]
        CpuBackend::OpenBlas => configure_openblas(instruction),
        #[cfg(feature = "cpu-blis")]
        CpuBackend::Blis => configure_blis(instruction),
        #[cfg(feature = "cpu-mkl")]
        CpuBackend::Mkl => configure_mkl(instruction),
    }
}

fn configure_thread_overrides() {
    if let Some(threads) = threading::thread_override() {
        #[cfg(feature = "cpu-openblas")]
        if env::var("OPENBLAS_NUM_THREADS").is_err() {
            env::set_var("OPENBLAS_NUM_THREADS", threads.to_string());
        }
        #[cfg(feature = "cpu-blis")]
        if env::var("BLIS_NUM_THREADS").is_err() {
            env::set_var("BLIS_NUM_THREADS", threads.to_string());
        }
        #[cfg(feature = "cpu-mkl")]
        if env::var("MKL_NUM_THREADS").is_err() {
            env::set_var("MKL_NUM_THREADS", threads.to_string());
        }
    }
}

fn detect_instruction_set() -> InstructionSet {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            InstructionSet::Avx512
        } else if std::arch::is_x86_feature_detected!("avx2") {
            InstructionSet::Avx2
        } else {
            InstructionSet::Baseline
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        InstructionSet::Baseline
    }
}

#[cfg(feature = "cpu-openblas")]
fn configure_openblas(instruction: InstructionSet) {
    if env::var("OPENBLAS_CORETYPE").is_ok() {
        return;
    }
    let value = match instruction {
        InstructionSet::Avx512 => Some("SKYLAKEX"),
        InstructionSet::Avx2 => Some("HASWELL"),
        InstructionSet::Baseline => None,
    };
    if let Some(val) = value {
        env::set_var("OPENBLAS_CORETYPE", val);
    }
}

#[cfg(feature = "cpu-blis")]
fn configure_blis(instruction: InstructionSet) {
    if env::var("BLIS_ARCH").is_ok() {
        return;
    }
    let value = match instruction {
        InstructionSet::Avx512 => Some("skx"),
        InstructionSet::Avx2 => Some("haswell"),
        InstructionSet::Baseline => None,
    };
    if let Some(val) = value {
        env::set_var("BLIS_ARCH", val);
    }
}

#[cfg(feature = "cpu-mkl")]
fn configure_mkl(instruction: InstructionSet) {
    if env::var("MKL_ENABLE_INSTRUCTIONS").is_ok() {
        return;
    }
    let value = match instruction {
        InstructionSet::Avx512 => Some("AVX512"),
        InstructionSet::Avx2 => Some("AVX2"),
        InstructionSet::Baseline => None,
    };
    if let Some(val) = value {
        env::set_var("MKL_ENABLE_INSTRUCTIONS", val);
    }
}

pub fn try_matmul(a: &MatrixBuffer, b: &MatrixBuffer) -> Option<CoreResult<MatrixBuffer>> {
    match (a.dtype(), b.dtype()) {
        (DType::Float32, DType::Float32) => Some(matmul_f32(a, b)),
        (DType::Float64, DType::Float64) => Some(matmul_f64(a, b)),
        _ => None,
    }
}

fn matmul_f32(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    ensure_backend_configured();
    let lhs = a.to_contiguous()?;
    let rhs = b.to_contiguous()?;
    let m = lhs.rows();
    let k = lhs.cols();
    let k_rhs = rhs.rows();
    let n = rhs.cols();
    if k != k_rhs {
        return Err(error::shape_mismatch(
            "matmul: inner dimensions do not match",
        ));
    }
    let lhs_slice = lhs.try_as_slice::<f32>()?;
    let rhs_slice = rhs.try_as_slice::<f32>()?;
    let mut out = vec![0f32; m * n];
    unsafe {
        cblas::sgemm(
            cblas::Layout::RowMajor,
            cblas::Transpose::None,
            cblas::Transpose::None,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            lhs_slice.as_ptr(),
            k as i32,
            rhs_slice.as_ptr(),
            n as i32,
            0.0,
            out.as_mut_ptr(),
            n as i32,
        );
    }
    MatrixBuffer::from_vec(out, m, n).map_err(Into::into)
}

fn matmul_f64(a: &MatrixBuffer, b: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    ensure_backend_configured();
    let lhs = a.to_contiguous()?;
    let rhs = b.to_contiguous()?;
    let m = lhs.rows();
    let k = lhs.cols();
    let k_rhs = rhs.rows();
    let n = rhs.cols();
    if k != k_rhs {
        return Err(error::shape_mismatch(
            "matmul: inner dimensions do not match",
        ));
    }
    let lhs_slice = lhs.try_as_slice::<f64>()?;
    let rhs_slice = rhs.try_as_slice::<f64>()?;
    let mut out = vec![0f64; m * n];
    unsafe {
        cblas::dgemm(
            cblas::Layout::RowMajor,
            cblas::Transpose::None,
            cblas::Transpose::None,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            lhs_slice.as_ptr(),
            k as i32,
            rhs_slice.as_ptr(),
            n as i32,
            0.0,
            out.as_mut_ptr(),
            n as i32,
        );
    }
    MatrixBuffer::from_vec(out, m, n).map_err(Into::into)
}
