#![cfg_attr(not(feature = "gpu-cuda"), allow(unused_variables, dead_code))]

#[cfg(feature = "gpu-cuda")]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
#[cfg(feature = "gpu-cuda")]
use num_rs_core::gpu;

#[cfg(feature = "gpu-cuda")]
fn gpu_available() -> bool {
    match gpu::backend_name() {
        Some("cuda") => true,
        Some(other) => {
            eprintln!("[bench] GPU backend detected but unsupported for these tests: {other}");
            false
        }
        None => false,
    }
}

#[cfg(feature = "gpu-cuda")]
fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            let row = i * n;
            let col = p * n;
            for j in 0..n {
                out[row + j] += a_val * b[col + j];
            }
        }
    }
    out
}

#[cfg(feature = "gpu-cuda")]
fn assert_close(label: &str, lhs: &[f32], rhs: &[f32], tolerance: f32) {
    let mut max_err = 0.0f32;
    for (l, r) in lhs.iter().zip(rhs.iter()) {
        let err = (l - r).abs();
        if err > max_err {
            max_err = err;
        }
    }
    if max_err > tolerance {
        panic!("[bench] {label} verification failed: max error {max_err}, tolerance {tolerance}");
    }
}

#[cfg(feature = "gpu-cuda")]
fn bench_reduce(c: &mut Criterion) {
    if !gpu_available() {
        eprintln!("[bench] skipping GPU reduce benchmarks: CUDA unavailable");
        return;
    }
    let sizes = [1 << 16, 1 << 18, 1 << 20];
    for &len in &sizes {
        let data: Vec<f32> = (0..len).map(|i| (i as f32).sin()).collect();
        let gpu_sum = gpu::reduce_sum_f32(&data).expect("warmup gpu reduce");
        let cpu_sum: f32 = data.iter().copied().sum();
        assert_close("reduce_sum", &[gpu_sum], &[cpu_sum], 1e-4);
        c.bench_with_input(
            BenchmarkId::new("reduce_sum", format!("gpu-{len}")),
            &data,
            |b, input| {
                b.iter(|| {
                    let _ = gpu::reduce_sum_f32(black_box(input)).unwrap();
                })
            },
        );
        c.bench_with_input(
            BenchmarkId::new("reduce_sum", format!("cpu-{len}")),
            &data,
            |b, input| {
                b.iter(|| {
                    let _ = black_box(input).iter().copied().sum::<f32>();
                })
            },
        );
    }
}

#[cfg(feature = "gpu-cuda")]
fn bench_matmul(c: &mut Criterion) {
    if !gpu_available() {
        eprintln!("[bench] skipping GPU matmul benchmarks: CUDA unavailable");
        return;
    }
    let shapes = [(64, 64, 64), (128, 128, 128), (256, 128, 256)];
    for &(m, k, n) in &shapes {
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.02).cos()).collect();
        let gpu_res = gpu::matmul_f32(&a, &b, m, k, n).expect("warmup gpu matmul");
        let cpu_res = cpu_matmul(&a, &b, m, k, n);
        assert_close("matmul", &gpu_res, &cpu_res, 1e-4);
        let dims_label = format!("{m}x{k}x{n}");
        let matmul_input = (a.clone(), b.clone());
        c.bench_with_input(
            BenchmarkId::new("matmul", format!("gpu-{dims_label}")),
            &matmul_input,
            |bch, input| {
                let (lhs, rhs) = (&input.0, &input.1);
                bch.iter(|| {
                    let _ = gpu::matmul_f32(
                        black_box(lhs.as_slice()),
                        black_box(rhs.as_slice()),
                        m,
                        k,
                        n,
                    )
                    .unwrap();
                })
            },
        );
        c.bench_with_input(
            BenchmarkId::new("matmul", format!("cpu-{dims_label}")),
            &matmul_input,
            |bch, input| {
                let (lhs, rhs) = (&input.0, &input.1);
                bch.iter(|| {
                    let _ = cpu_matmul(
                        black_box(lhs.as_slice()),
                        black_box(rhs.as_slice()),
                        m,
                        k,
                        n,
                    );
                })
            },
        );
    }
}

#[cfg(feature = "gpu-cuda")]
criterion_group!(gpu_vs_cpu, bench_reduce, bench_matmul);
#[cfg(feature = "gpu-cuda")]
criterion_main!(gpu_vs_cpu);

#[cfg(not(feature = "gpu-cuda"))]
fn main() {
    eprintln!("skipping gpu_vs_cpu benchmarks: build without gpu-cuda feature");
}
