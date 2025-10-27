#![cfg_attr(not(feature = "gpu-cuda"), allow(unused_variables, dead_code))]

#[cfg(feature = "gpu-cuda")]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
#[cfg(feature = "gpu-cuda")]
use num_rs_core::gpu;

#[cfg(feature = "gpu-cuda")]
fn gpu_available() -> bool {
    gpu::backend_name() == Some("cuda")
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
fn bench_reduce(c: &mut Criterion) {
    if !gpu_available() {
        eprintln!("[bench] skipping GPU reduce benchmarks: CUDA unavailable");
        return;
    }
    let len = 1 << 20;
    let data: Vec<f32> = (0..len).map(|i| (i as f32).sin()).collect();
    gpu::reduce_sum_f32(&data).expect("warmup gpu reduce");
    c.bench_with_input(BenchmarkId::new("reduce_sum", "gpu"), &data, |b, input| {
        b.iter(|| {
            let _ = gpu::reduce_sum_f32(black_box(input)).unwrap();
        })
    });
    c.bench_with_input(BenchmarkId::new("reduce_sum", "cpu"), &data, |b, input| {
        b.iter(|| {
            let _ = black_box(input).iter().copied().sum::<f32>();
        })
    });
}

#[cfg(feature = "gpu-cuda")]
fn bench_matmul(c: &mut Criterion) {
    if !gpu_available() {
        eprintln!("[bench] skipping GPU matmul benchmarks: CUDA unavailable");
        return;
    }
    let (m, k, n) = (128, 128, 128);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.02).cos()).collect();
    gpu::matmul_f32(&a, &b, m, k, n).expect("warmup gpu matmul");
    c.bench_function("matmul_gpu", |bch| {
        bch.iter(|| {
            let _ = gpu::matmul_f32(black_box(&a), black_box(&b), m, k, n).unwrap();
        })
    });
    c.bench_function("matmul_cpu", |bch| {
        bch.iter(|| {
            let _ = cpu_matmul(black_box(&a), black_box(&b), m, k, n);
        })
    });
}

#[cfg(feature = "gpu-cuda")]
criterion_group!(gpu_vs_cpu, bench_reduce, bench_matmul);
#[cfg(feature = "gpu-cuda")]
criterion_main!(gpu_vs_cpu);

#[cfg(not(feature = "gpu-cuda"))]
fn main() {
    eprintln!("skipping gpu_vs_cpu benchmarks: build without gpu-cuda feature");
}
