#![cfg_attr(
    not(any(feature = "gpu-cuda", feature = "gpu-rocm")),
    allow(unused_variables, dead_code)
)]

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
use num_rs_core::{gpu, reset_copy_bytes, take_copy_bytes};
#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
use std::f32::consts::PI;
#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
use std::sync::Arc;

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
fn gpu_backend_kind() -> Option<&'static str> {
    gpu::backend_name()
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
fn gpu_available() -> Option<&'static str> {
    match gpu_backend_kind() {
        Some(kind @ ("cuda" | "rocm")) => Some(kind),
        Some(other) => {
            eprintln!(
                "[bench] detected GPU backend '{other}', skipping because this benchmark supports only CUDA/ROCm"
            );
            None
        }
        None => {
            eprintln!("[bench] no GPU backend detected; skipping GPU benchmarks");
            None
        }
    }
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
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

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
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

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
#[derive(Clone, Copy)]
struct Conv2dSpec {
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    input_h: usize,
    input_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
impl Conv2dSpec {
    fn output_hw(&self) -> (usize, usize) {
        let out_h = (self.input_h + 2 * self.pad_h - self.kernel_h) / self.stride_h + 1;
        let out_w = (self.input_w + 2 * self.pad_w - self.kernel_w) / self.stride_w + 1;
        (out_h, out_w)
    }

    fn kernel_extent(&self) -> usize {
        self.in_channels * self.kernel_h * self.kernel_w
    }

    fn spatial_size(&self) -> usize {
        let (h, w) = self.output_hw();
        h * w
    }
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
fn im2col_single_batch(input: &[f32], spec: Conv2dSpec) -> Vec<f32> {
    debug_assert_eq!(spec.batch, 1, "bench only supports batch=1");
    let (out_h, out_w) = spec.output_hw();
    let cols_per_patch = spec.kernel_extent();
    let mut cols = vec![0.0f32; cols_per_patch * out_h * out_w];
    for oh in 0..out_h {
        for ow in 0..out_w {
            let row_offset = (oh * out_w + ow) * cols_per_patch;
            for ic in 0..spec.in_channels {
                for kh in 0..spec.kernel_h {
                    for kw in 0..spec.kernel_w {
                        let in_y = oh as isize * spec.stride_h as isize + kh as isize
                            - spec.pad_h as isize;
                        let in_x = ow as isize * spec.stride_w as isize + kw as isize
                            - spec.pad_w as isize;
                        let col = ((ic * spec.kernel_h + kh) * spec.kernel_w) + kw;
                        let dst = row_offset + col;
                        if in_y >= 0
                            && in_y < spec.input_h as isize
                            && in_x >= 0
                            && in_x < spec.input_w as isize
                        {
                            let src = ((ic * spec.input_h + in_y as usize) * spec.input_w)
                                + in_x as usize;
                            cols[dst] = input[src];
                        } else {
                            cols[dst] = 0.0;
                        }
                    }
                }
            }
        }
    }
    cols
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
fn reshape_filters(filters: &[f32], spec: Conv2dSpec) -> Vec<f32> {
    let cols_per_patch = spec.kernel_extent();
    let mut mat = vec![0.0f32; cols_per_patch * spec.out_channels];
    for oc in 0..spec.out_channels {
        for ic in 0..spec.in_channels {
            for kh in 0..spec.kernel_h {
                for kw in 0..spec.kernel_w {
                    let col = ((ic * spec.kernel_h + kh) * spec.kernel_w) + kw;
                    let src = ((((oc * spec.in_channels) + ic) * spec.kernel_h + kh)
                        * spec.kernel_w)
                        + kw;
                    mat[col * spec.out_channels + oc] = filters[src];
                }
            }
        }
    }
    mat
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
fn reorder_conv_output(matmul_out: &[f32], spec: Conv2dSpec) -> Vec<f32> {
    let (out_h, out_w) = spec.output_hw();
    let mut out = vec![0.0f32; spec.out_channels * out_h * out_w];
    for pos in 0..spec.spatial_size() {
        let oh = pos / out_w;
        let ow = pos % out_w;
        let row = &matmul_out[pos * spec.out_channels..(pos + 1) * spec.out_channels];
        for oc in 0..spec.out_channels {
            out[(oc * out_h + oh) * out_w + ow] = row[oc];
        }
    }
    out
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
fn gpu_conv2d_via_matmul(
    input: &[f32],
    filters: &[f32],
    spec: Conv2dSpec,
) -> Result<Vec<f32>, String> {
    let cols = im2col_single_batch(input, spec);
    let filt = reshape_filters(filters, spec);
    let raw = gpu::matmul_f32(
        &cols,
        &filt,
        spec.spatial_size(),
        spec.kernel_extent(),
        spec.out_channels,
    )?;
    Ok(reorder_conv_output(&raw, spec))
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
fn conv2d_cpu(input: &[f32], filters: &[f32], spec: Conv2dSpec) -> Vec<f32> {
    let (out_h, out_w) = spec.output_hw();
    let mut out = vec![0.0f32; spec.out_channels * out_h * out_w];
    for oc in 0..spec.out_channels {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut acc = 0.0f32;
                for ic in 0..spec.in_channels {
                    for kh in 0..spec.kernel_h {
                        for kw in 0..spec.kernel_w {
                            let in_y = oh as isize * spec.stride_h as isize + kh as isize
                                - spec.pad_h as isize;
                            let in_x = ow as isize * spec.stride_w as isize + kw as isize
                                - spec.pad_w as isize;
                            if in_y < 0
                                || in_x < 0
                                || in_y >= spec.input_h as isize
                                || in_x >= spec.input_w as isize
                            {
                                continue;
                            }
                            let input_idx = ((ic * spec.input_h + in_y as usize) * spec.input_w)
                                + in_x as usize;
                            let filter_idx = ((((oc * spec.in_channels) + ic) * spec.kernel_h
                                + kh)
                                * spec.kernel_w)
                                + kw;
                            acc += input[input_idx] * filters[filter_idx];
                        }
                    }
                }
                out[(oc * out_h + oh) * out_w + ow] = acc;
            }
        }
    }
    out
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
fn cpu_batched_matmul(
    a: &[f32],
    b: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * m * n];
    for batch_idx in 0..batch {
        let a_offset = batch_idx * m * k;
        let b_offset = batch_idx * k * n;
        let c_offset = batch_idx * m * n;
        for i in 0..m {
            for p in 0..k {
                let a_val = a[a_offset + i * k + p];
                for j in 0..n {
                    out[c_offset + i * n + j] += a_val * b[b_offset + p * n + j];
                }
            }
        }
    }
    out
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
fn build_dft_matrices(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cos_vals = vec![0.0f32; n * n];
    let mut sin_vals = vec![0.0f32; n * n];
    let n_f = n as f32;
    for k in 0..n {
        for t in 0..n {
            let angle = 2.0 * PI * k as f32 * t as f32 / n_f;
            let idx = k * n + t;
            cos_vals[idx] = angle.cos();
            sin_vals[idx] = -angle.sin();
        }
    }
    (cos_vals, sin_vals)
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
fn cpu_dft(signal: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = signal.len();
    let mut real = vec![0.0f32; n];
    let mut imag = vec![0.0f32; n];
    let n_f = n as f32;
    for k in 0..n {
        for t in 0..n {
            let angle = 2.0 * PI * k as f32 * t as f32 / n_f;
            let value = signal[t];
            real[k] += value * angle.cos();
            imag[k] -= value * angle.sin();
        }
    }
    (real, imag)
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
fn gpu_dft_using_matmul(
    signal: &[f32],
    cos: &[f32],
    sin: &[f32],
    n: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let real = gpu::matmul_f32(cos, signal, n, n, 1)?;
    let imag = gpu::matmul_f32(sin, signal, n, n, 1)?;
    Ok((real, imag))
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
#[derive(Clone)]
struct DftFixture {
    n: usize,
    signal: Vec<f32>,
    cos: Arc<Vec<f32>>,
    sin: Arc<Vec<f32>>,
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
fn build_dft_fixture(n: usize) -> DftFixture {
    let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).cos()).collect();
    let (cos_vals, sin_vals) = build_dft_matrices(n);
    DftFixture {
        n,
        signal,
        cos: Arc::new(cos_vals),
        sin: Arc::new(sin_vals),
    }
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
fn bench_reduce(c: &mut Criterion) {
    let Some(backend) = gpu_available() else {
        return;
    };

    let sizes = [1 << 16, 1 << 18, 1 << 20];
    for &len in &sizes {
        let data: Vec<f32> = (0..len).map(|i| (i as f32).sin()).collect();
        let gpu_sum = gpu::reduce_sum_f32(&data).expect("warmup gpu reduce");
        let cpu_sum: f32 = data.iter().copied().sum();
        assert_close("reduce_sum", &[gpu_sum], &[cpu_sum], 1e-4);
        c.bench_with_input(
            BenchmarkId::new("reduce_sum", format!("gpu-{backend}-{len}")),
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

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
fn bench_matmul(c: &mut Criterion) {
    let Some(backend) = gpu_available() else {
        return;
    };

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
            BenchmarkId::new("matmul", format!("gpu-{backend}-{dims_label}")),
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

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
fn bench_conv2d(c: &mut Criterion) {
    let Some(backend) = gpu_available() else {
        return;
    };

    let configs = [
        Conv2dSpec {
            batch: 1,
            in_channels: 16,
            out_channels: 32,
            input_h: 32,
            input_w: 32,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 1,
            pad_w: 1,
        },
        Conv2dSpec {
            batch: 1,
            in_channels: 32,
            out_channels: 32,
            input_h: 64,
            input_w: 64,
            kernel_h: 5,
            kernel_w: 5,
            stride_h: 2,
            stride_w: 2,
            pad_h: 2,
            pad_w: 2,
        },
    ];

    for spec in configs {
        let input_len = spec.in_channels * spec.input_h * spec.input_w;
        let filter_len = spec.out_channels * spec.kernel_extent();
        let input_vec: Vec<f32> = (0..input_len).map(|i| (i as f32 * 0.013).cos()).collect();
        let filter_vec: Vec<f32> = (0..filter_len).map(|i| (i as f32 * 0.019).sin()).collect();
        let gpu_out =
            gpu_conv2d_via_matmul(&input_vec, &filter_vec, spec).expect("warmup gpu conv2d");
        let cpu_out = conv2d_cpu(&input_vec, &filter_vec, spec);
        assert_close("conv2d", &gpu_out, &cpu_out, 1e-3);

        let label = format!(
            "n{}-c{}x{}x{}-k{}-{}x{}-s{}x{}-p{}x{}",
            spec.batch,
            spec.in_channels,
            spec.input_h,
            spec.input_w,
            spec.out_channels,
            spec.kernel_h,
            spec.kernel_w,
            spec.stride_h,
            spec.stride_w,
            spec.pad_h,
            spec.pad_w
        );
        let input = Arc::new(input_vec);
        let filters = Arc::new(filter_vec);
        let gpu_inputs = (Arc::clone(&input), Arc::clone(&filters));
        c.bench_with_input(
            BenchmarkId::new("conv2d", format!("gpu-{backend}-{label}")),
            &gpu_inputs,
            |bch, data| {
                bch.iter(|| {
                    let _ = gpu_conv2d_via_matmul(
                        black_box(data.0.as_slice()),
                        black_box(data.1.as_slice()),
                        spec,
                    )
                    .unwrap();
                })
            },
        );
        let cpu_inputs = (input, filters);
        c.bench_with_input(
            BenchmarkId::new("conv2d", format!("cpu-{label}")),
            &cpu_inputs,
            |bch, data| {
                bch.iter(|| {
                    let _ = conv2d_cpu(
                        black_box(data.0.as_slice()),
                        black_box(data.1.as_slice()),
                        spec,
                    );
                })
            },
        );
    }
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
fn bench_batched_gemm(c: &mut Criterion) {
    let Some(backend) = gpu_available() else {
        return;
    };

    let configs = [(8, 64, 64, 64), (16, 64, 128, 64), (4, 256, 128, 128)];
    for &(batch, m, k, n) in &configs {
        let a_len = batch * m * k;
        let b_len = batch * k * n;
        let a_vec: Vec<f32> = (0..a_len).map(|i| (i as f32 * 0.005).sin()).collect();
        let b_vec: Vec<f32> = (0..b_len).map(|i| (i as f32 * 0.007).cos()).collect();
        let gpu_out = gpu::matmul_batched_f32(&a_vec, &b_vec, batch, m, k, n, false, false)
            .expect("warmup gpu batched gemm");
        let cpu_out = cpu_batched_matmul(&a_vec, &b_vec, batch, m, k, n);
        assert_close("batched_gemm", &gpu_out, &cpu_out, 1e-4);

        let label = format!("batch{batch}-{m}x{k}x{n}");
        let a = Arc::new(a_vec);
        let b = Arc::new(b_vec);
        let gpu_inputs = (Arc::clone(&a), Arc::clone(&b));
        c.bench_with_input(
            BenchmarkId::new("batched_gemm", format!("gpu-{backend}-{label}")),
            &gpu_inputs,
            |bch, data| {
                bch.iter(|| {
                    let _ = gpu::matmul_batched_f32(
                        black_box(data.0.as_slice()),
                        black_box(data.1.as_slice()),
                        batch,
                        m,
                        k,
                        n,
                        false,
                        false,
                    )
                    .unwrap();
                })
            },
        );
        let cpu_inputs = (a, b);
        c.bench_with_input(
            BenchmarkId::new("batched_gemm", format!("cpu-{label}")),
            &cpu_inputs,
            |bch, data| {
                bch.iter(|| {
                    let _ = cpu_batched_matmul(
                        black_box(data.0.as_slice()),
                        black_box(data.1.as_slice()),
                        batch,
                        m,
                        k,
                        n,
                    );
                })
            },
        );
    }
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
fn bench_fft_dft(c: &mut Criterion) {
    let Some(backend) = gpu_available() else {
        return;
    };

    let sizes = [256usize, 512];
    for &n in &sizes {
        let fixture = build_dft_fixture(n);
        let (cpu_real, cpu_imag) = cpu_dft(&fixture.signal);
        let (gpu_real, gpu_imag) = gpu_dft_using_matmul(
            &fixture.signal,
            fixture.cos.as_slice(),
            fixture.sin.as_slice(),
            n,
        )
        .expect("warmup gpu fft");
        assert_close("fft-real", &gpu_real, &cpu_real, 1e-3);
        assert_close("fft-imag", &gpu_imag, &cpu_imag, 1e-3);

        let fixture_gpu = fixture.clone();
        c.bench_with_input(
            BenchmarkId::new("fft_dft", format!("gpu-{backend}-{n}")),
            &fixture_gpu,
            |bch, data| {
                bch.iter(|| {
                    let _ = gpu_dft_using_matmul(
                        black_box(data.signal.as_slice()),
                        black_box(data.cos.as_slice()),
                        black_box(data.sin.as_slice()),
                        data.n,
                    )
                    .unwrap();
                })
            },
        );
        c.bench_with_input(
            BenchmarkId::new("fft_dft", format!("cpu-{n}")),
            &fixture,
            |bch, data| {
                bch.iter(|| {
                    let _ = cpu_dft(black_box(data.signal.as_slice()));
                })
            },
        );
    }
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
fn bench_data_transfer(c: &mut Criterion) {
    let Some(backend) = gpu_available() else {
        return;
    };

    let sizes = [1 << 18, 1 << 20, 1 << 22];
    for &len in &sizes {
        let data: Vec<f32> = (0..len).map(|i| (i as f32 * 0.003).sin()).collect();
        c.bench_with_input(
            BenchmarkId::new("transfer", format!("reduce-h2d-{backend}-{len}")),
            &data,
            |bch, input| {
                bch.iter(|| {
                    reset_copy_bytes();
                    let _ = gpu::reduce_sum_f32(black_box(input)).unwrap();
                    let bytes = take_copy_bytes();
                    black_box(bytes);
                })
            },
        );
        let matmul_data = (
            data.clone(),
            (0..len)
                .map(|i| (i as f32 * 0.001).cos())
                .collect::<Vec<f32>>(),
        );
        let m = 1;
        let k = len;
        let n = 1;
        c.bench_with_input(
            BenchmarkId::new("transfer", format!("matmul-h2d-d2h-{backend}-{len}")),
            &matmul_data,
            |bch, (lhs, rhs)| {
                bch.iter(|| {
                    reset_copy_bytes();
                    let _ = gpu::matmul_f32(
                        black_box(lhs.as_slice()),
                        black_box(rhs.as_slice()),
                        m,
                        k,
                        n,
                    )
                    .unwrap();
                    let bytes = take_copy_bytes();
                    black_box(bytes);
                })
            },
        );
    }
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
criterion_group!(
    gpu_vs_cpu,
    bench_reduce,
    bench_matmul,
    bench_batched_gemm,
    bench_conv2d,
    bench_fft_dft,
    bench_data_transfer
);
#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
criterion_main!(gpu_vs_cpu);

#[cfg(not(any(feature = "gpu-cuda", feature = "gpu-rocm")))]
fn main() {
    eprintln!(
        "skipping gpu_vs_cpu benchmarks: build with `gpu-cuda` or `gpu-rocm` feature to enable"
    );
}
