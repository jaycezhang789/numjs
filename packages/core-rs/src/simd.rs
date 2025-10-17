use crate::element::Element as ElementTrait;
use num_traits::{Float, Zero};
use std::ops::AddAssign;

pub trait SimdFloat:
    Float + ElementTrait + Copy + Zero + AddAssign + std::ops::Mul<Output = Self> + Send + Sync
{
    fn vec_add(lhs: &[Self], rhs: &[Self], out: &mut [Self]);
    fn vec_mul(lhs: &[Self], rhs: &[Self], out: &mut [Self]);
    fn reduce_sum(values: &[Self]) -> Self;
    fn mul_add_segment(acc: &mut [Self], rhs: &[Self], scalar: Self, start: usize, end: usize);
}

fn fallback_vec_add<T>(lhs: &[T], rhs: &[T], out: &mut [T])
where
    T: Copy + AddAssign + std::ops::Add<Output = T>,
{
    for ((dest, &x), &y) in out.iter_mut().zip(lhs.iter()).zip(rhs.iter()) {
        *dest = x + y;
    }
}

fn fallback_vec_mul<T>(lhs: &[T], rhs: &[T], out: &mut [T])
where
    T: Copy + AddAssign + std::ops::Mul<Output = T>,
{
    for ((dest, &x), &y) in out.iter_mut().zip(lhs.iter()).zip(rhs.iter()) {
        *dest = x * y;
    }
}

fn fallback_reduce_sum<T>(values: &[T]) -> T
where
    T: Float,
{
    let mut sum = T::zero();
    let mut compensation = T::zero();
    for &value in values {
        let y = value - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    sum
}

fn fallback_mul_add_segment<T>(acc: &mut [T], rhs: &[T], scalar: T, start: usize, end: usize)
where
    T: Copy + AddAssign + std::ops::Mul<Output = T>,
{
    for idx in start..end {
        acc[idx] += scalar * rhs[idx];
    }
}

impl SimdFloat for f32 {
    fn vec_add(lhs: &[Self], rhs: &[Self], out: &mut [Self]) {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            unsafe {
                simd128::vec_add_f32(lhs, rhs, out);
                return;
            }
        }
        fallback_vec_add(lhs, rhs, out);
    }

    fn vec_mul(lhs: &[Self], rhs: &[Self], out: &mut [Self]) {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            unsafe {
                simd128::vec_mul_f32(lhs, rhs, out);
                return;
            }
        }
        fallback_vec_mul(lhs, rhs, out);
    }

    fn reduce_sum(values: &[Self]) -> Self {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            unsafe {
                return simd128::kahan_sum_f32(values);
            }
        }
        fallback_reduce_sum(values)
    }

    fn mul_add_segment(acc: &mut [Self], rhs: &[Self], scalar: Self, start: usize, end: usize) {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            unsafe {
                simd128::mul_add_f32(acc, rhs, scalar, start, end);
                return;
            }
        }
        fallback_mul_add_segment(acc, rhs, scalar, start, end);
    }
}

impl SimdFloat for f64 {
    fn vec_add(lhs: &[Self], rhs: &[Self], out: &mut [Self]) {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            unsafe {
                simd128::vec_add_f64(lhs, rhs, out);
                return;
            }
        }
        fallback_vec_add(lhs, rhs, out);
    }

    fn vec_mul(lhs: &[Self], rhs: &[Self], out: &mut [Self]) {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            unsafe {
                simd128::vec_mul_f64(lhs, rhs, out);
                return;
            }
        }
        fallback_vec_mul(lhs, rhs, out);
    }

    fn reduce_sum(values: &[Self]) -> Self {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            unsafe {
                return simd128::kahan_sum_f64(values);
            }
        }
        fallback_reduce_sum(values)
    }

    fn mul_add_segment(acc: &mut [Self], rhs: &[Self], scalar: Self, start: usize, end: usize) {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            unsafe {
                simd128::mul_add_f64(acc, rhs, scalar, start, end);
                return;
            }
        }
        fallback_mul_add_segment(acc, rhs, scalar, start, end);
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd128 {
    use core::arch::wasm32::*;

    #[inline]
    pub unsafe fn vec_add_f32(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
        let len = lhs.len().min(rhs.len()).min(out.len());
        let mut i = 0;
        while i + 4 <= len {
            let a = v128_load(lhs.as_ptr().add(i) as *const v128);
            let b = v128_load(rhs.as_ptr().add(i) as *const v128);
            let sum = f32x4_add(a, b);
            v128_store(out.as_mut_ptr().add(i) as *mut v128, sum);
            i += 4;
        }
        while i < len {
            out[i] = lhs[i] + rhs[i];
            i += 1;
        }
    }

    #[inline]
    pub unsafe fn vec_mul_f32(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
        let len = lhs.len().min(rhs.len()).min(out.len());
        let mut i = 0;
        while i + 4 <= len {
            let a = v128_load(lhs.as_ptr().add(i) as *const v128);
            let b = v128_load(rhs.as_ptr().add(i) as *const v128);
            let prod = f32x4_mul(a, b);
            v128_store(out.as_mut_ptr().add(i) as *mut v128, prod);
            i += 4;
        }
        while i < len {
            out[i] = lhs[i] * rhs[i];
            i += 1;
        }
    }

    #[inline]
    pub unsafe fn mul_add_f32(acc: &mut [f32], rhs: &[f32], scalar: f32, start: usize, end: usize) {
        let len = acc.len().min(rhs.len());
        if start >= len {
            return;
        }
        let mut i = start;
        let end = end.min(len);
        let scalar_vec = f32x4_splat(scalar);
        while i + 4 <= end {
            let acc_ptr = acc.as_mut_ptr().add(i) as *mut v128;
            let acc_val = v128_load(acc_ptr as *const v128);
            let rhs_val = v128_load(rhs.as_ptr().add(i) as *const v128);
            let prod = f32x4_mul(rhs_val, scalar_vec);
            let sum = f32x4_add(acc_val, prod);
            v128_store(acc_ptr, sum);
            i += 4;
        }
        while i < end {
            acc[i] += scalar * rhs[i];
            i += 1;
        }
    }

    #[inline]
    pub unsafe fn vec_add_f64(lhs: &[f64], rhs: &[f64], out: &mut [f64]) {
        let len = lhs.len().min(rhs.len()).min(out.len());
        let mut i = 0;
        while i + 2 <= len {
            let a = v128_load(lhs.as_ptr().add(i) as *const v128);
            let b = v128_load(rhs.as_ptr().add(i) as *const v128);
            let sum = f64x2_add(a, b);
            v128_store(out.as_mut_ptr().add(i) as *mut v128, sum);
            i += 2;
        }
        while i < len {
            out[i] = lhs[i] + rhs[i];
            i += 1;
        }
    }

    #[inline]
    pub unsafe fn vec_mul_f64(lhs: &[f64], rhs: &[f64], out: &mut [f64]) {
        let len = lhs.len().min(rhs.len()).min(out.len());
        let mut i = 0;
        while i + 2 <= len {
            let a = v128_load(lhs.as_ptr().add(i) as *const v128);
            let b = v128_load(rhs.as_ptr().add(i) as *const v128);
            let prod = f64x2_mul(a, b);
            v128_store(out.as_mut_ptr().add(i) as *mut v128, prod);
            i += 2;
        }
        while i < len {
            out[i] = lhs[i] * rhs[i];
            i += 1;
        }
    }

    #[inline]
    pub unsafe fn mul_add_f64(acc: &mut [f64], rhs: &[f64], scalar: f64, start: usize, end: usize) {
        let len = acc.len().min(rhs.len());
        if start >= len {
            return;
        }
        let mut i = start;
        let end = end.min(len);
        let scalar_vec = f64x2_splat(scalar);
        while i + 2 <= end {
            let acc_ptr = acc.as_mut_ptr().add(i) as *mut v128;
            let acc_val = v128_load(acc_ptr as *const v128);
            let rhs_val = v128_load(rhs.as_ptr().add(i) as *const v128);
            let prod = f64x2_mul(rhs_val, scalar_vec);
            let sum = f64x2_add(acc_val, prod);
            v128_store(acc_ptr, sum);
            i += 2;
        }
        while i < end {
            acc[i] += scalar * rhs[i];
            i += 1;
        }
    }

    #[inline]
    pub unsafe fn kahan_sum_f32(values: &[f32]) -> f32 {
        let len = values.len();
        let mut sum = 0.0f32;
        let mut compensation = 0.0f32;
        let mut i = 0;
        while i + 4 <= len {
            let chunk = v128_load(values.as_ptr().add(i) as *const v128);
            let mut lanes = [0f32; 4];
            v128_store(lanes.as_mut_ptr() as *mut v128, chunk);
            for &lane in &lanes {
                let y = lane - compensation;
                let t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
            }
            i += 4;
        }
        while i < len {
            let value = values[i];
            let y = value - compensation;
            let t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
            i += 1;
        }
        sum
    }

    #[inline]
    pub unsafe fn kahan_sum_f64(values: &[f64]) -> f64 {
        let len = values.len();
        let mut sum = 0.0f64;
        let mut compensation = 0.0f64;
        let mut i = 0;
        while i + 2 <= len {
            let chunk = v128_load(values.as_ptr().add(i) as *const v128);
            let mut lanes = [0f64; 2];
            v128_store(lanes.as_mut_ptr() as *mut v128, chunk);
            for &lane in &lanes {
                let y = lane - compensation;
                let t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
            }
            i += 2;
        }
        while i < len {
            let value = values[i];
            let y = value - compensation;
            let t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
            i += 1;
        }
        sum
    }
}
