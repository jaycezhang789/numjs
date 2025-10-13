use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis};

pub fn add_inplace(
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
    mut out: ArrayViewMut2<'_, f64>,
) {
    assert_eq!(a.dim(), b.dim());
    assert_eq!(a.dim(), out.dim());
    out.assign(&(&a + &b));
}

pub fn matmul(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Array2<f64> {
    a.dot(&b)
}

pub fn sum_axis0(a: ArrayView2<'_, f64>) -> Array2<f64> {
    a.sum_axis(Axis(0)).insert_axis(Axis(0))
}
