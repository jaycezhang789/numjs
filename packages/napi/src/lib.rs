use napi::bindgen_prelude::*;
use napi_derive::napi;
use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use num_rs_core::{add_inplace, matmul as core_matmul};

#[napi]
pub struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

#[napi]
impl Matrix {
    #[napi(constructor)]
    pub fn new(data: Float64Array, rows: u32, cols: u32) -> Result<Self> {
        let data: Vec<f64> = data.to_vec();
        if data.len() != (rows as usize * cols as usize) {
            return Err(Error::from_reason("Size mismatch"));
        }
        Ok(Self {
            data,
            rows: rows as usize,
            cols: cols as usize,
        })
    }

    #[napi(getter)]
    pub fn rows(&self) -> u32 {
        self.rows as u32
    }
    #[napi(getter)]
    pub fn cols(&self) -> u32 {
        self.cols as u32
    }

    #[napi]
    pub fn to_vec(&self, env: Env) -> Result<Float64Array> {
        let mut arr = Float64Array::new(env, self.data.len() as u32)?;
        arr.copy_from(&self.data)?;
        Ok(arr)
    }
}

fn view2(a: &Matrix) -> ArrayView2<'_, f64> {
    ArrayView2::from_shape((a.rows, a.cols), &a.data).unwrap()
}

fn viewmut2(out: &mut Matrix) -> ArrayViewMut2<'_, f64> {
    ArrayViewMut2::from_shape((out.rows, out.cols), &mut out.data).unwrap()
}

#[napi]
pub fn add(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(Error::from_reason("Shape mismatch"));
    }
    let mut out = Matrix {
        data: vec![0.0; a.rows * a.cols],
        rows: a.rows,
        cols: a.cols,
    };
    add_inplace(view2(a), view2(b), viewmut2(&mut out));
    Ok(out)
}

#[napi]
pub fn matmul(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    if a.cols != b.rows {
        return Err(Error::from_reason("Shape mismatch"));
    }
    let c: Array2<f64> = core_matmul(view2(a), view2(b));
    Ok(Matrix {
        data: c.into_raw_vec(),
        rows: a.rows,
        cols: b.cols,
    })
}
