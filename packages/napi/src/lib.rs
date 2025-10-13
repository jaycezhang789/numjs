
use napi::bindgen_prelude::*;
use napi::JsObject;
use napi_derive::napi;
use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use num_rs_core::{
    add_inplace,
    clip as core_clip,
    concat as core_concat,
    matmul as core_matmul,
    read_npy_matrix,
    stack_matrices as core_stack,
    where_select as core_where,
    write_npy_matrix,
    CoreResult,
};

#[cfg(feature = "linalg")]
use num_rs_core::{
    eigen as core_eigen, qr as core_qr, solve as core_solve, svd as core_svd,
};

#[derive(Clone)]
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
    pub fn to_vec(&self) -> Result<Float64Array> {
        Ok(Float64Array::from(self.data.clone()))
    }
}

impl Matrix {
    fn empty_like(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    fn from_array(array: Array2<f64>) -> Self {
        let (rows, cols) = array.dim();
        Self {
            data: array.into_raw_vec(),
            rows,
            cols,
        }
    }

    fn view(&self) -> ArrayView2<'_, f64> {
        ArrayView2::from_shape((self.rows, self.cols), &self.data).unwrap()
    }

    fn view_mut(&mut self) -> ArrayViewMut2<'_, f64> {
        ArrayViewMut2::from_shape((self.rows, self.cols), &mut self.data).unwrap()
    }
}

fn map_core<T>(res: CoreResult<T>) -> Result<T> {
    res.map_err(|err| Error::from_reason(err))
}

#[napi]
pub fn add(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(Error::from_reason("Shape mismatch"));
    }
    let mut out = Matrix::empty_like(a.rows, a.cols);
    add_inplace(a.view(), b.view(), out.view_mut());
    Ok(out)
}

#[napi]
pub fn matmul(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    if a.cols != b.rows {
        return Err(Error::from_reason("Shape mismatch"));
    }
    let c: Array2<f64> = core_matmul(a.view(), b.view());
    Ok(Matrix::from_array(c))
}

#[napi]
pub fn clip(matrix: &Matrix, min: f64, max: f64) -> Result<Matrix> {
    let clipped = core_clip(matrix.view(), min, max);
    Ok(Matrix::from_array(clipped))
}

#[napi]
pub fn where_select(condition: &Matrix, truthy: &Matrix, falsy: &Matrix) -> Result<Matrix> {
    let result = map_core(core_where(condition.view(), truthy.view(), falsy.view()))?;
    Ok(Matrix::from_array(result))
}

#[napi]
pub fn concat(a: &Matrix, b: &Matrix, axis: u32) -> Result<Matrix> {
    let axis = axis as usize;
    if axis > 1 {
        return Err(Error::from_reason("Axis out of bounds for concat"));
    }
    let views = [a.view(), b.view()];
    let result = map_core(core_concat(axis, &views))?;
    Ok(Matrix::from_array(result))
}

#[napi]
pub fn stack(a: &Matrix, b: &Matrix, axis: u32) -> Result<Matrix> {
    let axis = axis as usize;
    if axis > 1 {
        return Err(Error::from_reason("Axis out of bounds for stack"));
    }
    let views = [a.view(), b.view()];
    let result = map_core(core_stack(axis, &views))?;
    Ok(Matrix::from_array(result))
}

#[cfg(feature = "linalg")]
#[napi]
pub fn svd(env: Env, matrix: &Matrix) -> Result<JsObject> {
    let (u, s, vt) = map_core(core_svd(matrix.view()))?;
    let mut obj = env.create_object()?;
    obj.set("u", Matrix::from_array(u))?;
    obj.set("s", Float64Array::from(s.to_vec()))?;
    obj.set("vt", Matrix::from_array(vt))?;
    Ok(obj)
}

#[cfg(feature = "linalg")]
#[napi]
pub fn qr(env: Env, matrix: &Matrix) -> Result<JsObject> {
    let (q, r) = map_core(core_qr(matrix.view()))?;
    let mut obj = env.create_object()?;
    obj.set("q", Matrix::from_array(q))?;
    obj.set("r", Matrix::from_array(r))?;
    Ok(obj)
}

#[cfg(feature = "linalg")]
#[napi]
pub fn solve(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    if a.rows != a.cols {
        return Err(Error::from_reason("Matrix A must be square"));
    }
    let solution = map_core(core_solve(a.view(), b.view()))?;
    Ok(Matrix::from_array(solution))
}

#[cfg(feature = "linalg")]
#[napi]
pub fn eigen(env: Env, matrix: &Matrix) -> Result<JsObject> {
    let (values, vectors) = map_core(core_eigen(matrix.view()))?;
    let mut obj = env.create_object()?;
    obj.set("values", Float64Array::from(values.to_vec()))?;
    obj.set("vectors", Matrix::from_array(vectors))?;
    Ok(obj)
}

#[napi]
pub fn read_npy(buffer: Buffer) -> Result<Matrix> {
    let matrix = map_core(read_npy_matrix(&buffer))?;
    Ok(Matrix::from_array(matrix))
}

#[napi]
pub fn write_npy(matrix: &Matrix) -> Result<Buffer> {
    let bytes = map_core(write_npy_matrix(matrix.view()))?;
    Ok(Buffer::from(bytes))
}
