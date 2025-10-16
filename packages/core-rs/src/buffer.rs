use crate::dtype::DType;
use crate::element::Element;
use crate::metrics::record_copy_bytes;
use std::borrow::Cow;
use std::cmp::{max, min};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatrixBuffer {
    dtype: DType,
    rows: usize,
    cols: usize,
    data: Arc<Vec<u8>>,
    offset: isize,
    row_stride: isize,
    col_stride: isize,
    // Draft: fixed-point scale for DType::Fixed64
    fixed_scale: Option<i32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliceSpec {
    pub start: Option<isize>,
    pub end: Option<isize>,
    pub step: isize,
}

impl SliceSpec {
    pub fn new(start: Option<isize>, end: Option<isize>, step: isize) -> Result<Self, String> {
        if step == 0 {
            return Err("slice step must not be zero".into());
        }
        Ok(SliceSpec { start, end, step })
    }

    pub fn full() -> Self {
        SliceSpec {
            start: None,
            end: None,
            step: 1,
        }
    }
}

impl MatrixBuffer {
    pub fn from_vec<T: Element>(
        mut data: Vec<T>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, String> {
        if rows == 0 || cols == 0 {
            return Err("rows and cols must be greater than zero".into());
        }
        if data.len() != rows * cols {
            return Err("data length does not match shape".into());
        }
        let dtype = T::DTYPE;
        let byte_len = data.len() * dtype.size_of();
        let mut bytes = Vec::<u8>::with_capacity(byte_len);
        let ptr = data.as_mut_ptr() as *const u8;
        unsafe {
            bytes.extend_from_slice(std::slice::from_raw_parts(ptr, byte_len));
        }
        std::mem::forget(data);
        MatrixBuffer::new_internal(dtype, rows, cols, Arc::new(bytes), 0, cols as isize, 1)
    }

    pub fn from_bytes(
        dtype: DType,
        rows: usize,
        cols: usize,
        data: Vec<u8>,
    ) -> Result<Self, String> {
        if rows == 0 || cols == 0 {
            return Err("rows and cols must be greater than zero".into());
        }
        let expected = rows
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(dtype.size_of()))
            .ok_or_else(|| "shape is too large".to_string())?;
        if expected != data.len() {
            return Err("byte length does not match shape".into());
        }
        MatrixBuffer::new_internal(dtype, rows, cols, Arc::new(data), 0, cols as isize, 1)
    }

    fn new_internal(
        dtype: DType,
        rows: usize,
        cols: usize,
        data: Arc<Vec<u8>>,
        offset: isize,
        row_stride: isize,
        col_stride: isize,
    ) -> Result<Self, String> {
        if rows == 0 || cols == 0 {
            return Err("rows and cols must be greater than zero".into());
        }
        if data.len() % dtype.size_of() != 0 {
            return Err("backing buffer is not aligned to dtype width".into());
        }
        validate_view(
            data.len() / dtype.size_of(),
            rows,
            cols,
            offset,
            row_stride,
            col_stride,
        )?;
        Ok(Self {
            dtype,
            rows,
            cols,
            data,
            offset,
            row_stride,
            col_stride,
            fixed_scale: None,
        })
    }

    pub fn from_fixed_i64_vec(
        data: Vec<i64>,
        rows: usize,
        cols: usize,
        scale: i32,
    ) -> Result<Self, String> {
        if rows == 0 || cols == 0 {
            return Err("rows and cols must be greater than zero".into());
        }
        if data.len() != rows * cols {
            return Err("data length does not match shape".into());
        }
        let dtype = DType::Fixed64;
        let byte_len = data.len() * dtype.size_of();
        let mut bytes = Vec::<u8>::with_capacity(byte_len);
        let ptr = data.as_ptr() as *const u8;
        unsafe {
            bytes.extend_from_slice(std::slice::from_raw_parts(ptr, byte_len));
        }
        std::mem::forget(data);
        let mut buf = MatrixBuffer::new_internal(dtype, rows, cols, Arc::new(bytes), 0, cols as isize, 1)?;
        buf.fixed_scale = Some(scale);
        Ok(buf)
    }

    pub fn fixed_scale(&self) -> Option<i32> {
        self.fixed_scale
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    pub fn element_size(&self) -> usize {
        self.dtype.size_of()
    }

    pub fn offset(&self) -> isize {
        self.offset
    }

    pub fn row_stride(&self) -> isize {
        self.row_stride
    }

    pub fn col_stride(&self) -> isize {
        self.col_stride
    }

    pub fn is_standard_layout(&self) -> bool {
        self.col_stride == 1 && self.row_stride == self.cols as isize
    }

    pub fn as_slice<T: Element>(&self) -> Option<&[T]> {
        if T::DTYPE != self.dtype {
            return None;
        }
        if !self.is_standard_layout() {
            return None;
        }
        let byte_offset = (self.offset as isize) * self.element_size() as isize;
        unsafe {
            let ptr = self.data.as_ptr().offset(byte_offset) as *const T;
            Some(std::slice::from_raw_parts(ptr, self.len()))
        }
    }

    pub fn as_byte_slice(&self) -> Option<&[u8]> {
        if !self.is_standard_layout() {
            return None;
        }
        let byte_offset = (self.offset as isize) * self.element_size() as isize;
        let len_bytes = self.len() * self.element_size();
        let start = byte_offset as usize;
        let end = start + len_bytes;
        Some(&self.data.as_ref()[start..end])
    }

    pub fn as_slice_mut<T: Element>(&mut self) -> Option<&mut [T]> {
        if T::DTYPE != self.dtype {
            return None;
        }
        self.ensure_contiguous_mut();
        let len = self.len();
        let ptr = Arc::make_mut(&mut self.data).as_mut_ptr() as *mut T;
        unsafe { Some(std::slice::from_raw_parts_mut(ptr, len)) }
    }

    pub fn to_contiguous(&self) -> Result<Self, String> {
        if self.is_standard_layout() && self.offset == 0 {
            return Ok(self.clone());
        }
        let mut bytes = vec![0u8; self.len() * self.element_size()];
        self.copy_into_bytes(&mut bytes);
        record_copy_bytes(bytes.len());
        MatrixBuffer::from_bytes(self.dtype, self.rows, self.cols, bytes)
    }

    pub fn to_contiguous_bytes_vec(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; self.len() * self.element_size()];
        self.copy_into_bytes(&mut bytes);
        record_copy_bytes(bytes.len());
        bytes
    }

    pub fn append_into(&self, dst: &mut Vec<u8>) {
        let additional = self.len() * self.element_size();
        let start = dst.len();
        dst.resize(start + additional, 0);
        self.copy_into_bytes(&mut dst[start..start + additional]);
    }

    pub fn clone_with_dtype(&self, dtype: DType) -> Result<Self, String> {
        record_copy_bytes(self.len() * dtype.size_of());
        let bytes = self.to_contiguous_bytes_vec();
        MatrixBuffer::from_bytes(dtype, self.rows, self.cols, bytes)
    }

    pub fn reinterpret(&self, target: DType) -> Result<Self, String> {
        if target.size_of() != self.dtype.size_of() {
            return Err("reinterpret: dtype width mismatch".into());
        }
        MatrixBuffer::new_internal(
            target,
            self.rows,
            self.cols,
            Arc::clone(&self.data),
            self.offset,
            self.row_stride,
            self.col_stride,
        )
    }

    pub fn cast(&self, target: DType) -> Result<Self, String> {
        if target == self.dtype {
            return Ok(self.clone());
        }
        record_copy_bytes(self.len() * target.size_of());
        match target {
            DType::Bool => {
                let vec: Vec<bool> = self.to_bool_vec();
                MatrixBuffer::from_vec(vec, self.rows, self.cols)
            }
            DType::Int8 => cast_from_f64(self, i8::MIN as f64, i8::MAX as f64, |v| v.round() as i8),
            DType::Int16 => {
                cast_from_f64(self, i16::MIN as f64, i16::MAX as f64, |v| v.round() as i16)
            }
            DType::Int32 => {
                cast_from_f64(self, i32::MIN as f64, i32::MAX as f64, |v| v.round() as i32)
            }
            DType::Int64 => {
                cast_from_f64(self, i64::MIN as f64, i64::MAX as f64, |v| v.round() as i64)
            }
            DType::UInt8 => cast_from_f64(self, 0.0, u8::MAX as f64, |v| v.round() as u8),
            DType::UInt16 => cast_from_f64(self, 0.0, u16::MAX as f64, |v| v.round() as u16),
            DType::UInt32 => cast_from_f64(self, 0.0, u32::MAX as f64, |v| v.round() as u32),
            DType::UInt64 => cast_from_f64(self, 0.0, u64::MAX as f64, |v| v.round() as u64),
            DType::Float32 => {
                let vec: Vec<f32> = self.to_f64_vec().into_iter().map(|v| v as f32).collect();
                MatrixBuffer::from_vec(vec, self.rows, self.cols)
            }
            DType::Float64 => MatrixBuffer::from_vec(self.to_f64_vec(), self.rows, self.cols),
            DType::Fixed64 => Err("cast: casting to fixed64 not supported; construct with from_fixed_i64_vec".into()),
        }
    }

    pub fn to_f64_vec(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.len());
        match self.dtype {
            DType::Bool => {
                for row in 0..self.rows {
                    for col in 0..self.cols {
                        let value = self.read_typed::<bool>(row, col);
                        out.push(if value { 1.0 } else { 0.0 });
                    }
                }
            }
            DType::Int8 => collect_numeric(&mut out, self, |v: i8| v as f64),
            DType::Int16 => collect_numeric(&mut out, self, |v: i16| v as f64),
            DType::Int32 => collect_numeric(&mut out, self, |v: i32| v as f64),
            DType::Int64 => collect_numeric(&mut out, self, |v: i64| v as f64),
            DType::UInt8 => collect_numeric(&mut out, self, |v: u8| v as f64),
            DType::UInt16 => collect_numeric(&mut out, self, |v: u16| v as f64),
            DType::UInt32 => collect_numeric(&mut out, self, |v: u32| v as f64),
            DType::UInt64 => collect_numeric(&mut out, self, |v: u64| v as f64),
            DType::Float32 => collect_numeric(&mut out, self, |v: f32| v as f64),
            DType::Float64 => collect_numeric(&mut out, self, |v: f64| v),
            DType::Fixed64 => {
                let scale = self.fixed_scale.unwrap_or(0);
                let factor = 10f64.powi(scale as i32);
                let contiguous = self.to_contiguous().expect("contiguous");
                let bytes = contiguous.as_byte_slice().expect("bytes");
                let mut i = 0;
                while i + 8 <= bytes.len() {
                    let v = i64::from_ne_bytes(bytes[i..i + 8].try_into().unwrap());
                    out.push((v as f64) / factor);
                    i += 8;
                }
            }
        }
        out
    }

    pub fn to_bool_vec(&self) -> Vec<bool> {
        match self.dtype {
            DType::Bool => {
                let mut out = Vec::with_capacity(self.len());
                collect_numeric(&mut out, self, |v: bool| v);
                out
            }
            _ => self.to_f64_vec().into_iter().map(|v| v != 0.0).collect(),
        }
    }

    pub fn from_f64_vec(
        dtype: DType,
        rows: usize,
        cols: usize,
        data: Vec<f64>,
    ) -> Result<Self, String> {
        match dtype {
            DType::Bool => {
                let vec: Vec<bool> = data.into_iter().map(|v| v != 0.0).collect();
                Self::from_vec(vec, rows, cols)
            }
            DType::Int8 => {
                cast_vec_to_dtype(data, rows, cols, i8::MIN as f64, i8::MAX as f64, |v| {
                    v.round() as i8
                })
            }
            DType::Int16 => {
                cast_vec_to_dtype(data, rows, cols, i16::MIN as f64, i16::MAX as f64, |v| {
                    v.round() as i16
                })
            }
            DType::Int32 => {
                cast_vec_to_dtype(data, rows, cols, i32::MIN as f64, i32::MAX as f64, |v| {
                    v.round() as i32
                })
            }
            DType::Int64 => {
                cast_vec_to_dtype(data, rows, cols, i64::MIN as f64, i64::MAX as f64, |v| {
                    v.round() as i64
                })
            }
            DType::UInt8 => {
                cast_vec_to_dtype(data, rows, cols, 0.0, u8::MAX as f64, |v| v.round() as u8)
            }
            DType::UInt16 => {
                cast_vec_to_dtype(data, rows, cols, 0.0, u16::MAX as f64, |v| v.round() as u16)
            }
            DType::UInt32 => {
                cast_vec_to_dtype(data, rows, cols, 0.0, u32::MAX as f64, |v| v.round() as u32)
            }
            DType::UInt64 => {
                cast_vec_to_dtype(data, rows, cols, 0.0, u64::MAX as f64, |v| v.round() as u64)
            }
            DType::Float32 => {
                let vec: Vec<f32> = data.into_iter().map(|v| v as f32).collect();
                Self::from_vec(vec, rows, cols)
            }
            DType::Float64 => Self::from_vec(data, rows, cols),
            DType::Fixed64 => Err("from_f64_vec(Fixed64): requires explicit scale; use from_fixed_i64_vec".into()),
        }
    }

    pub fn broadcast_to(&self, rows: usize, cols: usize) -> Result<Self, String> {
        if rows == 0 || cols == 0 {
            return Err("broadcast target shape must be non-zero".into());
        }
        if self.rows == rows && self.cols == cols {
            return Ok(self.clone());
        }
        if !(self.rows == 1 || self.rows == rows) {
            return Err(format!(
                "cannot broadcast rows: source {} target {}",
                self.rows, rows
            ));
        }
        if !(self.cols == 1 || self.cols == cols) {
            return Err(format!(
                "cannot broadcast cols: source {} target {}",
                self.cols, cols
            ));
        }
        let mut view = self.clone();
        view.rows = rows;
        view.cols = cols;
        if self.rows == 1 && rows > 1 {
            view.row_stride = 0;
        }
        if self.cols == 1 && cols > 1 {
            view.col_stride = 0;
        }
        view.validate_bounds()?;
        Ok(view)
    }

    pub fn transpose(&self) -> Result<Self, String> {
        let mut view = self.clone();
        std::mem::swap(&mut view.rows, &mut view.cols);
        std::mem::swap(&mut view.row_stride, &mut view.col_stride);
        view.validate_bounds()?;
        Ok(view)
    }

    pub fn slice(&self, rows: SliceSpec, cols: SliceSpec) -> Result<Self, String> {
        let (row_start, row_len, row_step) = normalize_slice(self.rows, rows)?;
        let (col_start, col_len, col_step) = normalize_slice(self.cols, cols)?;
        let offset = self.offset
            + self.row_stride * row_start as isize
            + self.col_stride * col_start as isize;
        let row_stride = self.row_stride * row_step;
        let col_stride = self.col_stride * col_step;
        MatrixBuffer::new_internal(
            self.dtype,
            row_len,
            col_len,
            Arc::clone(&self.data),
            offset,
            row_stride,
            col_stride,
        )
    }

    pub fn row(&self, index: isize) -> Result<Self, String> {
        let idx = normalize_index(index, self.rows)? as isize;
        MatrixBuffer::new_internal(
            self.dtype,
            1,
            self.cols,
            Arc::clone(&self.data),
            self.offset + idx * self.row_stride,
            self.row_stride,
            self.col_stride,
        )
    }

    pub fn column(&self, index: isize) -> Result<Self, String> {
        let idx = normalize_index(index, self.cols)? as isize;
        MatrixBuffer::new_internal(
            self.dtype,
            self.rows,
            1,
            Arc::clone(&self.data),
            self.offset + idx * self.col_stride,
            self.row_stride,
            self.col_stride,
        )
    }

    pub fn take(&self, axis: usize, indices: &[isize]) -> Result<Self, String> {
        match axis {
            0 => self.take_rows(indices),
            1 => self.take_cols(indices),
            _ => Err("take: axis must be 0 or 1".into()),
        }
    }

    pub fn gather(
        &self,
        row_indices: &[isize],
        col_indices: &[isize],
        pairwise: bool,
    ) -> Result<Self, String> {
        if row_indices.is_empty() {
            return Err("gather: row indices must not be empty".into());
        }
        if pairwise && row_indices.len() != col_indices.len() {
            return Err(
                "gather pairwise requires the same number of row and column indices".into(),
            );
        }
        let elem_size = self.element_size();
        if pairwise {
            let mut data = vec![0u8; row_indices.len() * elem_size];
            for (i, (&row_raw, &col_raw)) in row_indices.iter().zip(col_indices).enumerate() {
                let row = normalize_index(row_raw, self.rows)?;
                let col = normalize_index(col_raw, self.cols)?;
                let src = self.element_ptr(row, col);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src,
                        data.as_mut_ptr().add(i * elem_size),
                        elem_size,
                    );
                }
            }
            MatrixBuffer::from_bytes(self.dtype, row_indices.len(), 1, data)
        } else {
            let cols_out = if col_indices.is_empty() {
                return Err("gather: column indices must not be empty for outer mode".into());
            } else {
                col_indices.len()
            };
            let mut data = vec![0u8; row_indices.len() * cols_out * elem_size];
            for (row_out, &row_raw) in row_indices.iter().enumerate() {
                let row = normalize_index(row_raw, self.rows)?;
                for (col_out, &col_raw) in col_indices.iter().enumerate() {
                    let col = normalize_index(col_raw, self.cols)?;
                    let src = self.element_ptr(row, col);
                    let dst = (row_out * cols_out + col_out) * elem_size;
                    unsafe {
                        std::ptr::copy_nonoverlapping(src, data.as_mut_ptr().add(dst), elem_size);
                    }
                }
            }
            MatrixBuffer::from_bytes(self.dtype, row_indices.len(), cols_out, data)
        }
    }

    pub fn scatter(
        &self,
        row_indices: &[isize],
        col_indices: &[isize],
        values: &MatrixBuffer,
        pairwise: bool,
    ) -> Result<Self, String> {
        let values_cow = if values.dtype() == self.dtype() {
            Cow::Borrowed(values)
        } else {
            Cow::Owned(values.cast(self.dtype()).map_err(|err| {
                format!("scatter: unable to cast source values to destination dtype: {err}")
            })?)
        };
        let values = values_cow.as_ref();

        let mut base = self.to_contiguous()?;
        base.ensure_contiguous_mut();
        let elem_size = base.element_size();
        let data = Arc::make_mut(&mut base.data);

        if pairwise {
            if row_indices.len() != col_indices.len() {
                return Err("scatter pairwise expects matching index lengths".into());
            }
            let value_bytes = values.to_contiguous_bytes_vec();
            if value_bytes.len() != row_indices.len() * elem_size {
                return Err("scatter pairwise expects values length to match indices".into());
            }
            for (i, (&row_raw, &col_raw)) in row_indices.iter().zip(col_indices.iter()).enumerate()
            {
                let row = normalize_index(row_raw, base.rows)?;
                let col = normalize_index(col_raw, base.cols)?;
                let dst = (row * base.cols + col) * elem_size;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        value_bytes.as_ptr().add(i * elem_size),
                        data.as_mut_ptr().add(dst),
                        elem_size,
                    );
                }
            }
            Ok(base)
        } else {
            if values.rows != row_indices.len() || values.cols != col_indices.len() {
                return Err("scatter outer expects values shape to match indices".into());
            }
            let value_bytes = values.to_contiguous_bytes_vec();
            for (row_out, &row_raw) in row_indices.iter().enumerate() {
                let row = normalize_index(row_raw, base.rows)?;
                for (col_out, &col_raw) in col_indices.iter().enumerate() {
                    let col = normalize_index(col_raw, base.cols)?;
                    let dst = (row * base.cols + col) * elem_size;
                    let src = (row_out * values.cols + col_out) * elem_size;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            value_bytes.as_ptr().add(src),
                            data.as_mut_ptr().add(dst),
                            elem_size,
                        );
                    }
                }
            }
            Ok(base)
        }
    }

    pub fn take_rows(&self, indices: &[isize]) -> Result<Self, String> {
        let elem_size = self.element_size();
        let mut data = vec![0u8; indices.len() * self.cols * elem_size];
        for (row_out, &idx_raw) in indices.iter().enumerate() {
            let idx = normalize_index(idx_raw, self.rows)?;
            let dst_slice =
                &mut data[row_out * self.cols * elem_size..(row_out + 1) * self.cols * elem_size];
            self.copy_row_into(idx, dst_slice);
        }
        MatrixBuffer::from_bytes(self.dtype, indices.len(), self.cols, data)
    }

    pub fn take_cols(&self, indices: &[isize]) -> Result<Self, String> {
        let elem_size = self.element_size();
        let mut data = vec![0u8; self.rows * indices.len() * elem_size];
        for row in 0..self.rows {
            for (col_out, &idx_raw) in indices.iter().enumerate() {
                let idx = normalize_index(idx_raw, self.cols)?;
                let src = self.element_ptr(row, idx);
                let dst = (row * indices.len() + col_out) * elem_size;
                unsafe {
                    std::ptr::copy_nonoverlapping(src, data.as_mut_ptr().add(dst), elem_size);
                }
            }
        }
        MatrixBuffer::from_bytes(self.dtype, self.rows, indices.len(), data)
    }

    fn ensure_contiguous_mut(&mut self) {
        if self.is_standard_layout() && self.offset == 0 && Arc::get_mut(&mut self.data).is_some() {
            return;
        }
        let mut bytes = vec![0u8; self.len() * self.element_size()];
        self.copy_into_bytes(&mut bytes);
        record_copy_bytes(bytes.len());
        self.data = Arc::new(bytes);
        self.offset = 0;
        self.row_stride = self.cols as isize;
        self.col_stride = 1;
    }

    fn validate_bounds(&self) -> Result<(), String> {
        validate_view(
            self.data.len() / self.element_size(),
            self.rows,
            self.cols,
            self.offset,
            self.row_stride,
            self.col_stride,
        )
    }

    fn element_offset(&self, row: usize, col: usize) -> isize {
        self.offset + self.row_stride * row as isize + self.col_stride * col as isize
    }

    fn element_ptr(&self, row: usize, col: usize) -> *const u8 {
        let byte_offset = self.element_offset(row, col) * self.element_size() as isize;
        unsafe { self.data.as_ptr().offset(byte_offset) }
    }

    fn read_typed<T: Element>(&self, row: usize, col: usize) -> T {
        debug_assert_eq!(self.dtype, T::DTYPE);
        unsafe { *(self.element_ptr(row, col) as *const T) }
    }

    fn copy_into_bytes(&self, dst: &mut [u8]) {
        let elem_size = self.element_size();
        let mut write_offset = 0usize;
        if self.col_stride == 1 {
            for row in 0..self.rows {
                let src = self.element_ptr(row, 0);
                let row_bytes = self.cols * elem_size;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src,
                        dst.as_mut_ptr().add(write_offset),
                        row_bytes,
                    );
                }
                write_offset += row_bytes;
            }
        } else {
            for row in 0..self.rows {
                for col in 0..self.cols {
                    let src = self.element_ptr(row, col);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src,
                            dst.as_mut_ptr().add(write_offset),
                            elem_size,
                        );
                    }
                    write_offset += elem_size;
                }
            }
        }
    }

    pub(crate) fn copy_row_into(&self, row: usize, dst: &mut [u8]) {
        let elem_size = self.element_size();
        if self.col_stride == 1 {
            let src = self.element_ptr(row, 0);
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len());
            }
        } else {
            for col in 0..self.cols {
                let src = self.element_ptr(row, col);
                let dst_index = col * elem_size;
                unsafe {
                    std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr().add(dst_index), elem_size);
                }
            }
        }
    }
}

fn cast_from_f64<T>(
    buffer: &MatrixBuffer,
    min: f64,
    max: f64,
    map: impl Fn(f64) -> T,
) -> Result<MatrixBuffer, String>
where
    T: Element,
{
    let vec: Vec<T> = buffer
        .to_f64_vec()
        .into_iter()
        .map(|v| clamp_float(v, min, max))
        .map(map)
        .collect();
    MatrixBuffer::from_vec(vec, buffer.rows, buffer.cols)
}

fn cast_vec_to_dtype<T>(
    data: Vec<f64>,
    rows: usize,
    cols: usize,
    min: f64,
    max: f64,
    map: impl Fn(f64) -> T,
) -> Result<MatrixBuffer, String>
where
    T: Element,
{
    let vec: Vec<T> = data
        .into_iter()
        .map(|v| clamp_float(v, min, max))
        .map(map)
        .collect();
    MatrixBuffer::from_vec(vec, rows, cols)
}

fn collect_numeric<T, U>(out: &mut Vec<U>, buffer: &MatrixBuffer, map: impl Fn(T) -> U)
where
    T: Element,
{
    for row in 0..buffer.rows {
        for col in 0..buffer.cols {
            let value = buffer.read_typed::<T>(row, col);
            out.push(map(value));
        }
    }
}

fn validate_view(
    total_elems: usize,
    rows: usize,
    cols: usize,
    offset: isize,
    row_stride: isize,
    col_stride: isize,
) -> Result<(), String> {
    let total = total_elems as isize;
    let row_extent = if rows == 0 { 0 } else { (rows - 1) as isize };
    let col_extent = if cols == 0 { 0 } else { (cols - 1) as isize };
    let mut min_idx = offset;
    let mut max_idx = offset;

    for &r in &[0, row_extent] {
        for &c in &[0, col_extent] {
            let idx = offset + r * row_stride + c * col_stride;
            min_idx = min(min_idx, idx);
            max_idx = max(max_idx, idx);
        }
    }

    if min_idx < 0 || max_idx >= total {
        return Err("view exceeds underlying buffer bounds".into());
    }
    Ok(())
}

fn clamp_float(value: f64, min: f64, max: f64) -> f64 {
    if value.is_nan() {
        min
    } else if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

pub fn normalize_index(index: isize, len: usize) -> Result<usize, String> {
    if len == 0 {
        return Err("cannot index into empty dimension".into());
    }
    let len_isize = len as isize;
    let mut idx = index;
    if idx < 0 {
        idx += len_isize;
    }
    if idx < 0 || idx >= len_isize {
        return Err(format!(
            "index {index} out of bounds for dimension of {len}"
        ));
    }
    Ok(idx as usize)
}

pub fn normalize_slice(len: usize, spec: SliceSpec) -> Result<(usize, usize, isize), String> {
    let step = spec.step;
    if step == 0 {
        return Err("slice step must not be zero".into());
    }
    let len_isize = len as isize;
    let default_start = if step > 0 { 0 } else { len as isize - 1 };
    let default_end = if step > 0 { len as isize } else { -1 };

    let mut start = spec.start.unwrap_or(default_start);
    let mut end = spec.end.unwrap_or(default_end);

    if start < 0 {
        start += len_isize;
    }
    if end < 0 {
        end += len_isize;
    }

    if step > 0 {
        start = start.clamp(0, len_isize);
        end = end.clamp(0, len_isize);
        if end <= start {
            return Ok((start as usize, 0, step));
        }
        let span = end - start;
        let count = (span + step - 1) / step;
        Ok((start as usize, count as usize, step))
    } else {
        let step_abs = -step;
        start = start.clamp(-1, len_isize - 1);
        end = end.clamp(-1, len_isize - 1);
        if end >= start {
            return Ok((start as usize, 0, step));
        }
        let span = start - end;
        let count = (span + step_abs - 1) / step_abs;
        Ok((start as usize, count as usize, step))
    }
}
