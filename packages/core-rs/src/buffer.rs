use crate::dtype::DType;
use crate::element::Element;
use crate::metrics::record_copy_bytes;
use num_traits::{Bounded, Float, NumCast, PrimInt, Signed, Unsigned};
use std::borrow::Cow;
use std::cmp::{max, min};
use std::fmt::Write;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CastingKind {
    Safe,
    SameKind,
    Unsafe,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverflowMode {
    Error,
    Clip,
    Wrap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundingMode {
    HalfAwayFromZero,
    HalfEven,
    Floor,
    Ceil,
    Trunc,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CastOptions {
    casting: CastingKind,
    overflow: OverflowMode,
    rounding: Option<RoundingMode>,
}

impl Default for CastOptions {
    fn default() -> Self {
        CastOptions {
            casting: CastingKind::Unsafe,
            overflow: OverflowMode::Clip,
            rounding: Some(RoundingMode::HalfAwayFromZero),
        }
    }
}

impl CastOptions {
    pub fn parse(spec: Option<&str>) -> Result<Self, String> {
        match spec {
            None => Ok(CastOptions::default()),
            Some(raw) => {
                let mut options = CastOptions {
                    casting: CastingKind::Unsafe,
                    overflow: OverflowMode::Clip,
                    rounding: None,
                };
                let mut casting_set = false;
                let mut overflow_set = false;
                let mut rounding_set = false;
                for token in raw
                    .split(|c: char| c == '|' || c == ',' || c.is_whitespace())
                    .filter(|token| !token.is_empty())
                {
                    let t = token.to_ascii_lowercase();
                    match t.as_str() {
                        "safe" => {
                            if casting_set {
                                return Err("casting: multiple casting modes specified".into());
                            }
                            casting_set = true;
                            options.casting = CastingKind::Safe;
                        }
                        "same_kind" | "samekind" => {
                            if casting_set {
                                return Err("casting: multiple casting modes specified".into());
                            }
                            casting_set = true;
                            options.casting = CastingKind::SameKind;
                        }
                        "unsafe" => {
                            if casting_set {
                                return Err("casting: multiple casting modes specified".into());
                            }
                            casting_set = true;
                            options.casting = CastingKind::Unsafe;
                        }
                        "clip" => {
                            if overflow_set {
                                return Err("casting: multiple overflow modes specified".into());
                            }
                            overflow_set = true;
                            options.overflow = OverflowMode::Clip;
                        }
                        "wrap" => {
                            if overflow_set {
                                return Err("casting: multiple overflow modes specified".into());
                            }
                            overflow_set = true;
                            options.overflow = OverflowMode::Wrap;
                        }
                        "error" | "strict" => {
                            if overflow_set {
                                return Err("casting: multiple overflow modes specified".into());
                            }
                            overflow_set = true;
                            options.overflow = OverflowMode::Error;
                        }
                        "round_half_away" | "round_half_up" | "round_nearest" => {
                            if rounding_set {
                                return Err("casting: multiple rounding modes specified".into());
                            }
                            rounding_set = true;
                            options.rounding = Some(RoundingMode::HalfAwayFromZero);
                        }
                        "round_half_even" | "round_bankers" => {
                            if rounding_set {
                                return Err("casting: multiple rounding modes specified".into());
                            }
                            rounding_set = true;
                            options.rounding = Some(RoundingMode::HalfEven);
                        }
                        "round_floor" => {
                            if rounding_set {
                                return Err("casting: multiple rounding modes specified".into());
                            }
                            rounding_set = true;
                            options.rounding = Some(RoundingMode::Floor);
                        }
                        "round_ceil" => {
                            if rounding_set {
                                return Err("casting: multiple rounding modes specified".into());
                            }
                            rounding_set = true;
                            options.rounding = Some(RoundingMode::Ceil);
                        }
                        "round_trunc" | "round_towards_zero" | "round_zero" => {
                            if rounding_set {
                                return Err("casting: multiple rounding modes specified".into());
                            }
                            rounding_set = true;
                            options.rounding = Some(RoundingMode::Trunc);
                        }
                        other => {
                            return Err(format!("casting: unrecognized token '{other}'"));
                        }
                    }
                }
                Ok(options)
            }
        }
    }

    pub fn casting(&self) -> CastingKind {
        self.casting
    }

    pub fn overflow(&self) -> OverflowMode {
        self.overflow
    }

    pub fn rounding(&self) -> Option<RoundingMode> {
        self.rounding
    }
}

impl MatrixBuffer {
    pub fn from_vec<T: Element>(
        mut data: Vec<T>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, String> {
        
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
        MatrixBuffer::new_internal_with_scale(
            dtype,
            rows,
            cols,
            Arc::new(bytes),
            0,
            cols as isize,
            1,
            None,
        )
    }

    pub fn from_bytes(
        dtype: DType,
        rows: usize,
        cols: usize,
        data: Vec<u8>,
    ) -> Result<Self, String> {
        
        let expected = rows
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(dtype.size_of()))
            .ok_or_else(|| "shape is too large".to_string())?;
        if expected != data.len() {
            return Err("byte length does not match shape".into());
        }
        MatrixBuffer::new_internal_with_scale(
            dtype,
            rows,
            cols,
            Arc::new(data),
            0,
            cols as isize,
            1,
            None,
        )
    }

    pub fn from_bytes_with_scale(
        dtype: DType,
        rows: usize,
        cols: usize,
        data: Vec<u8>,
        fixed_scale: Option<i32>,
    ) -> Result<Self, String> {
        MatrixBuffer::new_internal_with_scale(
            dtype,
            rows,
            cols,
            Arc::new(data),
            0,
            cols as isize,
            1,
            fixed_scale,
        )
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

    fn new_internal_with_scale(
        dtype: DType,
        rows: usize,
        cols: usize,
        data: Arc<Vec<u8>>,
        offset: isize,
        row_stride: isize,
        col_stride: isize,
        fixed_scale: Option<i32>,
    ) -> Result<Self, String> {
        let mut buffer =
            MatrixBuffer::new_internal(dtype, rows, cols, data, offset, row_stride, col_stride)?;
        buffer.fixed_scale = fixed_scale;
        Ok(buffer)
    }
    pub fn from_fixed_i64_vec(
        data: Vec<i64>,
        rows: usize,
        cols: usize,
        scale: i32,
    ) -> Result<Self, String> {
        
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
        MatrixBuffer::new_internal_with_scale(
            dtype,
            rows,
            cols,
            Arc::new(bytes),
            0,
            cols as isize,
            1,
            Some(scale),
        )
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

    pub fn is_contiguous(&self) -> bool {
        self.is_standard_layout() && self.offset == 0
    }

    pub fn ensure_standard_layout(&self) -> Result<(), String> {
        if self.col_stride != 1 {
            return Err("matrix buffer requires column stride of 1 for a contiguous view".into());
        }
        if self.row_stride != self.cols as isize {
            return Err("matrix buffer requires row stride equal to number of columns for a contiguous view".into());
        }
        Ok(())
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

    pub fn try_as_slice<T: Element>(&self) -> Result<&[T], String> {
        self.as_slice::<T>()
            .ok_or_else(|| self.build_view_error(Some(T::DTYPE)))
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

    pub fn try_as_byte_slice(&self) -> Result<&[u8], String> {
        self.as_byte_slice()
            .ok_or_else(|| self.build_view_error(None))
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

    pub fn try_as_slice_mut<T: Element>(&mut self) -> Result<&mut [T], String> {
        if T::DTYPE != self.dtype {
            return Err(self.build_view_error(Some(T::DTYPE)));
        }
        if !self.is_standard_layout() || self.offset != 0 {
            return Err(self.build_view_error(Some(T::DTYPE)));
        }
        let len = self.len();
        let ptr = Arc::make_mut(&mut self.data).as_mut_ptr();
        let byte_offset = (self.offset as isize) * self.element_size() as isize;
        unsafe {
            let ptr = ptr.offset(byte_offset) as *mut T;
            Ok(std::slice::from_raw_parts_mut(ptr, len))
        }
    }

    pub fn to_contiguous(&self) -> Result<Self, String> {
        if self.is_standard_layout() && self.offset == 0 {
            return Ok(self.clone());
        }
        let mut bytes = vec![0u8; self.len() * self.element_size()];
        self.copy_into_bytes(&mut bytes);
        record_copy_bytes(bytes.len());
        MatrixBuffer::from_bytes_with_scale(
            self.dtype,
            self.rows,
            self.cols,
            bytes,
            self.fixed_scale,
        )
    }

    pub fn to_contiguous_bytes_vec(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; self.len() * self.element_size()];
        self.copy_into_bytes(&mut bytes);
        record_copy_bytes(bytes.len());
        bytes
    }

    pub fn try_as_byte_arc(&self) -> Option<(Arc<Vec<u8>>, usize, usize)> {
        if !self.is_standard_layout() {
            return None;
        }
        if self.offset < 0 {
            return None;
        }
        let elem_size = self.element_size();
        let offset_bytes = (self.offset as usize).checked_mul(elem_size)?;
        let len_bytes = self.len().checked_mul(elem_size)?;
        if offset_bytes + len_bytes > self.data.len() {
            return None;
        }
        Some((self.data.clone(), offset_bytes, len_bytes))
    }

    fn from_bytes_like(&self, rows: usize, cols: usize, data: Vec<u8>) -> Result<Self, String> {
        if self.dtype == DType::Fixed64 {
            MatrixBuffer::from_bytes_with_scale(self.dtype, rows, cols, data, self.fixed_scale)
        } else {
            MatrixBuffer::from_bytes(self.dtype, rows, cols, data)
        }
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
        if dtype == DType::Fixed64 {
            MatrixBuffer::from_bytes_with_scale(
                dtype,
                self.rows,
                self.cols,
                bytes,
                self.fixed_scale,
            )
        } else {
            MatrixBuffer::from_bytes(dtype, self.rows, self.cols, bytes)
        }
    }

    pub fn reinterpret(&self, target: DType) -> Result<Self, String> {
        if target.size_of() != self.dtype.size_of() {
            return Err("reinterpret: dtype width mismatch".into());
        }
        MatrixBuffer::new_internal_with_scale(
            target,
            self.rows,
            self.cols,
            Arc::clone(&self.data),
            self.offset,
            self.row_stride,
            self.col_stride,
            if self.dtype == DType::Fixed64 {
                self.fixed_scale
            } else {
                None
            },
        )
    }

    pub fn cast(&self, target: DType) -> Result<Self, String> {
        self.cast_with_options(target, &CastOptions::default())
    }

    pub fn cast_with_spec(&self, target: DType, casting: Option<&str>) -> Result<Self, String> {
        let options = CastOptions::parse(casting)?;
        self.cast_with_options(target, &options)
    }

    pub fn cast_with_options(&self, target: DType, options: &CastOptions) -> Result<Self, String> {
        if target == self.dtype {
            return Ok(self.clone());
        }
        validate_casting_mode(self.dtype, target, options)?;
        if target == DType::Fixed64 {
            return Err(
                "cast: casting to fixed64 not supported; construct with from_fixed_i64_vec".into(),
            );
        }
        record_copy_bytes(self.len() * target.size_of());
        match target {
            DType::Bool => cast_to_bool(self, options),
            DType::Int8 => cast_to_signed::<i8>(self, options),
            DType::Int16 => cast_to_signed::<i16>(self, options),
            DType::Int32 => cast_to_signed::<i32>(self, options),
            DType::Int64 => cast_to_signed::<i64>(self, options),
            DType::UInt8 => cast_to_unsigned::<u8>(self, options),
            DType::UInt16 => cast_to_unsigned::<u16>(self, options),
            DType::UInt32 => cast_to_unsigned::<u32>(self, options),
            DType::UInt64 => cast_to_unsigned::<u64>(self, options),
            DType::Float32 => cast_to_float::<f32>(self, options),
            DType::Float64 => cast_to_float::<f64>(self, options),
            DType::Fixed64 => unreachable!(),
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
        if dtype == DType::Fixed64 {
            return Err(
                "from_f64_vec(Fixed64): requires explicit scale; use from_fixed_i64_vec".into(),
            );
        }
        let base = MatrixBuffer::from_vec(data, rows, cols)?;
        if dtype == DType::Float64 {
            Ok(base)
        } else {
            base.cast_with_options(dtype, &CastOptions::default())
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
        MatrixBuffer::new_internal_with_scale(
            self.dtype,
            row_len,
            col_len,
            Arc::clone(&self.data),
            offset,
            row_stride,
            col_stride,
            if self.dtype == DType::Fixed64 {
                self.fixed_scale
            } else {
                None
            },
        )
    }

    pub fn row(&self, index: isize) -> Result<Self, String> {
        let idx = normalize_index(index, self.rows)? as isize;
        MatrixBuffer::new_internal_with_scale(
            self.dtype,
            1,
            self.cols,
            Arc::clone(&self.data),
            self.offset + idx * self.row_stride,
            self.row_stride,
            self.col_stride,
            if self.dtype == DType::Fixed64 {
                self.fixed_scale
            } else {
                None
            },
        )
    }

    pub fn column(&self, index: isize) -> Result<Self, String> {
        let idx = normalize_index(index, self.cols)? as isize;
        MatrixBuffer::new_internal_with_scale(
            self.dtype,
            self.rows,
            1,
            Arc::clone(&self.data),
            self.offset + idx * self.col_stride,
            self.row_stride,
            self.col_stride,
            if self.dtype == DType::Fixed64 {
                self.fixed_scale
            } else {
                None
            },
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
            self.from_bytes_like(row_indices.len(), 1, data)
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
            self.from_bytes_like(row_indices.len(), cols_out, data)
        }
    }

    pub fn scatter(
        &self,
        row_indices: &[isize],
        col_indices: &[isize],
        values: &MatrixBuffer,
        pairwise: bool,
    ) -> Result<Self, String> {
        // Ensure fixed64 scale consistency when writing into a fixed64 destination
        if self.dtype == DType::Fixed64 {
            if values.dtype != DType::Fixed64 {
                return Err("scatter: values dtype must be fixed64 to match destination".into());
            }
            if values.fixed_scale != self.fixed_scale {
                return Err("scatter: fixed64 scale mismatch".into());
            }
        }
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
            // Record number of bytes written
            record_copy_bytes(value_bytes.len());
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
            record_copy_bytes(value_bytes.len());
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
        record_copy_bytes(data.len());
        self.from_bytes_like(indices.len(), self.cols, data)
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
        record_copy_bytes(data.len());
        self.from_bytes_like(self.rows, indices.len(), data)
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

    fn build_view_error(&self, expected: Option<DType>) -> String {
        let mut message = String::new();
        match expected {
            Some(dtype) => {
                let _ = write!(
                    &mut message,
                    "unable to obtain contiguous view as {:?}: ",
                    dtype
                );
                if self.dtype != dtype {
                    let _ = write!(&mut message, "buffer dtype is {:?}; ", self.dtype);
                }
            }
            None => {
                let _ = write!(&mut message, "unable to obtain contiguous byte view: ");
            }
        }
        if self.col_stride != 1 || self.row_stride != self.cols as isize {
            let _ = write!(
                &mut message,
                "layout is not standard (row_stride={}, col_stride={}); ",
                self.row_stride, self.col_stride
            );
        }
        if self.offset != 0 {
            let _ = write!(&mut message, "offset is {}; ", self.offset);
        }
        if message.ends_with("; ") {
            message.truncate(message.len() - 2);
        }
        message
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

fn cast_to_bool(buffer: &MatrixBuffer, _options: &CastOptions) -> Result<MatrixBuffer, String> {
    let vec: Vec<bool> = buffer.to_f64_vec().into_iter().map(|v| v != 0.0).collect();
    MatrixBuffer::from_vec(vec, buffer.rows, buffer.cols)
}

fn cast_to_float<F>(buffer: &MatrixBuffer, _options: &CastOptions) -> Result<MatrixBuffer, String>
where
    F: Element + Float + NumCast,
{
    let mut out = Vec::<F>::with_capacity(buffer.len());
    for value in buffer.to_f64_vec() {
        let casted: F = NumCast::from(value).ok_or_else(|| {
            format!(
                "cast({:?}->{:?}): value {value} cannot be represented",
                buffer.dtype(),
                F::DTYPE
            )
        })?;
        out.push(casted);
    }
    MatrixBuffer::from_vec(out, buffer.rows, buffer.cols)
}

fn cast_to_signed<T>(buffer: &MatrixBuffer, options: &CastOptions) -> Result<MatrixBuffer, String>
where
    T: Element + PrimInt + Signed + Bounded + NumCast,
{
    let src_dtype = buffer.dtype();
    let dst_dtype = T::DTYPE;
    let mut out = Vec::<T>::with_capacity(buffer.len());
    let values = buffer.to_f64_vec();
    let rounding_required = matches!(src_dtype, DType::Float32 | DType::Float64 | DType::Fixed64);
    let rounding_mode = if rounding_required {
        options
            .rounding()
            .ok_or_else(|| float_to_int_rounding_error(src_dtype, dst_dtype))?
    } else {
        options.rounding().unwrap_or(RoundingMode::Trunc)
    };
    let min: f64 = NumCast::from(T::min_value()).expect("signed minimum convertible to f64");
    let max: f64 = NumCast::from(T::max_value()).expect("signed maximum convertible to f64");
    let span = (max - min) + 1.0;
    for (idx, mut value) in values.into_iter().enumerate() {
        if value.is_nan() {
            return Err(nan_to_int_error(src_dtype, dst_dtype, idx));
        }
        if rounding_required || options.rounding().is_some() {
            value = apply_rounding(value, rounding_mode);
        }
        if !value.is_finite() {
            match options.overflow() {
                OverflowMode::Clip => {
                    value = if value.is_sign_positive() { max } else { min };
                }
                OverflowMode::Wrap | OverflowMode::Error => {
                    return Err(non_finite_error(src_dtype, dst_dtype, idx));
                }
            }
        }
        let coerced = match options.overflow() {
            OverflowMode::Error => {
                if value < min || value > max {
                    return Err(range_error(value, src_dtype, dst_dtype, idx));
                }
                value
            }
            OverflowMode::Clip => value.clamp(min, max),
            OverflowMode::Wrap => wrap_value(value, min, span),
        };
        let integer = coerced.round();
        let casted: T = NumCast::from(integer)
            .ok_or_else(|| range_error(integer, src_dtype, dst_dtype, idx))?;
        out.push(casted);
    }
    MatrixBuffer::from_vec(out, buffer.rows, buffer.cols)
}

fn cast_to_unsigned<T>(buffer: &MatrixBuffer, options: &CastOptions) -> Result<MatrixBuffer, String>
where
    T: Element + PrimInt + Unsigned + Bounded + NumCast,
{
    let src_dtype = buffer.dtype();
    let dst_dtype = T::DTYPE;
    let mut out = Vec::<T>::with_capacity(buffer.len());
    let values = buffer.to_f64_vec();
    let rounding_required = matches!(src_dtype, DType::Float32 | DType::Float64 | DType::Fixed64);
    let rounding_mode = if rounding_required {
        options
            .rounding()
            .ok_or_else(|| float_to_int_rounding_error(src_dtype, dst_dtype))?
    } else {
        options.rounding().unwrap_or(RoundingMode::Trunc)
    };
    let min = 0.0;
    let max: f64 = NumCast::from(T::max_value()).expect("unsigned maximum convertible to f64");
    let span = max + 1.0;
    for (idx, mut value) in values.into_iter().enumerate() {
        if value.is_nan() {
            return Err(nan_to_int_error(src_dtype, dst_dtype, idx));
        }
        if rounding_required || options.rounding().is_some() {
            value = apply_rounding(value, rounding_mode);
        }
        if !value.is_finite() {
            match options.overflow() {
                OverflowMode::Clip => {
                    value = if value.is_sign_positive() { max } else { min };
                }
                OverflowMode::Wrap | OverflowMode::Error => {
                    return Err(non_finite_error(src_dtype, dst_dtype, idx));
                }
            }
        }
        let coerced = match options.overflow() {
            OverflowMode::Error => {
                if value < min || value > max {
                    return Err(range_error(value, src_dtype, dst_dtype, idx));
                }
                value
            }
            OverflowMode::Clip => value.clamp(min, max),
            OverflowMode::Wrap => wrap_value(value, min, span),
        };
        let integer = coerced.round();
        let casted: T = NumCast::from(integer)
            .ok_or_else(|| range_error(integer, src_dtype, dst_dtype, idx))?;
        out.push(casted);
    }
    MatrixBuffer::from_vec(out, buffer.rows, buffer.cols)
}

fn wrap_value(value: f64, min: f64, span: f64) -> f64 {
    if span <= 0.0 {
        min
    } else {
        ((value - min).rem_euclid(span)) + min
    }
}

fn apply_rounding(value: f64, mode: RoundingMode) -> f64 {
    match mode {
        RoundingMode::HalfAwayFromZero => value.round(),
        RoundingMode::HalfEven => round_half_even(value),
        RoundingMode::Floor => value.floor(),
        RoundingMode::Ceil => value.ceil(),
        RoundingMode::Trunc => value.trunc(),
    }
}

fn round_half_even(value: f64) -> f64 {
    const EPS: f64 = 1e-12;
    let rounded = value.round();
    let floor = value.floor();
    let ceil = value.ceil();
    let diff_floor = (value - floor).abs();
    if (diff_floor - 0.5).abs() <= EPS {
        let floor_even = (floor as i128) % 2 == 0;
        return if floor_even { floor } else { ceil };
    }
    let diff_ceil = (value - ceil).abs();
    if (diff_ceil - 0.5).abs() <= EPS {
        let ceil_even = (ceil as i128) % 2 == 0;
        return if ceil_even { ceil } else { floor };
    }
    rounded
}

fn float_to_int_rounding_error(src: DType, dst: DType) -> String {
    format!(
        "cast({src:?}->{dst:?}): float to integer casts require a rounding mode (casting=\"round_*\")"
    )
}

fn nan_to_int_error(src: DType, dst: DType, index: usize) -> String {
    format!("cast({src:?}->{dst:?}): NaN cannot be represented as an integer (index {index})")
}

fn non_finite_error(src: DType, dst: DType, index: usize) -> String {
    format!("cast({src:?}->{dst:?}): non-finite value cannot be represented (index {index})")
}

fn range_error<V: std::fmt::Display>(value: V, src: DType, dst: DType, index: usize) -> String {
    format!("cast({src:?}->{dst:?}): value {value} falls outside destination range (index {index})")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CastCategory {
    Bool,
    Integer,
    Float,
    Fixed,
}

fn validate_casting_mode(src: DType, dst: DType, options: &CastOptions) -> Result<(), String> {
    match options.casting() {
        CastingKind::Unsafe => Ok(()),
        CastingKind::SameKind => {
            if cast_category(src) == cast_category(dst) {
                Ok(())
            } else {
                Err(format!(
                    "cast({src:?}->{dst:?}): violates casting='same_kind'"
                ))
            }
        }
        CastingKind::Safe => {
            if can_cast_safe(src, dst) {
                Ok(())
            } else {
                Err(format!("cast({src:?}->{dst:?}): violates casting='safe'"))
            }
        }
    }
}

fn cast_category(dtype: DType) -> CastCategory {
    match dtype {
        DType::Bool => CastCategory::Bool,
        DType::Int8
        | DType::Int16
        | DType::Int32
        | DType::Int64
        | DType::UInt8
        | DType::UInt16
        | DType::UInt32
        | DType::UInt64 => CastCategory::Integer,
        DType::Float32 | DType::Float64 => CastCategory::Float,
        DType::Fixed64 => CastCategory::Fixed,
    }
}

fn can_cast_safe(src: DType, dst: DType) -> bool {
    if src == dst {
        return true;
    }
    match src {
        DType::Bool => matches!(
            dst,
            DType::Bool
                | DType::Int8
                | DType::Int16
                | DType::Int32
                | DType::Int64
                | DType::UInt8
                | DType::UInt16
                | DType::UInt32
                | DType::UInt64
                | DType::Float32
                | DType::Float64
        ),
        DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64 => {
            if is_signed_integer(dst) {
                dtype_width_bits(dst) >= dtype_width_bits(src)
            } else if is_float_dtype(dst) {
                safe_int_to_float(dtype_width_bits(src), dst)
            } else {
                false
            }
        }
        DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64 => {
            if is_unsigned_integer(dst) {
                dtype_width_bits(dst) >= dtype_width_bits(src)
            } else if is_signed_integer(dst) {
                dtype_width_bits(dst) > dtype_width_bits(src)
            } else if is_float_dtype(dst) {
                safe_int_to_float(dtype_width_bits(src), dst)
            } else {
                false
            }
        }
        DType::Float32 => matches!(dst, DType::Float32 | DType::Float64),
        DType::Float64 => matches!(dst, DType::Float64),
        DType::Fixed64 => dst == DType::Fixed64,
    }
}

fn is_signed_integer(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64
    )
}

fn is_unsigned_integer(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64
    )
}

fn is_float_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::Float32 | DType::Float64)
}

fn dtype_width_bits(dtype: DType) -> u32 {
    match dtype {
        DType::Bool => 1,
        DType::Int8 | DType::UInt8 => 8,
        DType::Int16 | DType::UInt16 => 16,
        DType::Int32 | DType::UInt32 => 32,
        DType::Int64 | DType::UInt64 | DType::Fixed64 => 64,
        DType::Float32 | DType::Float64 => 0,
    }
}

fn float_mantissa_bits(dtype: DType) -> u32 {
    match dtype {
        DType::Float32 => 24,
        DType::Float64 => 53,
        _ => 0,
    }
}

fn safe_int_to_float(width_bits: u32, float_dtype: DType) -> bool {
    let mantissa = float_mantissa_bits(float_dtype);
    mantissa >= width_bits
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
    // Permit empty views (rows==0 or cols==0) as long as offset is within [0, total]
    if rows == 0 || cols == 0 {
        if offset < 0 || (offset as usize) > total_elems {
            return Err("view exceeds underlying buffer bounds".into());
        }
        return Ok(());
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_as_slice_success() {
        let buffer = MatrixBuffer::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2).expect("buffer");
        let view = buffer.try_as_slice::<f32>().expect("typed view");
        assert_eq!(view, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn try_as_slice_reports_dtype_mismatch() {
        let buffer = MatrixBuffer::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2).expect("buffer");
        let err = buffer.try_as_slice::<i32>().unwrap_err();
        assert!(
            err.contains("dtype"),
            "expected dtype mismatch in error message, got: {err}"
        );
    }

    #[test]
    fn try_as_slice_mut_requires_standard_layout() {
        let base = MatrixBuffer::from_vec(vec![1i32, 2, 3, 4], 2, 2).expect("buffer");
        let mut column = base.column(0).expect("column");
        let err = column.try_as_slice_mut::<i32>().unwrap_err();
        assert!(
            err.contains("layout"),
            "expected layout mention in error message, got: {err}"
        );
    }

    #[test]
    fn cast_with_safe_rejects_downcast() {
        let buffer = MatrixBuffer::from_vec(vec![10i16, -5, 20], 3, 1).expect("int16 buffer");
        let result = buffer.cast_with_spec(DType::Int8, Some("safe"));
        assert!(result.is_err(), "expected safe cast to reject narrowing");
    }

    #[test]
    fn cast_round_floor_float_to_int() {
        let buffer = MatrixBuffer::from_vec(vec![1.9f64, -1.2, -0.1], 3, 1).expect("float buffer");
        let casted = buffer
            .cast_with_spec(DType::Int32, Some("round_floor|clip"))
            .expect("round_floor cast");
        let values = casted.try_as_slice::<i32>().expect("slice");
        assert_eq!(values, &[1, -2, -1]);
    }

    #[test]
    fn cast_wrap_unsigned() {
        let buffer = MatrixBuffer::from_vec(vec![260.0f64, -5.0], 2, 1).expect("float buffer");
        let casted = buffer
            .cast_with_spec(DType::UInt8, Some("round_trunc|wrap"))
            .expect("wrap cast");
        let values = casted.try_as_slice::<u8>().expect("slice");
        assert_eq!(values, &[4, 251]);
    }

    #[test]
    fn cast_options_parse_mixed_tokens() {
        let options = CastOptions::parse(Some("same_kind | round_trunc | wrap")).unwrap();
        assert_eq!(options.casting(), CastingKind::SameKind);
        assert_eq!(options.overflow(), OverflowMode::Wrap);
        assert_eq!(options.rounding(), Some(RoundingMode::Trunc));
    }
}

