/// Internal helper used by the macro tests to ensure compilation.
pub(crate) fn __size_of<T>() -> usize {
    std::mem::size_of::<T>()
}

/// Dispatch a `DType` to its corresponding numeric Rust type (`i8`, `u8`, `f32`, 闂?.
///
/// ```
/// # use num_rs_core::{dtype::DType, match_numeric_dtype};
/// let dtype = DType::Int16;
/// let size = match_numeric_dtype!(dtype, Ty, {
///     std::mem::size_of::<Ty>()
/// });
/// assert_eq!(size, 2);
/// ```
#[macro_export]
macro_rules! match_numeric_dtype {
    ($dtype:expr, $T:ident, $body:block, fallback $fallback:expr) => {{
        #[allow(non_snake_case)]
        match $dtype {
            $crate::dtype::DType::Int8 => {
                type $T = i8;
                $body
            }
            $crate::dtype::DType::Int16 => {
                type $T = i16;
                $body
            }
            $crate::dtype::DType::Int32 => {
                type $T = i32;
                $body
            }
            $crate::dtype::DType::Int64 => {
                type $T = i64;
                $body
            }
            $crate::dtype::DType::UInt8 => {
                type $T = u8;
                $body
            }
            $crate::dtype::DType::UInt16 => {
                type $T = u16;
                $body
            }
            $crate::dtype::DType::UInt32 => {
                type $T = u32;
                $body
            }
            $crate::dtype::DType::UInt64 => {
                type $T = u64;
                $body
            }
            $crate::dtype::DType::Float32 => {
                type $T = f32;
                $body
            }
            $crate::dtype::DType::Float64 => {
                type $T = f64;
                $body
            }
            _ => $fallback,
        }
    }};
    ($dtype:expr, $T:ident, $body:block) => {{
        $crate::match_numeric_dtype!(
            $dtype,
            $T,
            $body,
            fallback panic!("unsupported numeric dtype: {:?}", $dtype)
        )
    }};
}

/// Dispatch to signed integer types (`i8`, `i16`, `i32`, `i64`).
#[macro_export]
macro_rules! match_signed_integer_dtype {
    ($dtype:expr, $T:ident, $body:block, fallback $fallback:expr) => {{
        #[allow(non_snake_case)]
        match $dtype {
            $crate::dtype::DType::Int8 => {
                type $T = i8;
                $body
            }
            $crate::dtype::DType::Int16 => {
                type $T = i16;
                $body
            }
            $crate::dtype::DType::Int32 => {
                type $T = i32;
                $body
            }
            $crate::dtype::DType::Int64 => {
                type $T = i64;
                $body
            }
            _ => $fallback,
        }
    }};
    ($dtype:expr, $T:ident, $body:block) => {{
        $crate::match_signed_integer_dtype!(
            $dtype,
            $T,
            $body,
            fallback panic!("unsupported signed integer dtype: {:?}", $dtype)
        )
    }};
}

/// Dispatch to unsigned integer types (`u8`, `u16`, `u32`, `u64`).
#[macro_export]
macro_rules! match_unsigned_integer_dtype {
    ($dtype:expr, $T:ident, $body:block, fallback $fallback:expr) => {{
        #[allow(non_snake_case)]
        match $dtype {
            $crate::dtype::DType::UInt8 => {
                type $T = u8;
                $body
            }
            $crate::dtype::DType::UInt16 => {
                type $T = u16;
                $body
            }
            $crate::dtype::DType::UInt32 => {
                type $T = u32;
                $body
            }
            $crate::dtype::DType::UInt64 => {
                type $T = u64;
                $body
            }
            _ => $fallback,
        }
    }};
    ($dtype:expr, $T:ident, $body:block) => {{
        $crate::match_unsigned_integer_dtype!(
            $dtype,
            $T,
            $body,
            fallback panic!("unsupported unsigned integer dtype: {:?}", $dtype)
        )
    }};
}

/// Dispatch to floating-point types (`f32`, `f64`).
#[macro_export]
macro_rules! match_float_dtype {
    ($dtype:expr, $T:ident, $body:block, fallback $fallback:expr) => {{
        #[allow(non_snake_case)]
        match $dtype {
            $crate::dtype::DType::Float32 => {
                type $T = f32;
                $body
            }
            $crate::dtype::DType::Float64 => {
                type $T = f64;
                $body
            }
            _ => $fallback,
        }
    }};
    ($dtype:expr, $T:ident, $body:block) => {{
        $crate::match_float_dtype!(
            $dtype,
            $T,
            $body,
            fallback panic!("unsupported float dtype: {:?}", $dtype)
        )
    }};
}

/// Helper macro to iterate over all numeric dtypes.
#[macro_export]
macro_rules! for_each_numeric_dtype {
    ($mac:ident) => {
        $mac!($crate::dtype::DType::Int8, i8);
        $mac!($crate::dtype::DType::Int16, i16);
        $mac!($crate::dtype::DType::Int32, i32);
        $mac!($crate::dtype::DType::Int64, i64);
        $mac!($crate::dtype::DType::UInt8, u8);
        $mac!($crate::dtype::DType::UInt16, u16);
        $mac!($crate::dtype::DType::UInt32, u32);
        $mac!($crate::dtype::DType::UInt64, u64);
        $mac!($crate::dtype::DType::Float32, f32);
        $mac!($crate::dtype::DType::Float64, f64);
    };
    ($mac:ident, $($args:tt)*) => {
        $mac!($crate::dtype::DType::Int8, i8, $($args)*);
        $mac!($crate::dtype::DType::Int16, i16, $($args)*);
        $mac!($crate::dtype::DType::Int32, i32, $($args)*);
        $mac!($crate::dtype::DType::Int64, i64, $($args)*);
        $mac!($crate::dtype::DType::UInt8, u8, $($args)*);
        $mac!($crate::dtype::DType::UInt16, u16, $($args)*);
        $mac!($crate::dtype::DType::UInt32, u32, $($args)*);
        $mac!($crate::dtype::DType::UInt64, u64, $($args)*);
        $mac!($crate::dtype::DType::Float32, f32, $($args)*);
        $mac!($crate::dtype::DType::Float64, f64, $($args)*);
    };
}

#[cfg(test)]
mod tests {
    use crate::dtype::DType;

    #[test]
    fn match_numeric_dispatches_size() {
        let dtype = DType::UInt16;
        let size = match_numeric_dtype!(dtype, Ty, { std::mem::size_of::<Ty>() });
        assert_eq!(size, 2);
    }

    #[test]
    fn match_numeric_fallback_allows_custom_result() {
        let dtype = DType::Bool;
        let result: Result<usize, &'static str> = match_numeric_dtype!(
            dtype,
            Ty,
            { Ok(std::mem::size_of::<Ty>()) },
            fallback Err("unsupported")
        );
        assert!(result.is_err());
    }

    #[test]
    fn match_signed_integer_dispatch() {
        let dtype = DType::Int32;
        let size = match_signed_integer_dtype!(dtype, Ty, { std::mem::size_of::<Ty>() });
        assert_eq!(size, 4);
    }

    #[test]
    fn match_float_dispatch() {
        let dtype = DType::Float64;
        let size = match_float_dtype!(dtype, Ty, { std::mem::size_of::<Ty>() });
        assert_eq!(size, 8);
    }
}
