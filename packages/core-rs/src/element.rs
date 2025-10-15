use crate::dtype::DType;

pub trait Element: Copy + 'static {
    const DTYPE: DType;
}

macro_rules! impl_element {
    ($t:ty, $dtype:expr) => {
        impl Element for $t {
            const DTYPE: DType = $dtype;
        }
    };
}

impl_element!(bool, DType::Bool);
impl_element!(i8, DType::Int8);
impl_element!(i16, DType::Int16);
impl_element!(i32, DType::Int32);
impl_element!(i64, DType::Int64);
impl_element!(u8, DType::UInt8);
impl_element!(u16, DType::UInt16);
impl_element!(u32, DType::UInt32);
impl_element!(u64, DType::UInt64);
impl_element!(f32, DType::Float32);
impl_element!(f64, DType::Float64);
