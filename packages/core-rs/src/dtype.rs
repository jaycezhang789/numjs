#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    // Draft: fixed-point signed 64-bit with per-buffer scale metadata
    Fixed64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TypeKind {
    Bool,
    Unsigned,
    Signed,
    Float,
}

impl DType {
    pub const fn size_of(self) -> usize {
        match self {
            DType::Bool => 1,
            DType::Int8 | DType::UInt8 => 1,
            DType::Int16 | DType::UInt16 => 2,
            DType::Int32 | DType::UInt32 | DType::Float32 => 4,
            DType::Int64 | DType::UInt64 | DType::Float64 | DType::Fixed64 => 8,
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(self, DType::Float32 | DType::Float64)
    }

    pub const fn is_signed(self) -> bool {
        matches!(
            self,
            DType::Int8
                | DType::Int16
                | DType::Int32
                | DType::Int64
                | DType::Float32
                | DType::Float64
                | DType::Fixed64
        )
    }

    const fn kind(self) -> TypeKind {
        match self {
            DType::Bool => TypeKind::Bool,
            DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64 => TypeKind::Unsigned,
            DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64 | DType::Fixed64 => {
                TypeKind::Signed
            }
            DType::Float32 | DType::Float64 => TypeKind::Float,
        }
    }

    const fn width(self) -> u8 {
        self.size_of() as u8
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            DType::Bool => "bool",
            DType::Int8 => "int8",
            DType::Int16 => "int16",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::UInt8 => "uint8",
            DType::UInt16 => "uint16",
            DType::UInt32 => "uint32",
            DType::UInt64 => "uint64",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Fixed64 => "fixed64",
        }
    }
}

impl core::str::FromStr for DType {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "bool" => Ok(DType::Bool),
            "int8" => Ok(DType::Int8),
            "int16" => Ok(DType::Int16),
            "int32" => Ok(DType::Int32),
            "int64" => Ok(DType::Int64),
            "uint8" => Ok(DType::UInt8),
            "uint16" => Ok(DType::UInt16),
            "uint32" => Ok(DType::UInt32),
            "uint64" => Ok(DType::UInt64),
            "float32" => Ok(DType::Float32),
            "float64" => Ok(DType::Float64),
            "fixed64" => Ok(DType::Fixed64),
            other => Err(format!("unknown dtype '{other}'")),
        }
    }
}

pub fn promote_pair(a: DType, b: DType) -> Result<DType, String> {
    if (a == DType::Fixed64 && b.is_float()) || (b == DType::Fixed64 && a.is_float()) {
        return Err(format!(
            "dtype promotion between '{}' and '{}' is not supported; cast explicitly with astype()",
            a.as_str(),
            b.as_str()
        ));
    }
    if a == b {
        return Ok(a);
    }
    match (a, b) {
        (DType::Bool, other) | (other, DType::Bool) => return Ok(other),
        _ => {}
    }

    let kind_a = a.kind();
    let kind_b = b.kind();

    if matches!(kind_a, TypeKind::Float) || matches!(kind_b, TypeKind::Float) {
        return Ok(promote_float(a, b));
    }

    let result = match (kind_a, kind_b) {
        (TypeKind::Signed, TypeKind::Signed) => promote_signed(a.width().max(b.width())),
        (TypeKind::Unsigned, TypeKind::Unsigned) => promote_unsigned(a.width().max(b.width())),
        (TypeKind::Signed, TypeKind::Unsigned) | (TypeKind::Unsigned, TypeKind::Signed) => {
            let signed_width = if matches!(kind_a, TypeKind::Signed) {
                a.width()
            } else {
                b.width()
            };
            let unsigned_width = if matches!(kind_a, TypeKind::Unsigned) {
                a.width()
            } else {
                b.width()
            };
            if signed_width >= unsigned_width {
                promote_signed(signed_width.max(unsigned_width))
            } else {
                DType::Float64
            }
        }
        (TypeKind::Bool, _) | (_, TypeKind::Bool) => unreachable!(),
        _ => DType::Float64,
    };
    Ok(result)
}

pub fn promote_many(dtypes: &[DType]) -> Result<Option<DType>, String> {
    let mut iter = dtypes.iter().copied();
    let first = match iter.next() {
        Some(value) => value,
        None => return Ok(None),
    };
    let mut acc = first;
    for item in iter {
        acc = promote_pair(acc, item)?;
    }
    Ok(Some(acc))
}

fn promote_float(a: DType, b: DType) -> DType {
    let mut result = if matches!(a, DType::Float64) || matches!(b, DType::Float64) {
        DType::Float64
    } else {
        DType::Float32
    };
    if matches!(a.kind(), TypeKind::Float) && matches!(b.kind(), TypeKind::Float) {
        result = if result == DType::Float32 {
            DType::Float32
        } else {
            DType::Float64
        };
    }
    if matches!(a.kind(), TypeKind::Signed) || matches!(a.kind(), TypeKind::Unsigned) {
        result = DType::Float64;
    }
    if matches!(b.kind(), TypeKind::Signed) || matches!(b.kind(), TypeKind::Unsigned) {
        result = DType::Float64;
    }
    result
}

fn promote_signed(width: u8) -> DType {
    match width {
        0 | 1 => DType::Int8,
        2 => DType::Int16,
        3 | 4 => DType::Int32,
        _ => DType::Int64,
    }
}

fn promote_unsigned(width: u8) -> DType {
    match width {
        0 | 1 => DType::UInt8,
        2 => DType::UInt16,
        3 | 4 => DType::UInt32,
        _ => DType::UInt64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promote_pair_rejects_fixed64_float_mix() {
        let err = promote_pair(DType::Fixed64, DType::Float64).unwrap_err();
        assert!(
            err.contains("fixed64") && err.contains("float64"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn promote_many_propagates_fixed64_float_error() {
        let err = promote_many(&[DType::Float32, DType::Fixed64])
            .expect_err("expected promotion to fail");
        assert!(err.contains("float32") && err.contains("fixed64"));
    }
}
