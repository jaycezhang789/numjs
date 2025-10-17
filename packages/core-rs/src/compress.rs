use crate::buffer::MatrixBuffer;
use crate::dtype::DType;
use crate::element::Element as ElementTrait;

pub type CoreResult<T> = Result<T, String>;

pub fn compress(mask: &MatrixBuffer, matrix: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    if mask.dtype() != DType::Bool {
        return Err("compress: mask must have dtype=bool".into());
    }
    if mask.rows() != matrix.rows() || mask.cols() != matrix.cols() {
        return Err(
            "compress: shapes must match; broadcast inputs to the same shape before compress"
                .into(),
        );
    }
    let mask_contig = mask.to_contiguous()?;
    let mask_slice = mask_contig.try_as_slice::<bool>()?;
    match matrix.dtype() {
        DType::Fixed64 => compress_fixed64(mask_slice, matrix),
        DType::Bool => compress_typed::<bool>(mask_slice, matrix),
        DType::Int8 => compress_typed::<i8>(mask_slice, matrix),
        DType::Int16 => compress_typed::<i16>(mask_slice, matrix),
        DType::Int32 => compress_typed::<i32>(mask_slice, matrix),
        DType::Int64 => compress_typed::<i64>(mask_slice, matrix),
        DType::UInt8 => compress_typed::<u8>(mask_slice, matrix),
        DType::UInt16 => compress_typed::<u16>(mask_slice, matrix),
        DType::UInt32 => compress_typed::<u32>(mask_slice, matrix),
        DType::UInt64 => compress_typed::<u64>(mask_slice, matrix),
        DType::Float32 => compress_typed::<f32>(mask_slice, matrix),
        DType::Float64 => compress_typed::<f64>(mask_slice, matrix),
    }
}

fn compress_typed<T>(mask: &[bool], matrix: &MatrixBuffer) -> CoreResult<MatrixBuffer>
where
    T: ElementTrait,
{
    let contig = matrix.to_contiguous()?;
    let data = contig.try_as_slice::<T>()?;
    let mut out: Vec<T> = Vec::with_capacity(data.len());
    for (idx, &flag) in mask.iter().enumerate() {
        if flag {
            out.push(data[idx]);
        }
    }
    // Allow empty result (0 x 1)
    MatrixBuffer::from_vec(out, mask.iter().filter(|&&b| b).count(), 1).map_err(Into::into)
}

fn compress_fixed64(mask: &[bool], matrix: &MatrixBuffer) -> CoreResult<MatrixBuffer> {
    let scale = matrix
        .fixed_scale()
        .ok_or_else(|| "compress(fixed64): missing scale metadata".to_string())?;
    let contig = matrix.to_contiguous()?;
    let bytes = contig
        .as_byte_slice()
        .ok_or_else(|| "compress(fixed64): contiguous bytes unavailable".to_string())?;
    let mut out: Vec<i64> = Vec::with_capacity(mask.len());
    for (i, chunk) in bytes.chunks_exact(8).enumerate() {
        if mask[i] {
            let arr = [
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ];
            out.push(i64::from_ne_bytes(arr));
        }
    }
    MatrixBuffer::from_fixed_i64_vec(out, mask.iter().filter(|&&b| b).count(), 1, scale)
}
