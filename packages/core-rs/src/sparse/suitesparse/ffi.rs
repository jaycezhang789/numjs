pub const STATUS_SUCCESS: i32 = 0;
pub const STATUS_BAD_SHAPE: i32 = -1;
pub const STATUS_OUT_OF_BOUNDS: i32 = -2;
pub const STATUS_ALLOC_FAILED: i32 = -3;
pub const STATUS_INTERNAL: i32 = -4;

extern "C" {
    pub fn numrs_suitesparse_csrmv(
        rows: usize,
        cols: usize,
        row_ptr: *const u32,
        col_idx: *const u32,
        values: *const f64,
        vector: *const f64,
        out: *mut f64,
    ) -> i32;

    pub fn numrs_suitesparse_csrgemm(
        rows: usize,
        cols: usize,
        rhs_cols: usize,
        row_ptr: *const u32,
        col_idx: *const u32,
        values: *const f64,
        rhs: *const f64,
        out: *mut f64,
    ) -> i32;

    pub fn numrs_suitesparse_csradd(
        rows: usize,
        cols: usize,
        row_ptr: *const u32,
        col_idx: *const u32,
        values: *const f64,
        dense: *const f64,
        out: *mut f64,
    ) -> i32;

    pub fn numrs_suitesparse_csrtranspose(
        rows: usize,
        cols: usize,
        row_ptr: *const u32,
        col_idx: *const u32,
        values: *const f64,
        out: *mut f64,
    ) -> i32;
}
