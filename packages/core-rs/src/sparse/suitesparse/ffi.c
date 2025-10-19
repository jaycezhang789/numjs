#include "ffi.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline size_t nnz_from_row_ptr(size_t rows, const uint32_t *row_ptr) {
    if (row_ptr == NULL) {
        return 0;
    }
    return (size_t)row_ptr[rows];
}

#ifdef NUMRS_SUITESPARSE_EMULATION

int numrs_suitesparse_csrmv(
    size_t rows,
    size_t cols,
    const uint32_t *row_ptr,
    const uint32_t *col_idx,
    const double *values,
    const double *vector,
    double *out) {
    if (rows == 0 || cols == 0) {
        return NUMRS_STATUS_BAD_SHAPE;
    }
    const size_t nnz = nnz_from_row_ptr(rows, row_ptr);
    for (size_t r = 0; r < rows; ++r) {
        out[r] = 0.0;
        const size_t start = (size_t)row_ptr[r];
        const size_t end = (size_t)row_ptr[r + 1];
        if (end > nnz || start > end) {
            return NUMRS_STATUS_OUT_OF_BOUNDS;
        }
        for (size_t idx = start; idx < end; ++idx) {
            const size_t c = (size_t)col_idx[idx];
            if (c >= cols) {
                return NUMRS_STATUS_OUT_OF_BOUNDS;
            }
            out[r] += values[idx] * vector[c];
        }
    }
    return NUMRS_STATUS_SUCCESS;
}

int numrs_suitesparse_csrgemm(
    size_t rows,
    size_t cols,
    size_t rhs_cols,
    const uint32_t *row_ptr,
    const uint32_t *col_idx,
    const double *values,
    const double *rhs,
    double *out) {
    if (rows == 0 || cols == 0 || rhs_cols == 0) {
        return NUMRS_STATUS_BAD_SHAPE;
    }
    const size_t nnz = nnz_from_row_ptr(rows, row_ptr);
    for (size_t r = 0; r < rows * rhs_cols; ++r) {
        out[r] = 0.0;
    }
    for (size_t r = 0; r < rows; ++r) {
        const size_t start = (size_t)row_ptr[r];
        const size_t end = (size_t)row_ptr[r + 1];
        if (end > nnz || start > end) {
            return NUMRS_STATUS_OUT_OF_BOUNDS;
        }
        for (size_t idx = start; idx < end; ++idx) {
            const size_t c = (size_t)col_idx[idx];
            if (c >= cols) {
                return NUMRS_STATUS_OUT_OF_BOUNDS;
            }
            const double a_val = values[idx];
            const size_t rhs_row_offset = c * rhs_cols;
            double *out_row = &out[r * rhs_cols];
            const double *rhs_row = &rhs[rhs_row_offset];
            for (size_t rc = 0; rc < rhs_cols; ++rc) {
                out_row[rc] += a_val * rhs_row[rc];
            }
        }
    }
    return NUMRS_STATUS_SUCCESS;
}

int numrs_suitesparse_csradd(
    size_t rows,
    size_t cols,
    const uint32_t *row_ptr,
    const uint32_t *col_idx,
    const double *values,
    const double *dense,
    double *out) {
    if (rows == 0 || cols == 0) {
        return NUMRS_STATUS_BAD_SHAPE;
    }
    const size_t total = rows * cols;
    for (size_t i = 0; i < total; ++i) {
        out[i] = dense[i];
    }
    const size_t nnz = nnz_from_row_ptr(rows, row_ptr);
    for (size_t r = 0; r < rows; ++r) {
        const size_t start = (size_t)row_ptr[r];
        const size_t end = (size_t)row_ptr[r + 1];
        if (end > nnz || start > end) {
            return NUMRS_STATUS_OUT_OF_BOUNDS;
        }
        for (size_t idx = start; idx < end; ++idx) {
            const size_t c = (size_t)col_idx[idx];
            if (c >= cols) {
                return NUMRS_STATUS_OUT_OF_BOUNDS;
            }
            out[r * cols + c] += values[idx];
        }
    }
    return NUMRS_STATUS_SUCCESS;
}

int numrs_suitesparse_csrtranspose(
    size_t rows,
    size_t cols,
    const uint32_t *row_ptr,
    const uint32_t *col_idx,
    const double *values,
    double *out) {
    if (rows == 0 || cols == 0) {
        return NUMRS_STATUS_BAD_SHAPE;
    }
    const size_t total = rows * cols;
    for (size_t i = 0; i < total; ++i) {
        out[i] = 0.0;
    }
    const size_t nnz = nnz_from_row_ptr(rows, row_ptr);
    for (size_t r = 0; r < rows; ++r) {
        const size_t start = (size_t)row_ptr[r];
        const size_t end = (size_t)row_ptr[r + 1];
        if (end > nnz || start > end) {
            return NUMRS_STATUS_OUT_OF_BOUNDS;
        }
        for (size_t idx = start; idx < end; ++idx) {
            const size_t c = (size_t)col_idx[idx];
            if (c >= cols) {
                return NUMRS_STATUS_OUT_OF_BOUNDS;
            }
            out[c * rows + r] = values[idx];
        }
    }
    return NUMRS_STATUS_SUCCESS;
}

#else

#include <limits.h>
#include "cs.h"

static int csr_to_csc(
    size_t rows,
    size_t cols,
    const uint32_t *row_ptr,
    const uint32_t *col_idx,
    const double *values,
    cs **out_mat) {
    if (rows > (size_t)INT_MAX || cols > (size_t)INT_MAX) {
        return NUMRS_STATUS_BAD_SHAPE;
    }
    const size_t nnz = nnz_from_row_ptr(rows, row_ptr);
    if (nnz > (size_t)INT_MAX) {
        return NUMRS_STATUS_BAD_SHAPE;
    }

    cs *mat = cs_spalloc((CS_INT)rows, (CS_INT)cols, (CS_INT)nnz, 1, 0);
    if (mat == NULL) {
        return NUMRS_STATUS_ALLOC_FAILED;
    }

    CS_INT *col_ptr = mat->p;
    CS_INT *row_idx = mat->i;
    double *vals = mat->x;

    memset(col_ptr, 0, (cols + 1) * sizeof(CS_INT));

    for (size_t r = 0; r < rows; ++r) {
        const size_t start = (size_t)row_ptr[r];
        const size_t end = (size_t)row_ptr[r + 1];
        if (end > nnz || start > end) {
            cs_spfree(mat);
            return NUMRS_STATUS_OUT_OF_BOUNDS;
        }
        for (size_t idx = start; idx < end; ++idx) {
            const uint32_t c = col_idx[idx];
            if ((size_t)c >= cols) {
                cs_spfree(mat);
                return NUMRS_STATUS_OUT_OF_BOUNDS;
            }
            col_ptr[c + 1]++;
        }
    }

    for (size_t c = 0; c < cols; ++c) {
        col_ptr[c + 1] += col_ptr[c];
    }

    size_t *next = (size_t *)calloc(cols, sizeof(size_t));
    if (next == NULL) {
        cs_spfree(mat);
        return NUMRS_STATUS_ALLOC_FAILED;
    }
    for (size_t c = 0; c < cols; ++c) {
        next[c] = (size_t)col_ptr[c];
    }

    for (size_t r = 0; r < rows; ++r) {
        const size_t start = (size_t)row_ptr[r];
        const size_t end = (size_t)row_ptr[r + 1];
        for (size_t idx = start; idx < end; ++idx) {
            const uint32_t c = col_idx[idx];
            size_t dest = next[c]++;
            row_idx[dest] = (CS_INT)r;
            vals[dest] = values[idx];
        }
    }

    free(next);
    mat->nz = -1;
    *out_mat = mat;
    return NUMRS_STATUS_SUCCESS;
}

int numrs_suitesparse_csrmv(
    size_t rows,
    size_t cols,
    const uint32_t *row_ptr,
    const uint32_t *col_idx,
    const double *values,
    const double *vector,
    double *out) {
    if (rows == 0 || cols == 0) {
        return NUMRS_STATUS_BAD_SHAPE;
    }
    cs *mat = NULL;
    int status = csr_to_csc(rows, cols, row_ptr, col_idx, values, &mat);
    if (status != NUMRS_STATUS_SUCCESS) {
        return status;
    }

    memset(out, 0, rows * sizeof(double));
    int ok = cs_gaxpy(mat, vector, out);
    cs_spfree(mat);
    return ok ? NUMRS_STATUS_SUCCESS : NUMRS_STATUS_INTERNAL;
}

int numrs_suitesparse_csrgemm(
    size_t rows,
    size_t cols,
    size_t rhs_cols,
    const uint32_t *row_ptr,
    const uint32_t *col_idx,
    const double *values,
    const double *rhs,
    double *out) {
    if (rows == 0 || cols == 0 || rhs_cols == 0) {
        return NUMRS_STATUS_BAD_SHAPE;
    }
    cs *mat = NULL;
    int status = csr_to_csc(rows, cols, row_ptr, col_idx, values, &mat);
    if (status != NUMRS_STATUS_SUCCESS) {
        return status;
    }

    double *rhs_col = (double *)malloc(cols * sizeof(double));
    double *acc = (double *)malloc(rows * sizeof(double));
    if (rhs_col == NULL || acc == NULL) {
        free(rhs_col);
        free(acc);
        cs_spfree(mat);
        return NUMRS_STATUS_ALLOC_FAILED;
    }

    int result_status = NUMRS_STATUS_SUCCESS;
    for (size_t col = 0; col < rhs_cols; ++col) {
        for (size_t r = 0; r < cols; ++r) {
            rhs_col[r] = rhs[r * rhs_cols + col];
        }
        memset(acc, 0, rows * sizeof(double));
        if (!cs_gaxpy(mat, rhs_col, acc)) {
            result_status = NUMRS_STATUS_INTERNAL;
            break;
        }
        for (size_t r = 0; r < rows; ++r) {
            out[r * rhs_cols + col] = acc[r];
        }
    }

    free(rhs_col);
    free(acc);
    cs_spfree(mat);
    return result_status;
}

int numrs_suitesparse_csradd(
    size_t rows,
    size_t cols,
    const uint32_t *row_ptr,
    const uint32_t *col_idx,
    const double *values,
    const double *dense,
    double *out) {
    if (rows == 0 || cols == 0) {
        return NUMRS_STATUS_BAD_SHAPE;
    }
    const size_t total = rows * cols;
    memcpy(out, dense, total * sizeof(double));
    const size_t nnz = nnz_from_row_ptr(rows, row_ptr);
    for (size_t r = 0; r < rows; ++r) {
        const size_t start = (size_t)row_ptr[r];
        const size_t end = (size_t)row_ptr[r + 1];
        if (end > nnz || start > end) {
            return NUMRS_STATUS_OUT_OF_BOUNDS;
        }
        for (size_t idx = start; idx < end; ++idx) {
            const uint32_t c = col_idx[idx];
            if ((size_t)c >= cols) {
                return NUMRS_STATUS_OUT_OF_BOUNDS;
            }
            out[r * cols + c] += values[idx];
        }
    }
    return NUMRS_STATUS_SUCCESS;
}

int numrs_suitesparse_csrtranspose(
    size_t rows,
    size_t cols,
    const uint32_t *row_ptr,
    const uint32_t *col_idx,
    const double *values,
    double *out) {
    if (rows == 0 || cols == 0) {
        return NUMRS_STATUS_BAD_SHAPE;
    }
    cs *mat = NULL;
    int status = csr_to_csc(rows, cols, row_ptr, col_idx, values, &mat);
    if (status != NUMRS_STATUS_SUCCESS) {
        return status;
    }

    cs *transpose = cs_transpose(mat, 1);
    cs_spfree(mat);
    if (transpose == NULL) {
        return NUMRS_STATUS_INTERNAL;
    }

    const size_t total = rows * cols;
    memset(out, 0, total * sizeof(double));

    for (CS_INT col = 0; col < transpose->n; ++col) {
        for (CS_INT p = transpose->p[col]; p < transpose->p[col + 1]; ++p) {
            CS_INT row = transpose->i[p];
            if (row < 0 || row >= transpose->m) {
                cs_spfree(transpose);
                return NUMRS_STATUS_INTERNAL;
            }
            size_t idx = ((size_t)row) * (size_t)transpose->n + (size_t)col;
            out[idx] = transpose->x[p];
        }
    }

    cs_spfree(transpose);
    return NUMRS_STATUS_SUCCESS;
}

#endif
