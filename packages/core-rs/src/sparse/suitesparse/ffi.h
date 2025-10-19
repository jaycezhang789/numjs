#ifndef NUMRS_SUITESPARSE_FFI_H
#define NUMRS_SUITESPARSE_FFI_H

#include <stddef.h>
#include <stdint.h>

#define NUMRS_STATUS_SUCCESS 0
#define NUMRS_STATUS_BAD_SHAPE -1
#define NUMRS_STATUS_OUT_OF_BOUNDS -2
#define NUMRS_STATUS_ALLOC_FAILED -3
#define NUMRS_STATUS_INTERNAL -4

int numrs_suitesparse_csrmv(
    size_t rows,
    size_t cols,
    const uint32_t *row_ptr,
    const uint32_t *col_idx,
    const double *values,
    const double *vector,
    double *out);

int numrs_suitesparse_csrgemm(
    size_t rows,
    size_t cols,
    size_t rhs_cols,
    const uint32_t *row_ptr,
    const uint32_t *col_idx,
    const double *values,
    const double *rhs,
    double *out);

int numrs_suitesparse_csradd(
    size_t rows,
    size_t cols,
    const uint32_t *row_ptr,
    const uint32_t *col_idx,
    const double *values,
    const double *dense,
    double *out);

int numrs_suitesparse_csrtranspose(
    size_t rows,
    size_t cols,
    const uint32_t *row_ptr,
    const uint32_t *col_idx,
    const double *values,
    double *out);

#endif
