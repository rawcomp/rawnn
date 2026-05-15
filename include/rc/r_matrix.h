#ifndef RC_R_MATRIX_H
#define RC_R_MATRIX_H

#include <rc/r_types.h>

/**
 * struct r_matrix_t - Simple row-major matrix container.
 * @data: Contiguous data buffer of size @rows * @cols.
 * @rows: Number of rows.
 * @cols: Number of columns.
 */
typedef struct r_matrix_t
{
    float *data;
    size_t rows;
    size_t cols;
} RMatrix;

RMatrix *r_create_matrix(size_t rows, size_t cols);
void r_free_matrix(RNONNULL RMatrix *matrix);
RMatrix *r_mat_mul(const RNONNULL RMatrix *mat1, const RNONNULL RMatrix *mat2);
RMatrix *r_mat_transpose(const RNONNULL RMatrix *matrix);
void r_print_matrix(const RNONNULL RMatrix *m, const RNONNULL char *name);

#endif
