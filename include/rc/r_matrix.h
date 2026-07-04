#ifndef RC_R_MATRIX_H
#define RC_R_MATRIX_H

#include <rc/r_types.h>
#include <stdio.h>

/**
 * struct r_matrix_t - Simple row-major matrix container.
 * @data: Contiguous data buffer of size @rows * @cols.
 * @rows: Number of rows.
 * @cols: Number of columns.
 */
typedef struct r_matrix_t {
	float *data;
	size_t rows;
	size_t cols;
} RMatrix;

RMatrix *r_create_matrix(size_t rows, size_t cols);
RMatrix *r_create_matrix_zeros(size_t rows, size_t cols);
RMatrix *r_create_matrix_from_data(size_t rows, size_t cols, const float *data);
void r_free_matrix(RMatrix *matrix);
RMatrix *r_matrix_mul(const RMatrix *mat1, const RMatrix *mat2);
RMatrix *r_matrix_transpose(const RMatrix *matrix);
void r_print_matrix(FILE *stream, const RMatrix *m, const char *name);

#endif
