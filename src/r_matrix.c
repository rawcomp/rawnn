#include <rc/r_matrix.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * r_create_matrix() - Allocate a matrix.
 * @rows: Number of rows.
 * @cols: Number of columns.
 *
 * Allocates the matrix structure and a contiguous data buffer of
 * @rows * @cols floats. The data buffer is not initialized.
 * Return: Pointer to the newly allocated matrix.
 */
RMatrix *r_create_matrix(size_t rows, size_t cols)
{
	RMatrix *matrix;
	size_t total_elements;

	if (rows > 0 && cols > SIZE_MAX / rows)
		return NULL;

	total_elements = rows * cols;

	matrix = malloc(sizeof(*matrix));
	if (!matrix)
		return NULL;

	matrix->rows = rows;
	matrix->cols = cols;

	if (total_elements == 0) {
		matrix->data = NULL;
		return matrix;
	}

	matrix->data = malloc(sizeof(*matrix->data) * total_elements);
	if (!matrix->data) {
		free(matrix);
		return NULL;
	}

	return matrix;
}

/**
 * r_create_matrix_zeros() - Allocate a matrix initialized to zero.
 * @rows: Number of rows.
 * @cols: Number of columns.
 *
 * Return: Pointer to the newly allocated and zeroed matrix.
 */
RMatrix *r_create_matrix_zeros(size_t rows, size_t cols)
{
	RMatrix *matrix;
	size_t total_elements;

	if (rows > 0 && cols > SIZE_MAX / rows)
		return NULL;

	total_elements = rows * cols;

	matrix = malloc(sizeof(*matrix));
	if (!matrix)
		return NULL;

	matrix->rows = rows;
	matrix->cols = cols;

	if (total_elements == 0) {
		matrix->data = NULL;
		return matrix;
	}

	matrix->data = calloc(total_elements, sizeof(*matrix->data));
	if (!matrix->data) {
		free(matrix);
		return NULL;
	}

	return matrix;
}

/**
 * r_create_matrix_from_data() - Allocate a matrix and copy data.
 * @rows: Number of rows.
 * @cols: Number of columns.
 * @data: Source data array.
 *
 * Return: Pointer to the newly allocated matrix containing @data.
 */
RMatrix *r_create_matrix_from_data(size_t rows, size_t cols, const float *data)
{
	RMatrix *matrix;
	size_t total_elements;

	if (!data)
		return NULL;

	matrix = r_create_matrix(rows, cols);
	if (!matrix)
		return NULL;

	total_elements = rows * cols;
	if (total_elements > 0)
		memcpy(matrix->data, data, total_elements * sizeof(float));

	return matrix;
}

/**
 * r_free_matrix() - Free a matrix and its data buffer.
 * @matrix: Matrix to free.
 *
 * Releases the matrix data buffer, clears the dimensions, and frees the
 * matrix structure.
 * Return: Nothing.
 */
void r_free_matrix(RMatrix *matrix)
{
	if (!matrix)
		return;

	free(matrix->data);
	matrix->rows = 0;
	matrix->cols = 0;
	free(matrix);
}

/**
 * r_matrix_mul() - Multiply two matrices.
 * @mat1: Left-hand matrix.
 * @mat2: Right-hand matrix.
 *
 * Computes @mat1 * @mat2 without dimension checking. The caller must
 * ensure @mat1->cols equals @mat2->rows.
 * Return: Newly allocated matrix containing the product.
 */
RMatrix *r_matrix_mul(const RMatrix *mat1, const RMatrix *mat2)
{
	RMatrix *result;
	size_t i;
	size_t j;
	size_t k;

	if (!mat1 || !mat2 || mat1->cols != mat2->rows) {
		fprintf(stderr, "[ERROR]: r_matrix_mul requires mat1->cols == "
				"mat2->rows\n");
		return NULL;
	}

	result = r_create_matrix_zeros(mat1->rows, mat2->cols);
	if (!result)
		return NULL;

	for (i = 0; i < mat1->rows; i++) {
		const float *row1 = &mat1->data[i * mat1->cols];
		float *row_res = &result->data[i * result->cols];

		for (k = 0; k < mat1->cols; k++) {
			float val1 = row1[k];
			const float *row2 = &mat2->data[k * mat2->cols];

			for (j = 0; j < mat2->cols; j++)
				row_res[j] += val1 * row2[j];
		}
	}

	return result;
}

#define TILE_SIZE 32

/**
 * r_matrix_transpose() - Transpose a matrix.
 * @matrix: Matrix to transpose.
 *
 * Allocates a new matrix with swapped dimensions and copies the data in
 * transposed order using cache-friendly block tiling.
 * Return: Newly allocated transposed matrix.
 */
RMatrix *r_matrix_transpose(const RMatrix *matrix)
{
	RMatrix *transposed;
	size_t i;
	size_t j;
	size_t ii;
	size_t jj;

	if (!matrix)
		return NULL;

	transposed = r_create_matrix(matrix->cols, matrix->rows);
	if (!transposed)
		return NULL;

	for (i = 0; i < matrix->rows; i += TILE_SIZE) {
		for (j = 0; j < matrix->cols; j += TILE_SIZE) {
			size_t max_i = (i + TILE_SIZE < matrix->rows)
					   ? i + TILE_SIZE
					   : matrix->rows;
			size_t max_j = (j + TILE_SIZE < matrix->cols)
					   ? j + TILE_SIZE
					   : matrix->cols;

			for (ii = i; ii < max_i; ii++) {
				const float *src_row =
				    &matrix->data[ii * matrix->cols];
				for (jj = j; jj < max_j; jj++)
					transposed
					    ->data[jj * transposed->cols + ii] =
					    src_row[jj];
			}
		}
	}

	return transposed;
}

/**
 * r_print_matrix(stdout, ) - Print a matrix with a label.
 * @stream: Output stream.
 * @m: Matrix to print.
 * @name: Label to print before the matrix.
 *
 * Outputs the matrix dimensions and values to @stream with a fixed
 * two-decimal format.
 * Return: Nothing.
 */
void r_print_matrix(FILE *stream, const RMatrix *m, const char *name)
{
	size_t i;
	size_t j;

	if (!stream || !m || !name)
		return;

	fprintf(stream, "%s (%zux%zu):\n", name, m->rows, m->cols);
	for (i = 0; i < m->rows; i++) {
		for (j = 0; j < m->cols; j++)
			fprintf(stream, "%.2f ",
				m->data[R_MATRIX_IDX(i, j, m->cols)]);
		fprintf(stream, "\n");
	}
}
