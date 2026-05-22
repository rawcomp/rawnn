#include <rc/r_matrix.h>

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
    RMatrix *matrix = malloc(sizeof(RMatrix));
    if (!matrix)
        return NULL;

    matrix->rows = rows;
    matrix->cols = cols;

    size_t total_elements = rows * cols;

    if (total_elements == 0)
    {
        matrix->data = NULL;
        return matrix;
    }

    // Check for overflow before allocation
    if (rows != 0 && total_elements / rows != cols)
    {
        free(matrix);
        return NULL;
    }

    matrix->data = malloc(sizeof(float) * total_elements);
    if (!matrix->data)
    {
        free(matrix);
        return NULL;
    }

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
void r_free_matrix(RNONNULL RMatrix *matrix)
{
    if (!matrix)
        return;
    free(matrix->data);
    matrix->rows = 0;
    matrix->cols = 0;
    free(matrix);
}

/**
 * r_mat_mul() - Multiply two matrices.
 * @mat1: Left-hand matrix.
 * @mat2: Right-hand matrix.
 *
 * Computes @mat1 * @mat2 without dimension checking. The caller must
 * ensure @mat1->cols equals @mat2->rows.
 * Return: Newly allocated matrix containing the product.
 */
RMatrix *r_mat_mul(const RNONNULL RMatrix *mat1, const RNONNULL RMatrix *mat2)
{
    if (mat1->cols != mat2->rows)
    {
        printf("[ERROR]: r_mat_mul requires mat1->cols == mat2->rows\n");
        return NULL;
    }

    RMatrix *result = r_create_matrix(mat1->rows, mat2->cols);
    if (!result)
        return NULL;

    size_t result_size = MatrixSize(result);
    for (size_t i = 0; i < result_size; i++)
        result->data[i] = 0.0f;

    // Cache-friendly i-k-j loop order
    for (size_t i = 0; i < mat1->rows; i++)
    {
        const float *row1 = &mat1->data[i * mat1->cols];
        float *row_res = &result->data[i * result->cols];

        for (size_t k = 0; k < mat1->cols; k++)
        {
            float val1 = row1[k];
            const float *row2 = &mat2->data[k * mat2->cols];
            for (size_t j = 0; j < mat2->cols; j++)
            {
                row_res[j] += val1 * row2[j];
            }
        }
    }
    return result;
}

/**
 * r_mat_transpose() - Transpose a matrix.
 * @matrix: Matrix to transpose.
 *
 * Allocates a new matrix with swapped dimensions and copies the data in
 * transposed order.
 * Return: Newly allocated transposed matrix.
 */
RMatrix *r_mat_transpose(const RNONNULL RMatrix *matrix)
{
    RMatrix *transposed_matrix = r_create_matrix(matrix->cols, matrix->rows);
    if (!transposed_matrix)
        return NULL;

    for (size_t i = 0; i < matrix->rows; i++)
    {
        const float *src_row = &matrix->data[i * matrix->cols];
        for (size_t j = 0; j < matrix->cols; j++)
        {
            transposed_matrix->data[j * transposed_matrix->cols + i] = src_row[j];
        }
    }
    return transposed_matrix;
}

/**
 * r_print_matrix() - Print a matrix with a label.
 * @m: Matrix to print.
 * @name: Label to print before the matrix.
 *
 * Outputs the matrix dimensions and values to stdout with a fixed
 * two-decimal format.
 * Return: Nothing.
 */
void r_print_matrix(const RNONNULL RMatrix *m, const RNONNULL char *name)
{
    printf("%s (%ldx%ld):\n", name, m->rows, m->cols);
    for (size_t i = 0; i < m->rows; i++)
    {
        for (size_t j = 0; j < m->cols; j++)
        {
            printf("%.2f ", m->data[RMatrixIDX(i, j, m->cols)]);
        }
        printf("\n");
    }
}
