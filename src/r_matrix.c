#include <rc/r_matrix.h>
#include <rc/r_types.h>

RMatrix *r_create_matrix(size_t rows, size_t cols)
{
    RMatrix *matrix = malloc(sizeof(RMatrix));

    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = malloc(sizeof(float) * matrix->rows * matrix->cols);

    return matrix;
}

void r_free_matrix(RNONNULL RMatrix *matrix)
{
    free(matrix->data);
    matrix->rows = 0;
    matrix->cols = 0;
    free(matrix);
}

RMatrix *r_mat_mul(const RNONNULL RMatrix *mat1, const RNONNULL RMatrix *mat2)
{
    RMatrix *result = r_create_matrix(mat1->rows, mat2->cols);

    for (size_t i = 0; i < mat1->rows; i++)
        for (size_t j = 0; j < mat2->cols; j++)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < mat2->rows; k++)
            {
                sum += mat1->data[RMatrixIDX(i, k, mat1->cols)] * mat2->data[RMatrixIDX(k, j, mat2->cols)];
            }
            result->data[RMatrixIDX(i, j, result->cols)] = sum;
        }
    return result;
}

RMatrix *r_mat_transpose(const RNONNULL RMatrix *matrix)
{
    RMatrix *transposed_matrix = r_create_matrix(matrix->cols, matrix->rows);
    for (size_t i = 0; i < matrix->rows; i++)
    {
        for (size_t j = 0; j < matrix->cols; j++)
        {
            transposed_matrix->data[RMatrixIDX(j, i, transposed_matrix->cols)] =
                matrix->data[RMatrixIDX(i, j, matrix->cols)];
        }
    }
    return transposed_matrix;
}

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