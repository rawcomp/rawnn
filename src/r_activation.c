#include <rc/r_activation.h>
#include <rc/r_matrix.h>
#include <rc/r_types.h>

RMatrix *r_activation_relu(RNONNULL RMatrix *matrix)
{
    RMatrix *result = r_create_matrix(matrix->rows, matrix->cols);
    for (size_t i = 0; i < matrix->rows; i++)
    {
        for (size_t j = 0; j < matrix->cols; j++)
        {
            float curr = matrix->data[RMatrixIDX(i, j, matrix->cols)];
            size_t currIndex = RMatrixIDX(i, j, matrix->cols);
            result->data[currIndex] = (curr > 0.0f) ? curr : 0.0f;
        }
    }
    return result;
}

RMatrix *r_activation_softmax(RNONNULL RMatrix *matrix)
{
    RMatrix *result = r_create_matrix(matrix->rows, matrix->cols);
    for (size_t i = 0; i < matrix->rows; i++)
    {
        float total = 0.0f;
        float hi = matrix->data[RMatrixIDX(i, 0, matrix->cols)];
        for (size_t j = 1; j < matrix->cols; j++)
        {
            if (hi < matrix->data[RMatrixIDX(i, j, matrix->cols)])
            {
                hi = matrix->data[RMatrixIDX(i, j, matrix->cols)];
            }
        }
        for (size_t k = 0; k < matrix->cols; k++)
        {
            result->data[RMatrixIDX(i, k, matrix->cols)] = matrix->data[RMatrixIDX(i, k, matrix->cols)] - hi;
            result->data[RMatrixIDX(i, k, matrix->cols)] = exp(result->data[RMatrixIDX(i, k, result->cols)]);
            total += result->data[RMatrixIDX(i, k, matrix->cols)];
        }

        for (size_t z = 0; z < matrix->cols; z++)
            result->data[RMatrixIDX(i, z, matrix->cols)] /= total;
    }
    return result;
}