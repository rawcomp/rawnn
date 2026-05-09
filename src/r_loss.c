#include <rc/r_loss.h>
#include <rc/r_matrix.h>
#include <rc/r_types.h>

float r_cross_entropy(const RNONNULL RMatrix *matrix, const RNONNULL RMatrix *src)
{
    float total = 0.0f;

    for (size_t i = 0; i < matrix->rows; i++)
    {
        for (size_t j = 0; j < matrix->cols; j++)
        {
            float prediction = matrix->data[RMatrixIDX(i, j, matrix->cols)];
            float correct = src->data[RMatrixIDX(i, j, src->cols)];
            total += correct * log(prediction + EPSILON);
        }
    }

    return -total / src->rows;
}

float r_mse_loss(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real)
{
    float result = 0.0f;

    for (size_t i = 0; i < real->rows; i++)
    {
        for (size_t j = 0; j < real->cols; j++)
        {
            float error = (real->data[RMatrixIDX(i, j, real->cols)] - pred->data[RMatrixIDX(i, j, pred->cols)]);
            result += error * error;
        }
    }

    return result / (real->cols * real->rows);
}

float r_mae_loss(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real)
{
    float result = 0.0f;
    for (size_t i = 0; i < real->rows; i++)
    {
        for (size_t j = 0; j < real->cols; j++)
        {
            float error = (real->data[RMatrixIDX(i, j, real->cols)] - pred->data[RMatrixIDX(i, j, pred->cols)]);
            result += fabsf(error);
        }
    }

    return result / (real->cols * real->rows);
}
