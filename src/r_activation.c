#include <rc/r_activation.h>

/**
 * r_activation_relu() - Apply ReLU activation to a matrix.
 * @matrix: Input matrix to transform.
 *
 * Allocates a new matrix with the same shape as @matrix and replaces
 * each element with max(0, x).
 * Return: Newly allocated matrix containing the ReLU result.
 */
RMatrix *r_activation_relu(RNONNULL RMatrix *matrix)
{
    RMatrix *result = r_create_matrix(matrix->rows, matrix->cols);
    const size_t count = MatrixSize(matrix);
    for (size_t i = 0; i < count; i++)
    {
        float curr = matrix->data[i];
        result->data[i] = (curr > 0.0f) ? curr : 0.0f;
    }
    return result;
}

/**
 * r_activation_leaky_relu() - Apply leaky ReLU activation to a matrix.
 * @matrix: Input matrix to transform.
 * @alpha: Slope used for negative values.
 *
 * Allocates a new matrix with the same shape as @matrix and applies a
 * piecewise linear activation with slope @alpha for negative inputs.
 * Return: Newly allocated matrix containing the leaky ReLU result.
 */
RMatrix *r_activation_leaky_relu(RNONNULL RMatrix *matrix, float alpha)
{
    RMatrix *result = r_create_matrix(matrix->rows, matrix->cols);
    const size_t count = MatrixSize(matrix);
    for (size_t i = 0; i < count; i++)
    {
        float val = matrix->data[i];
        result->data[i] = (val > 0.0f) ? val : val * alpha;
    }

    return result;
}

/**
 * r_activation_gelu() - Apply GELU activation to a matrix.
 * @matrix: Input matrix to transform.
 *
 * Allocates a new matrix with the same shape as @matrix and applies the
 * Gaussian error linear unit using erf() for the CDF term.
 * Return: Newly allocated matrix containing the GELU result.
 */
RMatrix *r_activation_gelu(RNONNULL RMatrix *matrix)
{
    RMatrix *result = r_create_matrix(matrix->rows, matrix->cols);
    float bsqr2 = 1 / M_SQRT2;
    const size_t count = MatrixSize(matrix);
    for (size_t i = 0; i < count; i++)
    {
        float term = matrix->data[i] * bsqr2;

        float error = erff(term);
        float prob = 0.5f * (1.0f + error);

        result->data[i] = matrix->data[i] * prob;
    }
    return result;
}

/**
 * r_activation_softmax() - Apply row-wise softmax activation.
 * @matrix: Input matrix to transform.
 *
 * Allocates a new matrix with the same shape as @matrix and computes a
 * softmax for each row, subtracting the row maximum for stability.
 * Return: Newly allocated matrix containing the softmax result.
 */
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
            result->data[RMatrixIDX(i, k, matrix->cols)] = exp(result->data[RMatrixIDX(i, k, matrix->cols)]);
            total += result->data[RMatrixIDX(i, k, matrix->cols)];
        }

        for (size_t z = 0; z < matrix->cols; z++)
            result->data[RMatrixIDX(i, z, matrix->cols)] /= total;
    }
    return result;
}

/**
 * r_activation_swish() - Apply swish activation to a matrix.
 * @matrix: Input matrix to transform.
 *
 * Allocates a new matrix with the same shape as @matrix and computes
 * x * sigmoid(x) for each element.
 * Return: Newly allocated matrix containing the swish result.
 */
RMatrix *r_activation_swish(RNONNULL RMatrix *matrix)
{
    RMatrix *result = r_create_matrix(matrix->rows, matrix->cols);
    const size_t count = MatrixSize(matrix);
    for (size_t i = 0; i < count; i++)
    {
        float val = matrix->data[i];
        float sigmoid = 1.0f / (1.0f + expf(-val));

        result->data[i] = val * sigmoid;
    }

    return result;
}