#include <rc/r_loss.h>

float r_cross_entropy(const RNONNULL RMatrix *matrix, const RNONNULL RMatrix *src)
{
    float total = 0.0f;
    const size_t count = MatrixSize(matrix);
    for (size_t i = 0; i < count; i++)
    {
        float prediction = matrix->data[i];
        float correct = src->data[i];
        total += correct * log(prediction + EPSILON);
    }

    return -total / src->rows;
}

float r_bin_cross_entropy(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real)
{
    float total = 0.0f;
    const size_t count = MatrixSize(real);
    for (size_t i = 0; i < count; i++)
    {
        float currentReal = real->data[i];
        float currentPred = pred->data[i];
        total += currentReal * log(currentPred + EPSILON) + (1.0f - currentReal) * log(1.0f - currentPred + EPSILON);
    }

    return -total / (real->cols * real->rows);
}

float r_cat_cross_entropy(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real)
{
    float total = 0.0f;
    const size_t count = MatrixSize(real);
    for (size_t i = 0; i < count; i++)
    {
        float prediction = pred->data[i];
        float correct = real->data[i];
        total += correct * log(prediction + EPSILON);
    }

    return -total / real->rows;
}

float r_mse_loss(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real)
{
    float result = 0.0f;
    const size_t count = MatrixSize(real);
    for (size_t i = 0; i < count; i++)
    {
        float error = real->data[i] - pred->data[i];
        result += error * error;
    }

    return result / (real->cols * real->rows);
}

float r_mae_loss(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real)
{
    float result = 0.0f;
    const size_t count = MatrixSize(real);
    for (size_t i = 0; i < count; i++)
    {
        float error = real->data[i] - pred->data[i];
        result += fabsf(error);
    }

    return result / (real->cols * real->rows);
}

float r_bin_focal_loss(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, float gamma, float alpha)
{
    float total = 0.0f;
    const size_t count = MatrixSize(real);
    for (size_t i = 0; i < count; i++)
    {
        float y = real->data[i];
        float p = pred->data[i];

        float term1 = alpha * y * powf(1.0f - p, gamma) * logf(p + EPSILON);

        float term2 = (1.0f - alpha) * (1.0f - y) * powf(p, gamma) * logf(1.0f - p + EPSILON);

        total += term1 + term2;
    }

    return -total / (real->cols * real->rows);
}

float r_cat_focal_loss(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, float gamma)
{
    float total = 0.0f;
    const size_t count = MatrixSize(real);
    for (size_t i = 0; i < count; i++)
    {
        float prediction = pred->data[i];
        float correct = real->data[i];

        if (correct > 0.0f)
        {
            float mod_factor = powf(1.0f - prediction, gamma);
            total += correct * mod_factor * logf(prediction + EPSILON);
        }
    }

    return -total / real->rows;
}
