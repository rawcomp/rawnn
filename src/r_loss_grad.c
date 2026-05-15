#include <rc/r_loss_grad.h>

/**
 * r_cross_entropy_grad() - Compute cross-entropy gradient.
 * @pred: Predicted probabilities.
 * @real: One-hot or target distribution matrix.
 * @grad: Output gradient matrix to fill.
 *
 * Writes the element-wise gradient for cross-entropy loss normalized by
 * the number of rows.
 * Return: Nothing.
 */
void r_cross_entropy_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, RMatrix *grad)
{
    const float norm = (float)pred->rows;
    const size_t count = MatrixSize(pred);
    for (size_t i = 0; i < count; i++)
    {
        const float p = pred->data[i];
        const float y = real->data[i];

        grad->data[i] = -y / (norm * (p + EPSILON));
    }
}

/**
 * r_cat_cross_entropy_grad() - Compute categorical cross-entropy gradient.
 * @pred: Predicted class probabilities.
 * @real: One-hot encoded target labels.
 * @grad: Output gradient matrix to fill.
 *
 * Writes the element-wise gradient for categorical cross-entropy loss
 * normalized by the number of rows.
 * Return: Nothing.
 */
void r_cat_cross_entropy_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, RMatrix *grad)
{
    const float norm = (float)pred->rows;
    const size_t count = MatrixSize(pred);
    for (size_t i = 0; i < count; i++)
    {
        const float p = pred->data[i];
        const float y = real->data[i];

        grad->data[i] = -y / (norm * (p + EPSILON));
    }
}

/**
 * r_bin_cross_entropy_grad() - Compute binary cross-entropy gradient.
 * @pred: Predicted probabilities.
 * @real: Target labels in {0, 1}.
 * @grad: Output gradient matrix to fill.
 *
 * Writes the element-wise gradient for binary cross-entropy loss
 * normalized by the total number of elements.
 * Return: Nothing.
 */
void r_bin_cross_entropy_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, RMatrix *grad)
{
    const float norm = (float)(pred->rows * pred->cols);
    const size_t count = MatrixSize(pred);
    for (size_t i = 0; i < count; i++)
    {
        const float p = pred->data[i];
        const float y = real->data[i];

        const float num_pos = -y / (p + EPSILON);
        const float num_neg = (1.0f - y) / (1.0f - p + EPSILON);

        grad->data[i] = (num_pos + num_neg) / norm;
    }
}

/**
 * r_mse_loss_grad() - Compute mean squared error gradient.
 * @pred: Predicted values.
 * @real: Target values.
 * @grad: Output gradient matrix to fill.
 *
 * Writes the element-wise gradient for MSE scaled by 2/n where n is the
 * number of elements.
 * Return: Nothing.
 */
void r_mse_loss_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, RMatrix *grad)
{
    const float norm = (float)(pred->rows * pred->cols);
    const float scale = 2.0f / norm;
    const size_t count = MatrixSize(pred);
    for (size_t i = 0; i < count; i++)
    {
        grad->data[i] = scale * (pred->data[i] - real->data[i]);
    }
}

/**
 * r_mae_loss_grad() - Compute mean absolute error gradient.
 * @pred: Predicted values.
 * @real: Target values.
 * @grad: Output gradient matrix to fill.
 *
 * Writes the element-wise gradient for MAE using the sign of the
 * difference and scales by 1/n where n is the number of elements.
 * Return: Nothing.
 */
void r_mae_loss_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, RMatrix *grad)
{
    const float norm = (float)(pred->rows * pred->cols);
    const float inv_norm = 1.0f / norm;
    const size_t count = MatrixSize(pred);
    for (size_t i = 0; i < count; i++)
    {
        const float diff = pred->data[i] - real->data[i];

        float sign = 0.0f;
        if (diff > 0.0f)
            sign = 1.0f;
        else if (diff < 0.0f)
            sign = -1.0f;

        grad->data[i] = sign * inv_norm;
    }
}

/**
 * r_bin_focal_loss_grad() - Compute binary focal loss gradient.
 * @pred: Predicted probabilities.
 * @real: Target labels in {0, 1}.
 * @gamma: Focusing parameter.
 * @alpha: Class weighting parameter.
 * @grad: Output gradient matrix to fill.
 *
 * Writes the element-wise gradient for binary focal loss normalized by
 * the total number of elements.
 * Return: Nothing.
 */
void r_bin_focal_loss_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, float gamma, float alpha,
                           RMatrix *grad)
{
    const float norm = (float)(pred->rows * pred->cols);
    const float inv_norm = 1.0f / norm;
    const size_t count = MatrixSize(pred);
    for (size_t i = 0; i < count; i++)
    {
        const float p = pred->data[i];
        const float y = real->data[i];

        const float omp = 1.0f - p;
        const float lp = logf(p + EPSILON);
        const float lop = logf(omp + EPSILON);

        const float omp_pow_gm1 = powf(omp, gamma - 1.0f);
        const float omp_pow_g = omp_pow_gm1 * omp;

        const float p_pow_gm1 = powf(p, gamma - 1.0f);
        const float p_pow_g = p_pow_gm1 * p;

        const float d_t1 = alpha * y * (-gamma * omp_pow_gm1 * lp + omp_pow_g / (p + EPSILON));

        const float d_t2 = (1.0f - alpha) * (1.0f - y) * (gamma * p_pow_gm1 * lop - p_pow_g / (omp + EPSILON));

        grad->data[i] = -(d_t1 + d_t2) * inv_norm;
    }
}

/**
 * r_cat_focal_loss_grad() - Compute categorical focal loss gradient.
 * @pred: Predicted class probabilities.
 * @real: One-hot encoded target labels.
 * @gamma: Focusing parameter.
 * @grad: Output gradient matrix to fill.
 *
 * Writes the element-wise gradient for categorical focal loss normalized
 * by the number of rows and only for the correct class per row.
 * Return: Nothing.
 */
void r_cat_focal_loss_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, float gamma, RMatrix *grad)
{
    const float norm = (float)pred->rows;
    const float inv_norm = 1.0f / norm;
    const size_t count = MatrixSize(pred);
    for (size_t i = 0; i < count; i++)
    {
        const float y = real->data[i];

        float g = 0.0f;
        if (y > 0.0f)
        {
            const float p = pred->data[i];
            const float omp = 1.0f - p;

            const float lp = logf(p + EPSILON);
            const float omp_pow_gm1 = powf(omp, gamma - 1.0f);
            const float omp_pow_g = omp_pow_gm1 * omp;

            g = -y * inv_norm * (-gamma * omp_pow_gm1 * lp + omp_pow_g / (p + EPSILON));
        }

        grad->data[i] = g;
    }
}
