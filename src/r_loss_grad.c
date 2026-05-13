#include <rc/r_loss_grad.h>

void r_cross_entropy_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, RMatrix *grad)
{
    const float norm = (float)pred->rows;

    for (size_t i = 0; i < pred->rows; i++)
    {
        for (size_t j = 0; j < pred->cols; j++)
        {
            const size_t idx_p = RMatrixIDX(i, j, pred->cols);
            const size_t idx_y = RMatrixIDX(i, j, real->cols);
            const size_t idx_g = RMatrixIDX(i, j, grad->cols);

            const float p = pred->data[idx_p];
            const float y = real->data[idx_y];

            grad->data[idx_g] = -y / (norm * (p + EPSILON));
        }
    }
}

void r_cat_cross_entropy_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, RMatrix *grad)
{
    const float norm = (float)pred->rows;

    for (size_t i = 0; i < pred->rows; i++)
    {
        for (size_t j = 0; j < pred->cols; j++)
        {
            const size_t idx_p = RMatrixIDX(i, j, pred->cols);
            const size_t idx_y = RMatrixIDX(i, j, real->cols);
            const size_t idx_g = RMatrixIDX(i, j, grad->cols);

            const float p = pred->data[idx_p];
            const float y = real->data[idx_y];

            grad->data[idx_g] = -y / (norm * (p + EPSILON));
        }
    }
}

void r_bin_cross_entropy_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, RMatrix *grad)
{
    const float norm = (float)(pred->rows * pred->cols);

    for (size_t i = 0; i < pred->rows; i++)
    {
        for (size_t j = 0; j < pred->cols; j++)
        {
            const size_t idx_p = RMatrixIDX(i, j, pred->cols);
            const size_t idx_y = RMatrixIDX(i, j, real->cols);
            const size_t idx_g = RMatrixIDX(i, j, grad->cols);

            const float p = pred->data[idx_p];
            const float y = real->data[idx_y];

            const float num_pos = -y / (p + EPSILON);
            const float num_neg = (1.0f - y) / (1.0f - p + EPSILON);

            grad->data[idx_g] = (num_pos + num_neg) / norm;
        }
    }
}

void r_mse_loss_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, RMatrix *grad)
{
    const float norm = (float)(pred->rows * pred->cols);
    const float scale = 2.0f / norm;

    for (size_t i = 0; i < pred->rows; i++)
    {
        for (size_t j = 0; j < pred->cols; j++)
        {
            const size_t idx_p = RMatrixIDX(i, j, pred->cols);
            const size_t idx_y = RMatrixIDX(i, j, real->cols);
            const size_t idx_g = RMatrixIDX(i, j, grad->cols);

            const float p = pred->data[idx_p];
            const float y = real->data[idx_y];

            grad->data[idx_g] = scale * (p - y);
        }
    }
}

void r_mae_loss_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, RMatrix *grad)
{
    const float norm = (float)(pred->rows * pred->cols);
    const float inv_norm = 1.0f / norm;

    for (size_t i = 0; i < pred->rows; i++)
    {
        for (size_t j = 0; j < pred->cols; j++)
        {
            const size_t idx_p = RMatrixIDX(i, j, pred->cols);
            const size_t idx_y = RMatrixIDX(i, j, real->cols);
            const size_t idx_g = RMatrixIDX(i, j, grad->cols);

            const float diff = pred->data[idx_p] - real->data[idx_y];

            float sign = 0.0f;
            if (diff > 0.0f)
                sign = 1.0f;
            else if (diff < 0.0f)
                sign = -1.0f;

            grad->data[idx_g] = sign * inv_norm;
        }
    }
}

void r_bin_focal_loss_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, float gamma, float alpha,
                           RMatrix *grad)
{
    const float norm = (float)(pred->rows * pred->cols);
    const float inv_norm = 1.0f / norm;

    for (size_t i = 0; i < pred->rows; i++)
    {
        for (size_t j = 0; j < pred->cols; j++)
        {
            const size_t idx_p = RMatrixIDX(i, j, pred->cols);
            const size_t idx_y = RMatrixIDX(i, j, real->cols);
            const size_t idx_g = RMatrixIDX(i, j, grad->cols);

            const float p = pred->data[idx_p];
            const float y = real->data[idx_y];

            const float omp = 1.0f - p;
            const float lp = logf(p + EPSILON);
            const float lop = logf(omp + EPSILON);

            const float omp_pow_gm1 = powf(omp, gamma - 1.0f);
            const float omp_pow_g = omp_pow_gm1 * omp;

            const float p_pow_gm1 = powf(p, gamma - 1.0f);
            const float p_pow_g = p_pow_gm1 * p;

            const float d_t1 = alpha * y * (-gamma * omp_pow_gm1 * lp + omp_pow_g / (p + EPSILON));

            const float d_t2 = (1.0f - alpha) * (1.0f - y) * (gamma * p_pow_gm1 * lop - p_pow_g / (omp + EPSILON));

            grad->data[idx_g] = -(d_t1 + d_t2) * inv_norm;
        }
    }
}

void r_cat_focal_loss_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, float gamma, RMatrix *grad)
{
    const float norm = (float)pred->rows;
    const float inv_norm = 1.0f / norm;

    for (size_t i = 0; i < pred->rows; i++)
    {
        for (size_t j = 0; j < pred->cols; j++)
        {
            const size_t idx_p = RMatrixIDX(i, j, pred->cols);
            const size_t idx_y = RMatrixIDX(i, j, real->cols);
            const size_t idx_g = RMatrixIDX(i, j, grad->cols);

            const float y = real->data[idx_y];

            float g = 0.0f;
            if (y > 0.0f)
            {
                const float p = pred->data[idx_p];
                const float omp = 1.0f - p;

                const float lp = logf(p + EPSILON);
                const float omp_pow_gm1 = powf(omp, gamma - 1.0f);
                const float omp_pow_g = omp_pow_gm1 * omp;

                g = -y * inv_norm * (-gamma * omp_pow_gm1 * lp + omp_pow_g / (p + EPSILON));
            }

            grad->data[idx_g] = g;
        }
    }
}
