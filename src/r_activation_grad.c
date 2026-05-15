#include <rc/r_activation_grad.h>

void r_activation_relu_grad(const RNONNULL RMatrix *input, const RNONNULL RMatrix *upstream_grad,
                            RNONNULL RMatrix *grad)
{
    const size_t count = MatrixSize(input);
    for (size_t i = 0; i < count; i++)
    {
        grad->data[i] = (input->data[i] > 0.0f) ? upstream_grad->data[i] : 0.0f;
    }
}

void r_activation_leaky_relu_grad(const RNONNULL RMatrix *input, const RNONNULL RMatrix *upstream_grad, float alpha,
                                  RNONNULL RMatrix *grad)
{
    const size_t count = MatrixSize(input);
    for (size_t i = 0; i < count; i++)
    {
        grad->data[i] = (input->data[i] > 0.0f) ? upstream_grad->data[i] : upstream_grad->data[i] * alpha;
    }
}

void r_activation_gelu_grad(const RNONNULL RMatrix *input, const RNONNULL RMatrix *upstream_grad,
                            RNONNULL RMatrix *grad)
{
    float bsqrt2 = 1 / M_SQRT2;
    float inv_sqrt_2pi = 1.0f / sqrtf(2.0f * M_PI);
    const size_t count = MatrixSize(input);
    for (size_t i = 0; i < count; i++)
    {
        float val = input->data[i];

        float cdf = 0.5f * (1.0f + erff(val * bsqrt2));
        float pdf = inv_sqrt_2pi * expf(-0.5f * val * val);

        float local_grad = cdf + (val * pdf);

        grad->data[i] = local_grad * upstream_grad->data[i];
    }
}

void r_activation_softmax_grad(const RNONNULL RMatrix *output, const RNONNULL RMatrix *upstream_grad,
                               RNONNULL RMatrix *grad)
{
    for (size_t i = 0; i < output->rows; i++)
    {
        float dot = 0.0f;
        for (size_t j = 0; j < output->cols; j++)
        {
            size_t index = RMatrixIDX(i, j, output->cols);
            dot += upstream_grad->data[index] * output->data[index];
        }

        for (size_t j = 0; j < output->cols; j++)
        {
            size_t index = RMatrixIDX(i, j, output->cols);
            grad->data[index] = output->data[index] * (upstream_grad->data[index] - dot);
        }
    }
}

void r_activation_swish_grad(const RNONNULL RMatrix *input, const RNONNULL RMatrix *upstream_grad,
                             RNONNULL RMatrix *grad)
{
    const size_t count = MatrixSize(input);
    for (size_t i = 0; i < count; i++)
    {
        float val = input->data[i];
        float sigmoid = 1.0f / (1.0f + expf(-val));
        float local_grad = sigmoid + val * sigmoid * (1.0f - sigmoid);

        grad->data[i] = upstream_grad->data[i] * local_grad;
    }
}
