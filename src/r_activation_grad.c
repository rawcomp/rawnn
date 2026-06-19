#include <rc/r_activation_grad.h>

/**
 * r_activation_relu_grad() - Compute ReLU gradient.
 * @input: Input matrix from the forward pass.
 * @upstream_grad: Gradient flowing from the next layer.
 * @grad: Output gradient matrix to fill.
 *
 * Writes the element-wise gradient into @grad by masking negative inputs
 * and scaling by @upstream_grad.
 * Return: Nothing.
 */
void r_activation_relu_grad(const RNONNULL RMatrix *input, const RNONNULL RMatrix *upstream_grad,
                            RNONNULL RMatrix *grad)
{
    const size_t count = MatrixSize(input);
    for (size_t i = 0; i < count; i++)
    {
        grad->data[i] = (input->data[i] > 0.0f) ? upstream_grad->data[i] : 0.0f;
    }
}

/**
 * r_activation_leaky_relu_grad() - Compute leaky ReLU gradient.
 * @input: Input matrix from the forward pass.
 * @upstream_grad: Gradient flowing from the next layer.
 * @alpha: Slope used for negative values.
 * @grad: Output gradient matrix to fill.
 *
 * Writes the element-wise gradient into @grad, scaling negative inputs by
 * @alpha while passing positive gradients through unchanged.
 * Return: Nothing.
 */
void r_activation_leaky_relu_grad(const RNONNULL RMatrix *input, const RNONNULL RMatrix *upstream_grad, float alpha,
                                  RNONNULL RMatrix *grad)
{
    const size_t count = MatrixSize(input);
    for (size_t i = 0; i < count; i++)
    {
        grad->data[i] = (input->data[i] > 0.0f) ? upstream_grad->data[i] : upstream_grad->data[i] * alpha;
    }
}

/**
 * r_activation_gelu_grad() - Compute GELU gradient.
 * @input: Input matrix from the forward pass.
 * @upstream_grad: Gradient flowing from the next layer.
 * @grad: Output gradient matrix to fill.
 *
 * Computes the GELU derivative using the normal CDF and PDF, then scales
 * by @upstream_grad into @grad.
 * Return: Nothing.
 */
void r_activation_gelu_grad(const RNONNULL RMatrix *input, const RNONNULL RMatrix *upstream_grad,
                            RNONNULL RMatrix *grad)
{
    const float bsqrt2 = (float)(1.0 / M_SQRT2);
    const float inv_sqrt_2pi = 1.0f / sqrtf(2.0f * (float)M_PI);
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

/**
 * r_activation_softmax_grad() - Compute softmax gradient.
 * @output: Softmax output from the forward pass.
 * @upstream_grad: Gradient flowing from the next layer.
 * @grad: Output gradient matrix to fill.
 *
 * Computes the Jacobian-vector product for softmax per row using
 * @output and @upstream_grad, writing the result to @grad.
 * Return: Nothing.
 */
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

/**
 * r_activation_swish_grad() - Compute swish gradient.
 * @input: Input matrix from the forward pass.
 * @upstream_grad: Gradient flowing from the next layer.
 * @grad: Output gradient matrix to fill.
 *
 * Computes the swish derivative using the sigmoid term and writes the
 * scaled gradient into @grad.
 * Return: Nothing.
 */
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
