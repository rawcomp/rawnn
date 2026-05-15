#include <rc/r_optimization.h>

/**
 * r_optimization_sgd() - Perform an SGD parameter update.
 * @theta: Parameter matrix updated in place.
 * @grad: Gradient matrix.
 * @eta: Learning rate.
 *
 * Applies a simple SGD update: theta -= eta * grad.
 * Return: Nothing.
 */
void r_optimization_sgd(RNONNULL RMatrix *theta, const RNONNULL RMatrix *grad, float eta)
{
    const size_t count = MatrixSize(theta);
    for (size_t i = 0; i < count; i++)
    {
        theta->data[i] -= eta * grad->data[i];
    }
}

/**
 * r_optimization_sgdm() - Perform SGD with momentum update.
 * @theta: Parameter matrix updated in place.
 * @grad: Gradient matrix.
 * @velocity: Momentum buffer updated in place.
 * @eta: Learning rate.
 * @beta: Momentum coefficient.
 *
 * Updates @velocity with an exponential moving average of @grad and
 * applies the momentum update to @theta.
 * Return: Nothing.
 */
void r_optimization_sgdm(RNONNULL RMatrix *theta, const RNONNULL RMatrix *grad, RNONNULL RMatrix *velocity, float eta,
                         float beta)
{
    const size_t count = MatrixSize(theta);
    for (size_t i = 0; i < count; i++)
    {
        velocity->data[i] = beta * velocity->data[i] + (1.0f - beta) * grad->data[i];
        theta->data[i] -= eta * velocity->data[i];
    }
}

/**
 * r_optimization_rmsprop() - Perform RMSProp parameter update.
 * @theta: Parameter matrix updated in place.
 * @grad: Gradient matrix.
 * @cache: Running average of squared gradients updated in place.
 * @eta: Learning rate.
 * @rho: Decay rate for the running average.
 * @eps: Small constant for numerical stability.
 *
 * Updates the running average of squared gradients and scales the
 * parameter update by the RMS term.
 * Return: Nothing.
 */
void r_optimization_rmsprop(RNONNULL RMatrix *theta, const RNONNULL RMatrix *grad, RNONNULL RMatrix *cache, float eta,
                            float rho, float eps)
{
    const size_t count = MatrixSize(theta);
    for (size_t i = 0; i < count; i++)
    {
        float g = grad->data[i];
        cache->data[i] = rho * cache->data[i] + (1.0f - rho) * (g * g);
        theta->data[i] -= (eta / (sqrtf(cache->data[i]) + eps)) * g;
    }
}

/**
 * r_optimization_adam() - Perform Adam parameter update.
 * @theta: Parameter matrix updated in place.
 * @grad: Gradient matrix.
 * @m: First-moment running average updated in place.
 * @v: Second-moment running average updated in place.
 * @eta: Learning rate.
 * @beta1: Decay rate for the first moment.
 * @beta2: Decay rate for the second moment.
 * @eps: Small constant for numerical stability.
 * @t: Time step used for bias correction.
 *
 * Updates the first and second moments, applies bias correction, and
 * updates @theta using the Adam rule.
 * Return: Nothing.
 */
void r_optimization_adam(RNONNULL RMatrix *theta, const RNONNULL RMatrix *grad, RNONNULL RMatrix *m,
                         RNONNULL RMatrix *v, float eta, float beta1, float beta2, float eps, size_t t)
{
    float beta1_t = 1.0f - powf(beta1, (float)t);
    float beta2_t = 1.0f - powf(beta2, (float)t);

    const size_t count = MatrixSize(theta);
    for (size_t i = 0; i < count; i++)
    {
        float g = grad->data[i];

        m->data[i] = beta1 * m->data[i] + (1.0f - beta1) * g;

        v->data[i] = beta2 * v->data[i] + (1.0f - beta2) * (g * g);

        float m_hat = m->data[i] / beta1_t;
        float v_hat = v->data[i] / beta2_t;

        theta->data[i] -= (eta / (sqrtf(v_hat) + eps)) * m_hat;
    }
}
