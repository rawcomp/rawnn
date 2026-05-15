#include <rc/r_optimization.h>

void r_optimization_sgd(RNONNULL RMatrix *theta, const RNONNULL RMatrix *grad, float eta)
{
    for (size_t i = 0; i < MatrixSize(theta); i++)
    {
        theta->data[i] -= eta * grad->data[i];
    }
}

void r_optimization_sgdm(RNONNULL RMatrix *theta, const RNONNULL RMatrix *grad, RNONNULL RMatrix *velocity, float eta,
                         float beta)
{
    for (size_t i = 0; i < MatrixSize(theta); i++)
    {
        velocity->data[i] = beta * velocity->data[i] + (1.0f - beta) * grad->data[i];
        theta->data[i] -= eta * velocity->data[i];
    }
}

void r_optimization_rmsprop(RNONNULL RMatrix *theta, const RNONNULL RMatrix *grad, RNONNULL RMatrix *cache, float eta,
                            float rho, float eps)
{
    for (size_t i = 0; i < MatrixSize(theta); i++)
    {
        float g = grad->data[i];
        cache->data[i] = rho * cache->data[i] + (1.0f - rho) * (g * g);
        theta->data[i] -= (eta / (sqrtf(cache->data[i]) + eps)) * g;
    }
}

void r_optimization_adam(RNONNULL RMatrix *theta, const RNONNULL RMatrix *grad, RNONNULL RMatrix *m,
                         RNONNULL RMatrix *v, float eta, float beta1, float beta2, float eps, size_t t)
{
    float beta1_t = 1.0f - powf(beta1, (float)t);
    float beta2_t = 1.0f - powf(beta2, (float)t);

    for (size_t i = 0; i < MatrixSize(theta); i++)
    {
        float g = grad->data[i];

        m->data[i] = beta1 * m->data[i] + (1.0f - beta1) * g;

        v->data[i] = beta2 * v->data[i] + (1.0f - beta2) * (g * g);

        float m_hat = m->data[i] / beta1_t;
        float v_hat = v->data[i] / beta2_t;

        theta->data[i] -= (eta / (sqrtf(v_hat) + eps)) * m_hat;
    }
}
