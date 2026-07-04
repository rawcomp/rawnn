#include <rc/r_optimization.h>
#include <rc/r_utils.h>

/**
 * r_optimization_sgd() - Perform an SGD parameter update.
 * @theta: Parameter matrix updated in place.
 * @grad: Gradient matrix.
 * @eta: Learning rate.
 *
 * Applies a simple SGD update: theta -= eta * grad.
 * Return: Nothing.
 */
void r_optimization_sgd(RMatrix *theta, const RMatrix *grad, float eta)
{
	size_t count;
	size_t i;

	if (!r_has_same_shape(theta, grad)) {
		fprintf(stderr, "[ERROR]: r_optimization_sgd requires theta "
				"and grad with matching shapes\n");
		return;
	}

	count = R_MATRIX_SIZE(theta);
	for (i = 0; i < count; i++)
		theta->data[i] -= eta * grad->data[i];
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
void r_optimization_sgdm(RMatrix *theta, const RMatrix *grad, RMatrix *velocity,
			 float eta, float beta)
{
	size_t count;
	size_t i;

	if (!r_has_same_shape(theta, grad) ||
	    !r_has_same_shape(theta, velocity)) {
		fprintf(stderr, "[ERROR]: r_optimization_sgdm requires theta, "
				"grad and velocity with matching shapes\n");
		return;
	}

	count = R_MATRIX_SIZE(theta);
	for (i = 0; i < count; i++) {
		velocity->data[i] =
		    beta * velocity->data[i] + (1.0f - beta) * grad->data[i];
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
void r_optimization_rmsprop(RMatrix *theta, const RMatrix *grad, RMatrix *cache,
			    float eta, float rho, float eps)
{
	size_t count;
	size_t i;

	if (!r_has_same_shape(theta, grad) || !r_has_same_shape(theta, cache)) {
		fprintf(stderr, "[ERROR]: r_optimization_rmsprop requires "
				"theta, grad and cache with matching shapes\n");
		return;
	}

	count = R_MATRIX_SIZE(theta);
	for (i = 0; i < count; i++) {
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
void r_optimization_adam(RMatrix *theta, const RMatrix *grad, RMatrix *m,
			 RMatrix *v, float eta, float beta1, float beta2,
			 float eps, size_t t)
{
	float beta1_t;
	float beta2_t;
	size_t count;
	size_t i;

	if (!r_has_same_shape(theta, grad) || !r_has_same_shape(theta, m) ||
	    !r_has_same_shape(theta, v)) {
		fprintf(stderr, "[ERROR]: r_optimization_adam requires theta, "
				"grad, m and v with matching shapes\n");
		return;
	}

	if (t == 0)
		t = 1;

	beta1_t = 1.0f - powf(beta1, (float)t);
	beta2_t = 1.0f - powf(beta2, (float)t);

	count = R_MATRIX_SIZE(theta);
	for (i = 0; i < count; i++) {
		float g = grad->data[i];
		float m_hat;
		float v_hat;

		m->data[i] = beta1 * m->data[i] + (1.0f - beta1) * g;
		v->data[i] = beta2 * v->data[i] + (1.0f - beta2) * (g * g);

		m_hat = m->data[i] / beta1_t;
		v_hat = v->data[i] / beta2_t;

		theta->data[i] -= (eta / (sqrtf(v_hat) + eps)) * m_hat;
	}
}
