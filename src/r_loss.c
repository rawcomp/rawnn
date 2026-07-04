#include <rc/r_loss.h>
#include <rc/r_utils.h>

static int validate_loss_inputs(const RMatrix *pred, const RMatrix *real,
				const char *name)
{
	if (!r_has_same_shape(pred, real)) {
		fprintf(stderr,
			"[ERROR]: %s requires prediction and target matrices "
			"with matching shapes\n",
			name);
		return 0;
	}
	if (pred->rows == 0 || pred->cols == 0) {
		fprintf(stderr, "[ERROR]: %s does not accept empty matrices\n",
			name);
		return 0;
	}
	return 1;
}

static inline float clamp(float val, float min_val, float max_val)
{
	return fmaxf(min_val, fminf(val, max_val));
}

/**
 * r_cross_entropy() - Compute cross-entropy loss.
 * @pred: Predicted probabilities.
 * @real: One-hot or target distribution matrix.
 *
 * Accumulates -y * log(p) over all elements and normalizes by
 * the number of rows.
 * Return: Mean cross-entropy loss per row.
 */
float r_cross_entropy(const RMatrix *pred, const RMatrix *real)
{
	double total = 0.0;
	size_t count;
	size_t i;

	if (!validate_loss_inputs(pred, real, "r_cross_entropy"))
		return NAN;

	count = R_MATRIX_SIZE(pred);
	for (i = 0; i < count; i++) {
		float prediction =
		    clamp(pred->data[i], EPSILON, 1.0f - EPSILON);
		float correct = real->data[i];

		total += (double)correct * logf(prediction);
	}

	return (float)(-total / real->rows);
}

/**
 * r_bin_cross_entropy() - Compute binary cross-entropy loss.
 * @pred: Predicted probabilities.
 * @real: Target labels in {0, 1}.
 *
 * Computes the element-wise binary cross-entropy and averages over all
 * elements.
 * Return: Mean binary cross-entropy loss.
 */
float r_bin_cross_entropy(const RMatrix *pred, const RMatrix *real)
{
	double total = 0.0;
	size_t count;
	size_t i;

	if (!validate_loss_inputs(pred, real, "r_bin_cross_entropy"))
		return NAN;

	count = R_MATRIX_SIZE(real);
	for (i = 0; i < count; i++) {
		float currentReal = real->data[i];
		float currentPred =
		    clamp(pred->data[i], EPSILON, 1.0f - EPSILON);

		total +=
		    (double)currentReal * logf(currentPred) +
		    (double)(1.0f - currentReal) * logf(1.0f - currentPred);
	}

	return (float)(-total / (real->cols * real->rows));
}

/**
 * r_cat_cross_entropy() - Compute categorical cross-entropy loss.
 * @pred: Predicted class probabilities.
 * @real: One-hot encoded target labels.
 *
 * Accumulates -y * log(p) over all elements and normalizes by
 * the number of rows.
 * Return: Mean categorical cross-entropy loss per row.
 */
float r_cat_cross_entropy(const RMatrix *pred, const RMatrix *real)
{
	/* r_cross_entropy is functionally identical for categorical targets */
	return r_cross_entropy(pred, real);
}

/**
 * r_mse_loss() - Compute mean squared error loss.
 * @pred: Predicted values.
 * @real: Target values.
 *
 * Computes the squared error per element and averages over all elements.
 * Return: Mean squared error.
 */
float r_mse_loss(const RMatrix *pred, const RMatrix *real)
{
	double result = 0.0;
	size_t count;
	size_t i;

	if (!validate_loss_inputs(pred, real, "r_mse_loss"))
		return NAN;

	count = R_MATRIX_SIZE(real);
	for (i = 0; i < count; i++) {
		float error = real->data[i] - pred->data[i];

		result += (double)(error * error);
	}

	return (float)(result / (real->cols * real->rows));
}

/**
 * r_mae_loss() - Compute mean absolute error loss.
 * @pred: Predicted values.
 * @real: Target values.
 *
 * Computes the absolute error per element and averages over all elements.
 * Return: Mean absolute error.
 */
float r_mae_loss(const RMatrix *pred, const RMatrix *real)
{
	double result = 0.0;
	size_t count;
	size_t i;

	if (!validate_loss_inputs(pred, real, "r_mae_loss"))
		return NAN;

	count = R_MATRIX_SIZE(real);
	for (i = 0; i < count; i++) {
		float error = real->data[i] - pred->data[i];

		result += (double)fabsf(error);
	}

	return (float)(result / (real->cols * real->rows));
}

static inline float fast_powf(float base, float gamma)
{
	if (fabsf(gamma - 2.0f) < EPSILON)
		return base * base;
	if (fabsf(gamma - 1.0f) < EPSILON)
		return base;
	if (fabsf(gamma - 3.0f) < EPSILON)
		return base * base * base;
	return expf(gamma * logf(base));
}

/**
 * r_bin_focal_loss() - Compute binary focal loss.
 * @pred: Predicted probabilities.
 * @real: Target labels in {0, 1}.
 * @gamma: Focusing parameter.
 * @alpha: Class weighting parameter.
 *
 * Applies the focal loss modulation to the binary cross-entropy and
 * averages over all elements.
 * Return: Mean binary focal loss.
 */
float r_bin_focal_loss(const RMatrix *pred, const RMatrix *real, float gamma,
		       float alpha)
{
	double total = 0.0;
	size_t count;
	size_t i;

	if (!validate_loss_inputs(pred, real, "r_bin_focal_loss"))
		return NAN;

	count = R_MATRIX_SIZE(real);
	for (i = 0; i < count; i++) {
		float y = real->data[i];
		float p = clamp(pred->data[i], EPSILON, 1.0f - EPSILON);
		float term1 = alpha * y * fast_powf(1.0f - p, gamma) * logf(p);
		float term2 = (1.0f - alpha) * (1.0f - y) *
			      fast_powf(p, gamma) * logf(1.0f - p);

		total += (double)(term1 + term2);
	}

	return (float)(-total / (real->cols * real->rows));
}

/**
 * r_cat_focal_loss() - Compute categorical focal loss.
 * @pred: Predicted class probabilities.
 * @real: One-hot encoded target labels.
 * @gamma: Focusing parameter.
 *
 * Applies the focal modulation to the correct class per row and
 * averages over the number of rows.
 * Return: Mean categorical focal loss per row.
 */
float r_cat_focal_loss(const RMatrix *pred, const RMatrix *real, float gamma)
{
	double total = 0.0;
	size_t count;
	size_t i;

	if (!validate_loss_inputs(pred, real, "r_cat_focal_loss"))
		return NAN;

	count = R_MATRIX_SIZE(real);
	for (i = 0; i < count; i++) {
		float prediction =
		    clamp(pred->data[i], EPSILON, 1.0f - EPSILON);
		float correct = real->data[i];

		if (correct > 0.0f) {
			float mod_factor = fast_powf(1.0f - prediction, gamma);

			total +=
			    (double)(correct * mod_factor * logf(prediction));
		}
	}

	return (float)(-total / real->rows);
}
