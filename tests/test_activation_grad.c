#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <rc/rc.h>

#define DEFAULT_ABS_TOL 1e-5f
#define DEFAULT_REL_TOL 1e-5f
#define NUM_GRAD_ABS_TOL 1e-3f
#define NUM_GRAD_REL_TOL 1e-2f

static RMatrix *make_matrix(size_t rows, size_t cols, const float *values)
{
	RMatrix *matrix = r_create_matrix(rows, cols);
	const size_t count = rows * cols;
	if (count > 0 && values != NULL) {
		memcpy(matrix->data, values, sizeof(float) * count);
	}
	return matrix;
}

static RMatrix *clone_matrix(const RMatrix *src)
{
	return make_matrix(src->rows, src->cols, src->data);
}

static float matrix_dot(const RMatrix *lhs, const RMatrix *rhs)
{
	assert(lhs->rows == rhs->rows && lhs->cols == rhs->cols);
	const size_t count = lhs->rows * lhs->cols;
	float sum = 0.0f;
	for (size_t i = 0; i < count; i++) {
		sum += lhs->data[i] * rhs->data[i];
	}
	return sum;
}

static void assert_float_close(float actual, float expected, float abs_tol,
			       float rel_tol, const char *msg)
{
	const float diff = fabsf(actual - expected);
	const float scale = fmaxf(fabsf(expected), 1.0f);
	const float tol = fmaxf(abs_tol, rel_tol * scale);

	if (diff > tol) {
		fprintf(stderr,
			"ASSERT_FLOAT_CLOSE failed: %s (actual=%.8f "
			"expected=%.8f diff=%.8f tol=%.8f)\n",
			msg, actual, expected, diff, tol);
	}

	assert(diff <= tol);
}

static void assert_true(int condition, const char *msg)
{
	if (!condition) {
		fprintf(stderr, "ASSERT_TRUE failed: %s\n", msg);
	}
	assert(condition);
}

static void assert_matrix_close(const RMatrix *matrix, const float *expected,
				size_t count, float abs_tol, float rel_tol,
				const char *label)
{
	for (size_t i = 0; i < count; i++) {
		char msg[96];
		snprintf(msg, sizeof(msg), "%s idx=%zu", label, i);
		assert_float_close(matrix->data[i], expected[i], abs_tol,
				   rel_tol, msg);
	}
}

static float numeric_gelu_grad(const RMatrix *input, const RMatrix *upstream,
			       size_t idx)
{
	const float delta = 1e-4f;

	RMatrix *plus = clone_matrix(input);
	RMatrix *minus = clone_matrix(input);

	plus->data[idx] += delta;
	minus->data[idx] -= delta;

	RMatrix *out_plus = r_activation_gelu(plus);
	RMatrix *out_minus = r_activation_gelu(minus);

	const float loss_plus = matrix_dot(out_plus, upstream);
	const float loss_minus = matrix_dot(out_minus, upstream);

	r_free_matrix(plus);
	r_free_matrix(minus);
	r_free_matrix(out_plus);
	r_free_matrix(out_minus);

	return (loss_plus - loss_minus) / (2.0f * delta);
}

static float numeric_swish_grad(const RMatrix *input, const RMatrix *upstream,
				size_t idx)
{
	const float delta = 1e-4f;

	RMatrix *plus = clone_matrix(input);
	RMatrix *minus = clone_matrix(input);

	plus->data[idx] += delta;
	minus->data[idx] -= delta;

	RMatrix *out_plus = r_activation_swish(plus);
	RMatrix *out_minus = r_activation_swish(minus);

	const float loss_plus = matrix_dot(out_plus, upstream);
	const float loss_minus = matrix_dot(out_minus, upstream);

	r_free_matrix(plus);
	r_free_matrix(minus);
	r_free_matrix(out_plus);
	r_free_matrix(out_minus);

	return (loss_plus - loss_minus) / (2.0f * delta);
}

static float numeric_softmax_grad(const RMatrix *input, const RMatrix *upstream,
				  size_t idx)
{
	const float delta = 1e-4f;

	RMatrix *plus = clone_matrix(input);
	RMatrix *minus = clone_matrix(input);

	plus->data[idx] += delta;
	minus->data[idx] -= delta;

	RMatrix *out_plus = r_activation_softmax(plus);
	RMatrix *out_minus = r_activation_softmax(minus);

	const float loss_plus = matrix_dot(out_plus, upstream);
	const float loss_minus = matrix_dot(out_minus, upstream);

	r_free_matrix(plus);
	r_free_matrix(minus);
	r_free_matrix(out_plus);
	r_free_matrix(out_minus);

	return (loss_plus - loss_minus) / (2.0f * delta);
}

static void test_relu_grad_basic_values(void)
{
	const float input_vals[] = {-1.0f, 0.0f, 2.0f, 3.5f};
	const float upstream_vals[] = {1.0f, -2.0f, 0.5f, 2.0f};
	const float expected_vals[] = {0.0f, 0.0f, 0.5f, 2.0f};

	RMatrix *input = make_matrix(2, 2, input_vals);
	RMatrix *upstream = make_matrix(2, 2, upstream_vals);
	RMatrix *grad = r_create_matrix(2, 2);

	r_activation_relu_grad(input, upstream, grad);

	assert_matrix_close(grad, expected_vals, 4, DEFAULT_ABS_TOL,
			    DEFAULT_REL_TOL, "relu_grad_basic");

	r_free_matrix(input);
	r_free_matrix(upstream);
	r_free_matrix(grad);
}

static void test_relu_grad_nan_input_zeroed(void)
{
	const float input_vals[] = {NAN, 1.0f};
	const float upstream_vals[] = {3.0f, 4.0f};
	const float expected_vals[] = {0.0f, 4.0f};

	RMatrix *input = make_matrix(1, 2, input_vals);
	RMatrix *upstream = make_matrix(1, 2, upstream_vals);
	RMatrix *grad = r_create_matrix(1, 2);

	r_activation_relu_grad(input, upstream, grad);

	assert_true(!isnan(grad->data[0]), "relu_grad_nan output is finite");
	assert_matrix_close(grad, expected_vals, 2, DEFAULT_ABS_TOL,
			    DEFAULT_REL_TOL, "relu_grad_nan");

	r_free_matrix(input);
	r_free_matrix(upstream);
	r_free_matrix(grad);
}

static void test_leaky_relu_grad_basic_values(void)
{
	const float input_vals[] = {-1.0f, 0.0f, 2.0f};
	const float upstream_vals[] = {1.0f, -2.0f, 3.0f};
	const float expected_vals[] = {0.1f, -0.2f, 3.0f};
	const float alpha = 0.1f;

	RMatrix *input = make_matrix(1, 3, input_vals);
	RMatrix *upstream = make_matrix(1, 3, upstream_vals);
	RMatrix *grad = r_create_matrix(1, 3);

	r_activation_leaky_relu_grad(input, upstream, alpha, grad);

	assert_matrix_close(grad, expected_vals, 3, DEFAULT_ABS_TOL,
			    DEFAULT_REL_TOL, "leaky_relu_grad_basic");

	r_free_matrix(input);
	r_free_matrix(upstream);
	r_free_matrix(grad);
}

static void test_leaky_relu_grad_alpha_zero_is_relu(void)
{
	const float input_vals[] = {-1.0f, 1.0f};
	const float upstream_vals[] = {5.0f, -3.0f};
	const float expected_vals[] = {0.0f, -3.0f};
	const float alpha = 0.0f;

	RMatrix *input = make_matrix(1, 2, input_vals);
	RMatrix *upstream = make_matrix(1, 2, upstream_vals);
	RMatrix *grad = r_create_matrix(1, 2);

	r_activation_leaky_relu_grad(input, upstream, alpha, grad);

	assert_matrix_close(grad, expected_vals, 2, DEFAULT_ABS_TOL,
			    DEFAULT_REL_TOL, "leaky_relu_grad_alpha_zero");

	r_free_matrix(input);
	r_free_matrix(upstream);
	r_free_matrix(grad);
}

static void test_gelu_grad_numeric(void)
{
	const float input_vals[] = {-1.0f, -0.5f, 0.5f, 1.0f};
	const float upstream_vals[] = {0.7f, -1.2f, 0.3f, 2.0f};

	RMatrix *input = make_matrix(2, 2, input_vals);
	RMatrix *upstream = make_matrix(2, 2, upstream_vals);
	RMatrix *grad = r_create_matrix(2, 2);

	r_activation_gelu_grad(input, upstream, grad);

	for (size_t i = 0; i < 4; i++) {
		const float expected = numeric_gelu_grad(input, upstream, i);
		char msg[96];
		snprintf(msg, sizeof(msg), "gelu_grad_numeric idx=%zu", i);
		assert_float_close(grad->data[i], expected, NUM_GRAD_ABS_TOL,
				   NUM_GRAD_REL_TOL, msg);
	}

	r_free_matrix(input);
	r_free_matrix(upstream);
	r_free_matrix(grad);
}

static void test_gelu_grad_zero_input(void)
{
	const float input_vals[] = {0.0f};
	const float upstream_vals[] = {2.0f};
	const float expected_vals[] = {1.0f};

	RMatrix *input = make_matrix(1, 1, input_vals);
	RMatrix *upstream = make_matrix(1, 1, upstream_vals);
	RMatrix *grad = r_create_matrix(1, 1);

	r_activation_gelu_grad(input, upstream, grad);

	assert_matrix_close(grad, expected_vals, 1, DEFAULT_ABS_TOL,
			    DEFAULT_REL_TOL, "gelu_grad_zero");

	r_free_matrix(input);
	r_free_matrix(upstream);
	r_free_matrix(grad);
}

static void test_softmax_grad_numeric(void)
{
	const float input_vals[] = {1.0f, 0.5f, -1.0f, 0.0f, 2.0f, -2.0f};
	const float upstream_vals[] = {0.2f, -0.1f, 0.3f, 1.0f, -0.5f, 0.0f};

	RMatrix *input = make_matrix(2, 3, input_vals);
	RMatrix *upstream = make_matrix(2, 3, upstream_vals);
	RMatrix *output = r_activation_softmax(input);
	RMatrix *grad = r_create_matrix(2, 3);

	r_activation_softmax_grad(output, upstream, grad);

	for (size_t i = 0; i < 6; i++) {
		const float expected = numeric_softmax_grad(input, upstream, i);
		char msg[96];
		snprintf(msg, sizeof(msg), "softmax_grad_numeric idx=%zu", i);
		assert_float_close(grad->data[i], expected, NUM_GRAD_ABS_TOL,
				   NUM_GRAD_REL_TOL, msg);
	}

	for (size_t row = 0; row < grad->rows; row++) {
		float sum = 0.0f;
		for (size_t col = 0; col < grad->cols; col++) {
			sum += grad->data[R_MATRIX_IDX(row, col, grad->cols)];
		}
		char msg[96];
		snprintf(msg, sizeof(msg), "softmax_grad_row_sum row=%zu", row);
		assert_float_close(sum, 0.0f, DEFAULT_ABS_TOL, DEFAULT_REL_TOL,
				   msg);
	}

	r_free_matrix(input);
	r_free_matrix(upstream);
	r_free_matrix(output);
	r_free_matrix(grad);
}

static void test_swish_grad_numeric(void)
{
	const float input_vals[] = {-2.0f, -0.5f, 0.0f, 1.5f};
	const float upstream_vals[] = {1.0f, -0.5f, 0.25f, 2.0f};

	RMatrix *input = make_matrix(1, 4, input_vals);
	RMatrix *upstream = make_matrix(1, 4, upstream_vals);
	RMatrix *grad = r_create_matrix(1, 4);

	r_activation_swish_grad(input, upstream, grad);

	for (size_t i = 0; i < 4; i++) {
		const float expected = numeric_swish_grad(input, upstream, i);
		char msg[96];
		snprintf(msg, sizeof(msg), "swish_grad_numeric idx=%zu", i);
		assert_float_close(grad->data[i], expected, NUM_GRAD_ABS_TOL,
				   NUM_GRAD_REL_TOL, msg);
	}

	r_free_matrix(input);
	r_free_matrix(upstream);
	r_free_matrix(grad);
}

static void test_swish_grad_nan_propagation(void)
{
	const float input_vals[] = {NAN};
	const float upstream_vals[] = {1.0f};

	RMatrix *input = make_matrix(1, 1, input_vals);
	RMatrix *upstream = make_matrix(1, 1, upstream_vals);
	RMatrix *grad = r_create_matrix(1, 1);

	r_activation_swish_grad(input, upstream, grad);

	assert_true(isnan(grad->data[0]), "swish_grad_nan propagates");

	r_free_matrix(input);
	r_free_matrix(upstream);
	r_free_matrix(grad);
}

static void test_zero_sized_matrices_no_crash(void)
{
	RMatrix *input_rows0 = r_create_matrix(0, 3);
	RMatrix *upstream_rows0 = r_create_matrix(0, 3);
	RMatrix *grad_rows0 = r_create_matrix(0, 3);

	r_activation_relu_grad(input_rows0, upstream_rows0, grad_rows0);
	r_activation_leaky_relu_grad(input_rows0, upstream_rows0, 0.1f,
				     grad_rows0);
	r_activation_gelu_grad(input_rows0, upstream_rows0, grad_rows0);
	r_activation_softmax_grad(input_rows0, upstream_rows0, grad_rows0);
	r_activation_swish_grad(input_rows0, upstream_rows0, grad_rows0);

	assert_true(grad_rows0->rows == 0 && grad_rows0->cols == 3,
		    "zero_rows preserves shape");

	r_free_matrix(input_rows0);
	r_free_matrix(upstream_rows0);
	r_free_matrix(grad_rows0);

	RMatrix *input_cols0 = r_create_matrix(2, 0);
	RMatrix *upstream_cols0 = r_create_matrix(2, 0);
	RMatrix *grad_cols0 = r_create_matrix(2, 0);

	r_activation_relu_grad(input_cols0, upstream_cols0, grad_cols0);
	r_activation_leaky_relu_grad(input_cols0, upstream_cols0, 0.1f,
				     grad_cols0);
	r_activation_gelu_grad(input_cols0, upstream_cols0, grad_cols0);
	r_activation_softmax_grad(input_cols0, upstream_cols0, grad_cols0);
	r_activation_swish_grad(input_cols0, upstream_cols0, grad_cols0);

	assert_true(grad_cols0->rows == 2 && grad_cols0->cols == 0,
		    "zero_cols preserves shape");

	r_free_matrix(input_cols0);
	r_free_matrix(upstream_cols0);
	r_free_matrix(grad_cols0);
}

int main(void)
{
	test_relu_grad_basic_values();
	test_relu_grad_nan_input_zeroed();
	test_leaky_relu_grad_basic_values();
	test_leaky_relu_grad_alpha_zero_is_relu();
	test_gelu_grad_numeric();
	test_gelu_grad_zero_input();
	test_softmax_grad_numeric();
	test_swish_grad_numeric();
	test_swish_grad_nan_propagation();
	test_zero_sized_matrices_no_crash();

	printf("All activation gradient tests passed.\n");
	return 0;
}