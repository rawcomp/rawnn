#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <rc/rc.h>

#define DEFAULT_ABS_TOL 1e-5f
#define DEFAULT_REL_TOL 1e-5f

static RMatrix *make_matrix(size_t rows, size_t cols, const float *values)
{
	RMatrix *matrix = r_create_matrix(rows, cols);
	const size_t count = rows * cols;
	if (count > 0 && values != NULL) {
		memcpy(matrix->data, values, sizeof(float) * count);
	}
	return matrix;
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

static float row_sum(const RMatrix *matrix, size_t row)
{
	float sum = 0.0f;
	for (size_t j = 0; j < matrix->cols; j++) {
		sum += matrix->data[R_MATRIX_IDX(row, j, matrix->cols)];
	}
	return sum;
}

static void test_relu_basic_values(void)
{
	const float input_vals[] = {-1.0f, 0.0f, 2.0f, -3.5f, 4.2f, -0.001f};
	const float expected_vals[] = {0.0f, 0.0f, 2.0f, 0.0f, 4.2f, 0.0f};

	RMatrix *input = make_matrix(2, 3, input_vals);
	RMatrix *output = r_activation_relu(input);

	assert_true(output->rows == 2 && output->cols == 3,
		    "relu_basic preserves shape");
	assert_matrix_close(output, expected_vals, 6, DEFAULT_ABS_TOL,
			    DEFAULT_REL_TOL, "relu_basic");

	r_free_matrix(input);
	r_free_matrix(output);
}

static void test_relu_preserves_input_and_allocates_new(void)
{
	const float input_vals[] = {1.0f, -2.0f, 3.0f, -4.0f};

	RMatrix *input = make_matrix(1, 4, input_vals);
	RMatrix *output = r_activation_relu(input);

	assert_true(output != input, "relu returns new matrix");
	assert_true(output->data != input->data,
		    "relu does not alias input data");

	for (size_t i = 0; i < 4; i++) {
		char msg[96];
		snprintf(msg, sizeof(msg), "relu_input_unchanged idx=%zu", i);
		assert_float_close(input->data[i], input_vals[i],
				   DEFAULT_ABS_TOL, DEFAULT_REL_TOL, msg);
	}

	r_free_matrix(input);
	r_free_matrix(output);
}

static void test_relu_zero_sizes(void)
{
	RMatrix *input_rows0 = r_create_matrix(0, 3);
	RMatrix *output_rows0 = r_activation_relu(input_rows0);
	assert_true(output_rows0->rows == 0 && output_rows0->cols == 3,
		    "relu_zero_rows preserves shape");
	r_free_matrix(input_rows0);
	r_free_matrix(output_rows0);

	RMatrix *input_cols0 = r_create_matrix(3, 0);
	RMatrix *output_cols0 = r_activation_relu(input_cols0);
	assert_true(output_cols0->rows == 3 && output_cols0->cols == 0,
		    "relu_zero_cols preserves shape");
	r_free_matrix(input_cols0);
	r_free_matrix(output_cols0);
}

static void test_relu_nan_converts_to_zero(void)
{
	const float input_vals[] = {NAN, 1.0f, -1.0f};
	const float expected_vals[] = {0.0f, 1.0f, 0.0f};

	RMatrix *input = make_matrix(1, 3, input_vals);
	RMatrix *output = r_activation_relu(input);

	assert_true(!isnan(output->data[0]), "relu_nan output is finite");
	assert_matrix_close(output, expected_vals, 3, DEFAULT_ABS_TOL,
			    DEFAULT_REL_TOL, "relu_nan");

	r_free_matrix(input);
	r_free_matrix(output);
}

static void test_gelu_known_values(void)
{
	const float input_vals[] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
	const float expected_vals[] = {0.0f, 0.8413447f, -0.1586553f,
				       1.9544997f, -0.045500264f};

	RMatrix *input = make_matrix(1, 5, input_vals);
	RMatrix *output = r_activation_gelu(input);

	assert_matrix_close(output, expected_vals, 5, DEFAULT_ABS_TOL,
			    DEFAULT_REL_TOL, "gelu_known");

	r_free_matrix(input);
	r_free_matrix(output);
}

static void test_gelu_preserves_input_and_shape(void)
{
	const float input_vals[] = {0.5f, -0.5f, 1.5f, -1.5f};

	RMatrix *input = make_matrix(2, 2, input_vals);
	RMatrix *output = r_activation_gelu(input);

	assert_true(output->rows == 2 && output->cols == 2,
		    "gelu preserves shape");
	assert_true(output != input, "gelu returns new matrix");
	assert_true(output->data != input->data,
		    "gelu does not alias input data");

	for (size_t i = 0; i < 4; i++) {
		char msg[96];
		snprintf(msg, sizeof(msg), "gelu_input_unchanged idx=%zu", i);
		assert_float_close(input->data[i], input_vals[i],
				   DEFAULT_ABS_TOL, DEFAULT_REL_TOL, msg);
	}

	r_free_matrix(input);
	r_free_matrix(output);
}

static void test_gelu_zero_sizes(void)
{
	RMatrix *input_rows0 = r_create_matrix(0, 4);
	RMatrix *output_rows0 = r_activation_gelu(input_rows0);
	assert_true(output_rows0->rows == 0 && output_rows0->cols == 4,
		    "gelu_zero_rows preserves shape");
	r_free_matrix(input_rows0);
	r_free_matrix(output_rows0);

	RMatrix *input_cols0 = r_create_matrix(4, 0);
	RMatrix *output_cols0 = r_activation_gelu(input_cols0);
	assert_true(output_cols0->rows == 4 && output_cols0->cols == 0,
		    "gelu_zero_cols preserves shape");
	r_free_matrix(input_cols0);
	r_free_matrix(output_cols0);
}

static void test_gelu_large_magnitude_limits(void)
{
	const float input_vals[] = {6.0f, -6.0f};

	RMatrix *input = make_matrix(1, 2, input_vals);
	RMatrix *output = r_activation_gelu(input);

	assert_float_close(output->data[0], 6.0f, 1e-4f, DEFAULT_REL_TOL,
			   "gelu_large_positive");
	assert_float_close(output->data[1], 0.0f, 1e-4f, DEFAULT_REL_TOL,
			   "gelu_large_negative");

	r_free_matrix(input);
	r_free_matrix(output);
}

static void test_gelu_nan_propagation(void)
{
	const float input_vals[] = {NAN};

	RMatrix *input = make_matrix(1, 1, input_vals);
	RMatrix *output = r_activation_gelu(input);

	assert_true(isnan(output->data[0]), "gelu_nan propagates");

	r_free_matrix(input);
	r_free_matrix(output);
}

static void test_softmax_basic_values(void)
{
	const float input_vals[] = {1.0f, 2.0f, 3.0f};
	const float expected_vals[] = {0.09003057f, 0.24472847f, 0.66524096f};

	RMatrix *input = make_matrix(1, 3, input_vals);
	RMatrix *output = r_activation_softmax(input);

	assert_matrix_close(output, expected_vals, 3, DEFAULT_ABS_TOL,
			    DEFAULT_REL_TOL, "softmax_basic");

	r_free_matrix(input);
	r_free_matrix(output);
}

static void test_softmax_row_sums_and_bounds(void)
{
	const float input_vals[] = {1.0f, 2.0f, -3.0f, 0.5f,
				    0.1f, 0.2f, 0.3f,  0.4f};

	RMatrix *input = make_matrix(2, 4, input_vals);
	RMatrix *output = r_activation_softmax(input);

	for (size_t i = 0; i < output->rows; i++) {
		char msg[96];
		snprintf(msg, sizeof(msg), "softmax_row_sum row=%zu", i);
		assert_float_close(row_sum(output, i), 1.0f, DEFAULT_ABS_TOL,
				   DEFAULT_REL_TOL, msg);

		for (size_t j = 0; j < output->cols; j++) {
			const float value =
			    output->data[R_MATRIX_IDX(i, j, output->cols)];
			snprintf(msg, sizeof(msg),
				 "softmax_bounds row=%zu col=%zu", i, j);
			assert_true(isfinite(value) && value >= 0.0f &&
					value <= 1.0f,
				    msg);
		}
	}

	r_free_matrix(input);
	r_free_matrix(output);
}

static void test_softmax_shift_invariance(void)
{
	const float input_vals[] = {1.0f, -1.0f, 0.5f, 2.5f};
	const float shifted_vals[] = {101.0f, 99.0f, 100.5f, 102.5f};

	RMatrix *input = make_matrix(1, 4, input_vals);
	RMatrix *shifted = make_matrix(1, 4, shifted_vals);
	RMatrix *output = r_activation_softmax(input);
	RMatrix *output_shifted = r_activation_softmax(shifted);

	for (size_t j = 0; j < 4; j++) {
		char msg[96];
		snprintf(msg, sizeof(msg), "softmax_shift_invariance col=%zu",
			 j);
		assert_float_close(output_shifted->data[j], output->data[j],
				   DEFAULT_ABS_TOL, DEFAULT_REL_TOL, msg);
	}

	r_free_matrix(input);
	r_free_matrix(shifted);
	r_free_matrix(output);
	r_free_matrix(output_shifted);
}

static void test_softmax_uniform_for_equal_inputs(void)
{
	const float input_vals[] = {2.0f, 2.0f, 2.0f, 2.0f};
	const float expected = 0.25f;

	RMatrix *input = make_matrix(1, 4, input_vals);
	RMatrix *output = r_activation_softmax(input);

	for (size_t j = 0; j < 4; j++) {
		char msg[96];
		snprintf(msg, sizeof(msg), "softmax_uniform col=%zu", j);
		assert_float_close(output->data[j], expected, DEFAULT_ABS_TOL,
				   DEFAULT_REL_TOL, msg);
	}

	r_free_matrix(input);
	r_free_matrix(output);
}

static void test_softmax_single_element(void)
{
	const float input_vals[] = {5.0f, -2.0f, 0.0f};

	RMatrix *input = make_matrix(3, 1, input_vals);
	RMatrix *output = r_activation_softmax(input);

	for (size_t i = 0; i < output->rows; i++) {
		char msg[96];
		snprintf(msg, sizeof(msg), "softmax_single_element row=%zu", i);
		assert_float_close(
		    output->data[R_MATRIX_IDX(i, 0, output->cols)], 1.0f,
		    DEFAULT_ABS_TOL, DEFAULT_REL_TOL, msg);
	}

	r_free_matrix(input);
	r_free_matrix(output);
}

static void test_softmax_large_values_stability(void)
{
	const float input_vals[] = {1000.0f, 1001.0f, 1002.0f};
	const float expected_vals[] = {0.09003057f, 0.24472847f, 0.66524096f};

	RMatrix *input = make_matrix(1, 3, input_vals);
	RMatrix *output = r_activation_softmax(input);

	for (size_t j = 0; j < 3; j++) {
		char msg[96];
		snprintf(msg, sizeof(msg), "softmax_large_values col=%zu", j);
		assert_true(isfinite(output->data[j]), msg);
		assert_float_close(output->data[j], expected_vals[j],
				   DEFAULT_ABS_TOL, DEFAULT_REL_TOL, msg);
	}

	assert_true(output->data[2] > output->data[1] &&
			output->data[1] > output->data[0],
		    "softmax_large_values preserves ordering");
	assert_float_close(row_sum(output, 0), 1.0f, DEFAULT_ABS_TOL,
			   DEFAULT_REL_TOL, "softmax_large_values sum");

	r_free_matrix(input);
	r_free_matrix(output);
}

static void test_softmax_zero_rows(void)
{
	RMatrix *input = r_create_matrix(0, 3);
	RMatrix *output = r_activation_softmax(input);

	assert_true(output->rows == 0, "softmax_zero_rows preserves row count");
	assert_true(output->cols == 3, "softmax_zero_rows preserves col count");

	r_free_matrix(input);
	r_free_matrix(output);
}

static void test_softmax_input_unchanged(void)
{
	const float input_vals[] = {0.1f, 0.2f, 0.3f, 0.4f};

	RMatrix *input = make_matrix(1, 4, input_vals);
	RMatrix *output = r_activation_softmax(input);

	for (size_t i = 0; i < 4; i++) {
		char msg[96];
		snprintf(msg, sizeof(msg), "softmax_input_unchanged idx=%zu",
			 i);
		assert_float_close(input->data[i], input_vals[i],
				   DEFAULT_ABS_TOL, DEFAULT_REL_TOL, msg);
	}

	r_free_matrix(input);
	r_free_matrix(output);
}

static void test_softmax_nan_propagation(void)
{
	const float input_vals[] = {NAN, 1.0f, 2.0f};

	RMatrix *input = make_matrix(1, 3, input_vals);
	RMatrix *output = r_activation_softmax(input);

	for (size_t i = 0; i < 3; i++) {
		char msg[96];
		snprintf(msg, sizeof(msg), "softmax_nan idx=%zu", i);
		assert_true(isnan(output->data[i]), msg);
	}

	r_free_matrix(input);
	r_free_matrix(output);
}

int main(void)
{
	test_relu_basic_values();
	test_relu_preserves_input_and_allocates_new();
	test_relu_zero_sizes();
	test_relu_nan_converts_to_zero();

	test_gelu_known_values();
	test_gelu_preserves_input_and_shape();
	test_gelu_zero_sizes();
	test_gelu_large_magnitude_limits();
	test_gelu_nan_propagation();

	test_softmax_basic_values();
	test_softmax_row_sums_and_bounds();
	test_softmax_shift_invariance();
	test_softmax_uniform_for_equal_inputs();
	test_softmax_single_element();
	test_softmax_large_values_stability();
	test_softmax_zero_rows();
	test_softmax_input_unchanged();
	test_softmax_nan_propagation();

	printf("All activation tests passed.\n");
	return 0;
}