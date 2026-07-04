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

static void set_layer_weights(RLayerDense *layer, const float *values)
{
	const size_t count = layer->weights->rows * layer->weights->cols;
	if (count > 0 && values != NULL) {
		memcpy(layer->weights->data, values, sizeof(float) * count);
	}
}

static void set_layer_biases(RLayerDense *layer, const float *values)
{
	const size_t count = layer->biases->size;
	if (count > 0 && values != NULL) {
		memcpy(layer->biases->data, values, sizeof(float) * count);
	}
}

static void compute_dense_expected(const float *inputs, size_t batch,
				   size_t n_inputs, const float *weights,
				   size_t n_neurons, const float *biases,
				   float *out)
{
	for (size_t i = 0; i < batch; i++) {
		for (size_t j = 0; j < n_neurons; j++) {
			float sum = 0.0f;
			if (n_inputs > 0 && weights != NULL) {
				for (size_t k = 0; k < n_inputs; k++) {
					const float x =
					    inputs[(i * n_inputs) + k];
					const float w =
					    weights[(j * n_inputs) + k];
					sum += x * w;
				}
			}
			out[(i * n_neurons) + j] = sum + biases[j];
		}
	}
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

static void assert_matrix_close(const RMatrix *actual, const float *expected,
				size_t rows, size_t cols, float abs_tol,
				float rel_tol, const char *label)
{
	assert(actual != NULL && "matrix should not be NULL");
	assert(actual->rows == rows && actual->cols == cols &&
	       "matrix dimensions mismatch");

	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			const size_t idx = R_MATRIX_IDX(i, j, cols);
			char msg[128];
			snprintf(msg, sizeof(msg), "%s r=%zu c=%zu", label, i,
				 j);
			assert_float_close(actual->data[idx], expected[idx],
					   abs_tol, rel_tol, msg);
		}
	}
}

static void test_create_layer_dimensions(void)
{
	RLayerDense *layer = r_create_layer(3, 2);

	assert(layer != NULL && "layer should be allocated");
	assert(layer->weights != NULL && "weights should be allocated");
	assert(layer->biases != NULL && "biases should be allocated");
	assert(layer->weights->rows == 2 && layer->weights->cols == 3 &&
	       "weights dimensions");
	assert(layer->biases->size == 2 && "biases dimensions");

	r_free_layer(layer);
}

static void test_layer_forward_basic(void)
{
	const float input_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	const float weight_vals[] = {0.2f, 0.4f, -0.5f, 1.0f, -1.0f, 0.5f};
	const float bias_vals[] = {0.1f, -0.2f};

	RMatrix *input = make_matrix(2, 3, input_vals);
	RLayerDense *layer = r_create_layer(3, 2);
	set_layer_weights(layer, weight_vals);
	set_layer_biases(layer, bias_vals);

	float expected[4];
	compute_dense_expected(input_vals, 2, 3, weight_vals, 2, bias_vals,
			       expected);

	RMatrix *output = r_layer_forward(layer, input);

	assert_matrix_close(output, expected, 2, 2, DEFAULT_ABS_TOL,
			    DEFAULT_REL_TOL, "layer_forward_basic");

	r_free_matrix(input);
	r_free_matrix(output);
	r_free_layer(layer);
}

static void test_layer_forward_bias_only_zero_inputs(void)
{
	const float bias_vals[] = {1.5f, -2.0f, 0.0f};

	RMatrix *input = r_create_matrix(2, 0);
	RLayerDense *layer = r_create_layer(0, 3);
	set_layer_biases(layer, bias_vals);

	float expected[6];
	compute_dense_expected(NULL, 2, 0, NULL, 3, bias_vals, expected);

	RMatrix *output = r_layer_forward(layer, input);

	assert_matrix_close(output, expected, 2, 3, DEFAULT_ABS_TOL,
			    DEFAULT_REL_TOL,
			    "layer_forward_zero_inputs_bias_only");

	r_free_matrix(input);
	r_free_matrix(output);
	r_free_layer(layer);
}

static void test_layer_forward_zero_batch_size(void)
{
	const float weight_vals[] = {0.1f, 0.2f, 0.3f, -0.4f, 0.5f, -0.6f};
	const float bias_vals[] = {0.7f, -0.8f};

	RMatrix *input = r_create_matrix(0, 3);
	RLayerDense *layer = r_create_layer(3, 2);
	set_layer_weights(layer, weight_vals);
	set_layer_biases(layer, bias_vals);

	RMatrix *output = r_layer_forward(layer, input);

	assert(output != NULL && "output should be allocated");
	assert(output->rows == 0 && output->cols == 2 &&
	       "zero batch should keep cols");

	r_free_matrix(input);
	r_free_matrix(output);
	r_free_layer(layer);
}

static void test_layer_forward_nan_propagation(void)
{
	const float input_vals[] = {NAN, 1.0f};
	const float weight_vals[] = {0.5f, -1.0f};
	const float bias_vals[] = {0.0f};

	RMatrix *input = make_matrix(1, 2, input_vals);
	RLayerDense *layer = r_create_layer(2, 1);
	set_layer_weights(layer, weight_vals);
	set_layer_biases(layer, bias_vals);

	RMatrix *output = r_layer_forward(layer, input);

	assert(isnan(output->data[0]) &&
	       "layer_forward should propagate NaN inputs");

	r_free_matrix(input);
	r_free_matrix(output);
	r_free_layer(layer);
}

static void test_multi_layer_forward_matches_manual(void)
{
	const float input_vals[] = {1.0f, -1.0f, 0.5f, 2.0f};

	const float w1[] = {0.1f, 0.2f, -0.3f, 0.4f, 0.5f, -0.6f};
	const float b1[] = {0.01f, 0.02f, -0.03f};

	const float w2[] = {0.7f, -0.8f, 0.9f, -1.0f, 0.5f, 0.3f};
	const float b2[] = {0.1f, -0.2f};

	const float w3[] = {0.25f, -0.75f};
	const float b3[] = {0.05f};

	float expected_l1[6];
	float expected_l2[4];
	float expected_l3[2];

	compute_dense_expected(input_vals, 2, 2, w1, 3, b1, expected_l1);
	compute_dense_expected(expected_l1, 2, 3, w2, 2, b2, expected_l2);
	compute_dense_expected(expected_l2, 2, 2, w3, 1, b3, expected_l3);

	RMatrix *input = make_matrix(2, 2, input_vals);

	RLayerDense *l1 = r_create_layer(2, 3);
	set_layer_weights(l1, w1);
	set_layer_biases(l1, b1);

	RLayerDense *l2 = r_create_layer(3, 2);
	set_layer_weights(l2, w2);
	set_layer_biases(l2, b2);

	RLayerDense *l3 = r_create_layer(2, 1);
	set_layer_weights(l3, w3);
	set_layer_biases(l3, b3);

	RMatrix *out1 = r_layer_forward(l1, input);
	RMatrix *out2 = r_layer_forward(l2, out1);
	RMatrix *out3 = r_layer_forward(l3, out2);

	assert_matrix_close(out1, expected_l1, 2, 3, DEFAULT_ABS_TOL,
			    DEFAULT_REL_TOL, "multi_layer_out1");
	assert_matrix_close(out2, expected_l2, 2, 2, DEFAULT_ABS_TOL,
			    DEFAULT_REL_TOL, "multi_layer_out2");
	assert_matrix_close(out3, expected_l3, 2, 1, DEFAULT_ABS_TOL,
			    DEFAULT_REL_TOL, "multi_layer_out3");

	r_free_matrix(input);
	r_free_matrix(out1);
	r_free_matrix(out2);
	r_free_matrix(out3);
	r_free_layer(l1);
	r_free_layer(l2);
	r_free_layer(l3);
}

int main(void)
{
	test_create_layer_dimensions();
	test_layer_forward_basic();
	test_layer_forward_bias_only_zero_inputs();
	test_layer_forward_zero_batch_size();
	test_layer_forward_nan_propagation();
	test_multi_layer_forward_matches_manual();

	printf("All multi-layer tests passed.\n");
	return 0;
}
