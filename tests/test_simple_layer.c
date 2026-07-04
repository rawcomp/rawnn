#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <rc/rc.h>

#define DEFAULT_ABS_TOL 1e-4f
#define DEFAULT_REL_TOL 1e-4f

static RMatrix *make_matrix(size_t rows, size_t cols, const float *values)
{
	RMatrix *matrix = r_create_matrix(rows, cols);
	const size_t count = rows * cols;
	if (count > 0 && values != NULL) {
		memcpy(matrix->data, values, sizeof(float) * count);
	}
	return matrix;
}

static RLayerDense *make_layer(size_t n_inputs, size_t n_neurons,
			       const float *weights, const float *biases)
{
	RLayerDense *layer = r_create_layer(n_inputs, n_neurons);
	const size_t weight_count = n_inputs * n_neurons;
	if (weight_count > 0 && weights != NULL) {
		memcpy(layer->weights->data, weights,
		       sizeof(float) * weight_count);
	}
	if (n_neurons > 0 && biases != NULL) {
		memcpy(layer->biases->data, biases, sizeof(float) * n_neurons);
	}
	return layer;
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
				size_t rows, size_t cols, const char *msg)
{
	assert(actual->rows == rows && actual->cols == cols &&
	       "matrix shape mismatch");

	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			const size_t idx = R_MATRIX_IDX(i, j, actual->cols);
			char buf[128];
			snprintf(buf, sizeof(buf), "%s idx=%zu", msg, idx);
			assert_float_close(actual->data[idx], expected[idx],
					   DEFAULT_ABS_TOL, DEFAULT_REL_TOL,
					   buf);
		}
	}
}

static void test_create_layer_shapes(void)
{
	RLayerDense *layer = r_create_layer(4, 3);

	assert(layer != NULL && "layer allocation failed");
	assert(layer->weights != NULL && layer->biases != NULL &&
	       "layer members allocation failed");
	assert(layer->weights->rows == 3 && layer->weights->cols == 4 &&
	       "weights shape mismatch");
	assert(layer->biases->size == 3 && "bias size mismatch");

	r_free_layer(layer);
}

static void test_layer_forward_single_neuron(void)
{
	const float input_vals[] = {1.0f, 2.0f, 3.0f};
	const float weight_vals[] = {0.2f, 0.8f, -0.5f};
	const float bias_vals[] = {2.0f};

	RMatrix *inputs = make_matrix(1, 3, input_vals);
	RLayerDense *layer = make_layer(3, 1, weight_vals, bias_vals);
	RMatrix *output = r_layer_forward(layer, inputs);

	assert(output->rows == 1 && output->cols == 1 &&
	       "single neuron output shape mismatch");
	assert_float_close(output->data[0], 2.3f, DEFAULT_ABS_TOL,
			   DEFAULT_REL_TOL, "single neuron output");

	r_free_matrix(inputs);
	r_free_matrix(output);
	r_free_layer(layer);
}

static void test_layer_forward_single_input_three_neurons(void)
{
	const float input_vals[] = {1.0f, 2.0f, 3.0f, 2.5f};
	const float weight_vals[] = {0.2f,   0.8f,   -0.5f, 1.0f,
				     0.5f,   -0.91f, 0.26f, -0.5f,
				     -0.26f, -0.27f, 0.17f, 0.87f};
	const float bias_vals[] = {2.0f, 3.0f, 0.5f};

	const float expected[] = {4.8f, 1.21f, 2.385f};

	RMatrix *inputs = make_matrix(1, 4, input_vals);
	RLayerDense *layer = make_layer(4, 3, weight_vals, bias_vals);
	RMatrix *output = r_layer_forward(layer, inputs);

	assert_matrix_close(output, expected, 1, 3,
			    "single input three neurons");

	r_free_matrix(inputs);
	r_free_matrix(output);
	r_free_layer(layer);
}

static void test_layer_forward_batch_matches_expected(void)
{
	const float batch_vals[] = {1.0f,  2.0f, 3.0f, 2.5f, 2.0f, 5.0f,
				    -1.0f, 2.0f, 3.0f, 1.0f, 4.0f, 0.5f};
	const float weight_vals[] = {0.2f,   0.8f,   -0.5f, 1.0f,
				     0.5f,   -0.91f, 0.26f, -0.5f,
				     -0.26f, -0.27f, 0.17f, 0.87f};
	const float bias_vals[] = {2.0f, 3.0f, 0.5f};

	const float expected[] = {4.8f, 1.21f, 2.385f, 8.9f,  -1.81f,
				  0.2f, 1.9f,  4.38f,  0.565f};

	RMatrix *inputs = make_matrix(3, 4, batch_vals);
	RLayerDense *layer = make_layer(4, 3, weight_vals, bias_vals);
	RMatrix *output = r_layer_forward(layer, inputs);

	assert_matrix_close(output, expected, 3, 3, "batch forward");

	r_free_matrix(inputs);
	r_free_matrix(output);
	r_free_layer(layer);
}

static void test_layer_forward_bias_only(void)
{
	const float input_vals[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	const float weight_vals[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	const float bias_vals[] = {1.0f, -2.0f};

	const float expected[] = {1.0f, -2.0f, 1.0f, -2.0f};

	RMatrix *inputs = make_matrix(2, 3, input_vals);
	RLayerDense *layer = make_layer(3, 2, weight_vals, bias_vals);
	RMatrix *output = r_layer_forward(layer, inputs);

	assert_matrix_close(output, expected, 2, 2, "bias only output");

	r_free_matrix(inputs);
	r_free_matrix(output);
	r_free_layer(layer);
}

static void test_layer_forward_zero_rows_no_crash(void)
{
	RMatrix *inputs = make_matrix(0, 3, NULL);
	RLayerDense *layer = make_layer(3, 2, NULL, NULL);
	RMatrix *output = r_layer_forward(layer, inputs);

	assert(output->rows == 0 && output->cols == 2 &&
	       "zero-row output shape mismatch");

	r_free_matrix(inputs);
	r_free_matrix(output);
	r_free_layer(layer);
}

static void test_layer_forward_nan_propagation(void)
{
	const float input_vals[] = {NAN, 1.0f};
	const float weight_vals[] = {1.0f, 1.0f};
	const float bias_vals[] = {0.0f};

	RMatrix *inputs = make_matrix(1, 2, input_vals);
	RLayerDense *layer = make_layer(2, 1, weight_vals, bias_vals);
	RMatrix *output = r_layer_forward(layer, inputs);

	assert(isnan(output->data[0]) &&
	       "nan inputs should propagate to output");

	r_free_matrix(inputs);
	r_free_matrix(output);
	r_free_layer(layer);
}

int main(void)
{
	test_create_layer_shapes();
	test_layer_forward_single_neuron();
	test_layer_forward_single_input_three_neurons();
	test_layer_forward_batch_matches_expected();
	test_layer_forward_bias_only();
	test_layer_forward_zero_rows_no_crash();
	test_layer_forward_nan_propagation();

	printf("All simple layer tests passed.\n");
	return 0;
}
