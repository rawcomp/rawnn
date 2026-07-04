#include <rc/r_layer_dense.h>

/**
 * r_create_layer() - Allocate a dense layer with weights and biases.
 * @n_inputs: Number of input features.
 * @n_neurons: Number of neurons in the layer.
 *
 * Allocates a layer structure, a weight matrix of shape
 * (@n_neurons, @n_inputs), and a bias vector of length @n_neurons. The
 * allocated buffers are not initialized.
 * Return: Pointer to the newly allocated layer.
 */
RLayerDense *r_create_layer(size_t n_inputs, size_t n_neurons)
{
	RLayerDense *layer;

	layer = malloc(sizeof(*layer));
	if (!layer)
		return NULL;

	layer->biases = r_create_vector(n_neurons);
	layer->weights = r_create_matrix(n_neurons, n_inputs);

	if (!layer->biases || !layer->weights) {
		r_free_vector(layer->biases);
		r_free_matrix(layer->weights);
		free(layer);
		return NULL;
	}

	return layer;
}

/**
 * r_free_layer() - Free a dense layer and its buffers.
 * @layer: Layer to free.
 *
 * Releases the weight matrix, bias vector, and the layer itself.
 * Return: Nothing.
 */
void r_free_layer(RLayerDense *layer)
{
	if (!layer)
		return;

	r_free_matrix(layer->weights);
	r_free_vector(layer->biases);
	free(layer);
}

/**
 * r_layer_forward() - Compute the forward pass of a dense layer.
 * @layer: Layer containing weights and biases.
 * @inputs: Input matrix where each row is a sample.
 *
 * Computes inputs * weights^T + biases efficiently without allocating
 * a transposed weight matrix. Uses an i-k-j loop order.
 * Return: Newly allocated matrix containing the layer output.
 */
RMatrix *r_layer_forward(const RLayerDense *layer, const RMatrix *inputs)
{
	RMatrix *result;
	size_t i;
	size_t j;
	size_t k;

	if (!layer || !inputs)
		return NULL;

	if (inputs->cols != layer->weights->cols) {
		fprintf(stderr, "[ERROR]: r_layer_forward requires "
				"inputs->cols == layer->weights->cols\n");
		return NULL;
	}
	if (layer->biases->size != layer->weights->rows) {
		fprintf(stderr, "[ERROR]: r_layer_forward requires "
				"biases->size == number of neurons\n");
		return NULL;
	}

	result = r_create_matrix(inputs->rows, layer->weights->rows);
	if (!result)
		return NULL;

	for (i = 0; i < inputs->rows; i++) {
		const float *in_row = &inputs->data[i * inputs->cols];
		float *out_row = &result->data[i * result->cols];

		for (j = 0; j < layer->weights->rows; j++)
			out_row[j] = layer->biases->data[j];

		for (k = 0; k < inputs->cols; k++) {
			float in_val = in_row[k];

			for (j = 0; j < layer->weights->rows; j++)
				out_row[j] +=
				    in_val *
				    layer->weights
					->data[j * layer->weights->cols + k];
		}
	}

	return result;
}
