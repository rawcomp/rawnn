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
    RLayerDense *layer = malloc(sizeof(RLayerDense));

    layer->biases = r_create_vector(n_neurons);
    layer->weights = r_create_matrix(n_neurons, n_inputs);

    return layer;
}

/**
 * r_free_layer() - Free a dense layer and its buffers.
 * @layer: Layer to free.
 *
 * Releases the weight matrix, bias vector, and the layer itself.
 * Return: Nothing.
 */
void r_free_layer(RNONNULL RLayerDense *layer)
{
    r_free_matrix(layer->weights);
    r_free_vector(layer->biases);
    free(layer);
}

/**
 * r_layer_forward() - Compute the forward pass of a dense layer.
 * @layer: Layer containing weights and biases.
 * @inputs: Input matrix where each row is a sample.
 *
 * Multiplies @inputs by the transpose of @layer->weights and adds the
 * bias vector to each row of the result.
 * Return: Newly allocated matrix containing the layer output.
 */
RMatrix *r_layer_forward(const RNONNULL RLayerDense *layer, const RNONNULL RMatrix *inputs)
{
    RMatrix *transposed_weights = r_mat_transpose(layer->weights);
    RMatrix *result = r_mat_mul(inputs, transposed_weights);

    r_free_matrix(transposed_weights);

    for (size_t i = 0; i < result->rows; i++)
    {
        // seg fault?
        // r_add_bias(result->data, layer->biases->data[i]);
        for (size_t j = 0; j < result->cols; j++)
        {
            result->data[RMatrixIDX(i, j, result->cols)] += layer->biases->data[j];
        }
    }
    return result;
}
