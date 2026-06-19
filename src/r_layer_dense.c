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
    if (!layer)
        return NULL;

    layer->biases = r_create_vector(n_neurons);
    layer->weights = r_create_matrix(n_neurons, n_inputs);

    if (!layer->biases || !layer->weights)
    {
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
void r_free_layer(RNONNULL RLayerDense *layer)
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
 * a transposed weight matrix.
 * Return: Newly allocated matrix containing the layer output.
 */
RMatrix *r_layer_forward(const RNONNULL RLayerDense *layer, const RNONNULL RMatrix *inputs)
{
    if (inputs->cols != layer->weights->cols)
    {
        printf("[ERROR]: r_layer_forward requires inputs->cols == layer->weights->cols\n");
        return NULL;
    }
    if (layer->biases->size != layer->weights->rows)
    {
        printf("[ERROR]: r_layer_forward requires biases->size == number of neurons\n");
        return NULL;
    }

    RMatrix *result = r_create_matrix(inputs->rows, layer->weights->rows);
    if (!result)
        return NULL;

    // Fused implicit transposition and matrix multiplication plus bias addition
    // Computes: result[i][j] = sum_k(inputs[i][k] * weights[j][k]) + biases[j]
    for (size_t i = 0; i < inputs->rows; i++)
    {
        const float *in_row = &inputs->data[i * inputs->cols];
        float *out_row = &result->data[i * result->cols];

        for (size_t j = 0; j < layer->weights->rows; j++)
        {
            const float *w_row = &layer->weights->data[j * layer->weights->cols];
            float sum = layer->biases->data[j]; // Initialize with bias
            
            for (size_t k = 0; k < inputs->cols; k++)
            {
                sum += in_row[k] * w_row[k];
            }
            out_row[j] = sum;
        }
    }

    return result;
}
