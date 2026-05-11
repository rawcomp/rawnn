#include "common.h"

int main(void)
{
    float batch_data[3][4] = {{1.0, 2.0, 3.0, 2.5}, {2.0, 5.0, -1.0, 2.0}, {3.0, 1.0, 4.0, 0.5}};

    RMatrix *batch = r_create_matrix(3, 4);
    // printf("Matrix created\n");

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            batch->data[i * 4 + j] = batch_data[i][j];
        }
    }

    RLayerDense *layer = malloc(sizeof(RLayerDense));
    layer->weights = r_create_matrix(3, 4);
    layer->biases = r_create_vector(3);
    // printf("Layer created\n");

    float w_data[3][4] = {{0.2, 0.8, -0.5, 1.0}, {0.5, -0.91, 0.26, -0.5}, {-0.26, -0.27, 0.17, 0.87}};
    float b_data[3] = {2.0, 3.0, 0.5};

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            layer->weights->data[i * 4 + j] = w_data[i][j];
        }
        layer->biases->data[i] = b_data[i];
    }

    RMatrix *output_batch = r_layer_forward(layer, batch);
    // printf("Layer forwared\n");

    RMatrix *probabilities = r_activation_softmax(output_batch);
    r_print_matrix(probabilities, "Probabilities (SOFTMAX)");

    r_free_matrix(batch);
    r_free_matrix(output_batch);
    r_free_layer(layer);
    r_free_matrix(probabilities);

    return 0;
}