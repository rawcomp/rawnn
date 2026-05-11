#include "common.h"

int main(void)
{
    printf("=== NEURAL NETWORKS FROM SCRATCH (C) n===\n\n");

    printf("1. SINGLE NEURON\n");
    float inputs_single[3] = {1.0, 2.0, 3.0};
    float weights_single[3] = {0.2, 0.8, -0.5};
    float bias_single = 2.0;

    float output_single = 0.0;
    for (int i = 0; i < 3; i++)
    {
        output_single += inputs_single[i] * weights_single[i];
    }
    output_single += bias_single;
    printf("Output = %.2f  (expected: 2.3)\n\n", output_single);

    printf("2. LAYER WITH 3 NEURONS\n");
    float inputs_layer[4] = {1.0, 2.0, 3.0, 2.5};

    float weights1[4] = {0.2, 0.8, -0.5, 1.0};
    float b1 = 2.0;
    float weights2[4] = {0.5, -0.91, 0.26, -0.5};
    float b2 = 3.0;
    float weights3[4] = {-0.26, -0.27, 0.17, 0.87};
    float b3 = 0.5;

    float out1 = 0, out2 = 0, out3 = 0;
    for (int i = 0; i < 4; i++)
    {
        out1 += inputs_layer[i] * weights1[i];
        out2 += inputs_layer[i] * weights2[i];
        out3 += inputs_layer[i] * weights3[i];
    }
    out1 += b1;
    out2 += b2;
    out3 += b3;

    printf("Neuron 1: %.2f\n", out1);
    printf("Neuron 2: %.2f\n", out2);
    printf("Neuron 3: %.2f\n\n", out3);

    printf("3. BATCH + LAYER\n");

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

    r_print_matrix(output_batch, "Layer output (batch)");

    r_free_matrix(batch);
    r_free_matrix(output_batch);
    r_free_layer(layer);

    return 0;
}