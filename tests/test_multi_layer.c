#include "common.h"

int main(void)
{
    RLayerDense *l1 = r_create_layer(5, 10);

    RLayerDense *l2 = r_create_layer(10, 5);

    RLayerDense *l3 = r_create_layer(5, 2);

    float batchl1[10][5] = {{3.45, 1.22, 4.87, 0.54, 2.11}, {4.01, 2.76, 1.99, 3.33, 0.15},
                            {1.55, 4.44, 2.05, 3.81, 4.92}, {0.88, 1.12, 3.65, 2.45, 4.10},
                            {2.99, 0.45, 1.76, 4.56, 3.12}, {1.23, 3.44, 4.89, 2.67, 0.91},
                            {4.12, 2.34, 1.87, 3.55, 4.09}, {0.67, 1.98, 2.88, 4.76, 3.21},
                            {3.34, 4.65, 0.23, 1.56, 2.78}, {2.45, 3.89, 4.12, 1.34, 0.76}};

    float input[10][5] = {{1.87, 4.23, 3.11, 0.99, 2.45}, {4.65, 1.34, 2.87, 3.76, 0.45},
                          {2.12, 3.98, 4.55, 1.23, 3.67}, {0.55, 2.44, 1.89, 4.12, 3.33},
                          {3.78, 4.89, 0.67, 2.15, 1.45}, {4.34, 1.12, 2.56, 3.88, 4.91},
                          {1.99, 3.45, 4.76, 0.23, 2.88}, {2.67, 4.12, 1.34, 3.55, 0.87},
                          {3.89, 2.34, 4.45, 1.76, 2.11}, {0.45, 1.88, 3.67, 4.23, 2.99}};

    float biasesl1[10] = {1.25, 6.33, 4.12, 0.89, 5.55, 2.76, 3.91, 6.98, 1.05, 4.44};

    float biasesl2[5] = {3.14, 5.82, 0.45, 6.71, 2.39};

    float biasesl3[2] = {0.50, -0.50};

    for (size_t i = 0; i < l1->biases->size; i++)
    {
        l1->biases->data[i] = biasesl1[i];
    }
    for (size_t i = 0; i < l2->biases->size; i++)
    {
        l2->biases->data[i] = biasesl2[i];
    }

    for (size_t i = 0; i < l3->biases->size; i++)
    {
        l3->biases->data[i] = biasesl3[i];
    }

    for (size_t i = 0; i < l1->weights->rows; i++)
    {
        for (size_t j = 0; j < l1->weights->cols; j++)
        {
            l1->weights->data[RMatrixIDX(i, j, l1->weights->cols)] = batchl1[j][i];
        }
    }

    for (size_t i = 0; i < l2->weights->rows; i++)
    {
        for (size_t j = 0; j < l2->weights->cols; j++)
        {
            l2->weights->data[RMatrixIDX(i, j, l2->weights->cols)] = 0.1f * (i + j + 1);
        }
    }

    for (size_t i = 0; i < l3->weights->rows; i++)
    {
        for (size_t j = 0; j < l3->weights->cols; j++)
        {
            l3->weights->data[RMatrixIDX(i, j, l3->weights->cols)] = 0.2f * (i + 1) - 0.1f * j;
        }
    }

    RMatrix *input_matrix = r_create_matrix(10, 5);
    for (size_t i = 0; i < input_matrix->rows; i++)
    {
        for (size_t j = 0; j < input_matrix->cols; j++)
        {
            input_matrix->data[RMatrixIDX(i, j, input_matrix->cols)] = input[i][j];
        }
    }

    RMatrix *output_l1 = r_layer_forward(l1, input_matrix);
    r_print_matrix(output_l1, "OUTPUT FIRST LAYER");

    RMatrix *output_l2 = r_layer_forward(l2, output_l1);
    r_print_matrix(output_l2, "OUTPUT SECOND LAYER");

    RMatrix *output_l3 = r_layer_forward(l3, output_l2);
    r_print_matrix(output_l3, "OUTPUT THIRD LAYER");

    return 0;
}