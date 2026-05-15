#ifndef RC_R_LAYER_DENSE_H
#define RC_R_LAYER_DENSE_H

#include <rc/r_types.h>
#include <rc/r_matrix.h>
#include <rc/r_vector.h>

/**
 * struct r_layer_dense_t - Parameters for a dense layer.
 * @weights: Weight matrix of shape (n_neurons, n_inputs).
 * @biases: Bias vector of length n_neurons.
 */
typedef struct r_layer_dense_t
{
    RMatrix *weights;
    RVector *biases;
} RLayerDense;

RLayerDense *r_create_layer(size_t n_inputs, size_t n_neurons);
void r_free_layer(RNONNULL RLayerDense *layer);
RMatrix *r_layer_forward(const RNONNULL RLayerDense *layer, const RNONNULL RMatrix *inputs);

#endif
