#ifndef RC_R_ACTIVATION_H
#define RC_R_ACTIVATION_H

#include <rc/r_matrix.h>
#include <rc/r_types.h>

RMatrix *r_activation_relu(const RMatrix *matrix);
void r_activation_relu_inplace(RMatrix *matrix);

RMatrix *r_activation_leaky_relu(const RMatrix *matrix, float alpha);
void r_activation_leaky_relu_inplace(RMatrix *matrix, float alpha);

RMatrix *r_activation_gelu(const RMatrix *matrix);
void r_activation_gelu_inplace(RMatrix *matrix);

RMatrix *r_activation_softmax(const RMatrix *matrix);
void r_activation_softmax_inplace(RMatrix *matrix);

RMatrix *r_activation_swish(const RMatrix *matrix);
void r_activation_swish_inplace(RMatrix *matrix);

#endif
