#ifndef RC_R_ACTIVATION_GRAD_H
#define RC_R_ACTIVATION_GRAD_H

#include <rc/r_matrix.h>
#include <rc/r_types.h>

void r_activation_relu_grad(const RMatrix *input, const RMatrix *upstream_grad,
			    RMatrix *grad);
void r_activation_leaky_relu_grad(const RMatrix *input,
				  const RMatrix *upstream_grad, float alpha,
				  RMatrix *grad);
void r_activation_gelu_grad(const RMatrix *input, const RMatrix *upstream_grad,
			    RMatrix *grad);
void r_activation_softmax_grad(const RMatrix *output,
			       const RMatrix *upstream_grad, RMatrix *grad);
void r_activation_swish_grad(const RMatrix *input, const RMatrix *upstream_grad,
			     RMatrix *grad);

#endif