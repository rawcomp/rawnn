#ifndef RC_R_OPTIMIZATION_H
#define RC_R_OPTIMIZATION_H

#include <rc/r_matrix.h>
#include <rc/r_types.h>

void r_optimization_sgd(RMatrix *theta, const RMatrix *grad, float eta);
void r_optimization_sgdm(RMatrix *theta, const RMatrix *grad, RMatrix *velocity,
			 float eta, float beta);
void r_optimization_rmsprop(RMatrix *theta, const RMatrix *grad, RMatrix *cache,
			    float eta, float rho, float eps);
void r_optimization_adam(RMatrix *theta, const RMatrix *grad, RMatrix *m,
			 RMatrix *v, float eta, float beta1, float beta2,
			 float eps, size_t t);

#endif
