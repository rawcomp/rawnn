#ifndef RC_R_OPTIMIZATION_H
#define RC_R_OPTIMIZATION_H

#include <rc/r_types.h>
#include <rc/r_matrix.h>

void r_optimization_sgd(RNONNULL RMatrix *theta, const RNONNULL RMatrix *grad, float eta);
void r_optimization_sgdm(RNONNULL RMatrix *theta, const RNONNULL RMatrix *grad, RNONNULL RMatrix *velocity, float eta, float beta);
void r_optimization_rmsprop(RNONNULL RMatrix *theta, const RNONNULL RMatrix *grad, RNONNULL RMatrix *cache, float eta, float rho, float eps);
void r_optimization_adam(RNONNULL RMatrix *theta, const RNONNULL RMatrix *grad, RNONNULL RMatrix *m, RNONNULL RMatrix *v, float eta, float beta1, float beta2, float eps, size_t t);

#endif
