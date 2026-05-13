#ifndef RC_R_LOSS_GRAD_H
#define RC_R_LOSS_GRAD_H

#include <rc/r_types.h>
#include <rc/r_matrix.h>

void r_cross_entropy_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, RNONNULL RMatrix *grad);
void r_bin_cross_entropy_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, RNONNULL RMatrix *grad);
void r_cat_cross_entropy_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, RNONNULL RMatrix *grad);
void r_mse_loss_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, RNONNULL RMatrix *grad);
void r_mae_loss_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, RNONNULL RMatrix *grad);
void r_bin_focal_loss_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, float gamma, float alpha, RNONNULL RMatrix *grad);
void r_cat_focal_loss_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, float gamma, RNONNULL RMatrix *grad);

#endif
