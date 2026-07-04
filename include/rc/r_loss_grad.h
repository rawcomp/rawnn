#ifndef RC_R_LOSS_GRAD_H
#define RC_R_LOSS_GRAD_H

#include <rc/r_matrix.h>
#include <rc/r_types.h>

void r_cross_entropy_grad(const RMatrix *pred, const RMatrix *real,
			  RMatrix *grad);
void r_bin_cross_entropy_grad(const RMatrix *pred, const RMatrix *real,
			      RMatrix *grad);
void r_cat_cross_entropy_grad(const RMatrix *pred, const RMatrix *real,
			      RMatrix *grad);
void r_mse_loss_grad(const RMatrix *pred, const RMatrix *real, RMatrix *grad);
void r_mae_loss_grad(const RMatrix *pred, const RMatrix *real, RMatrix *grad);
void r_bin_focal_loss_grad(const RMatrix *pred, const RMatrix *real,
			   float gamma, float alpha, RMatrix *grad);
void r_cat_focal_loss_grad(const RMatrix *pred, const RMatrix *real,
			   float gamma, RMatrix *grad);

#endif
