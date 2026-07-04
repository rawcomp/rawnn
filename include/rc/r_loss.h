#ifndef RC_R_LOSS_H
#define RC_R_LOSS_H

#include <rc/r_matrix.h>
#include <rc/r_types.h>

float r_cross_entropy(const RMatrix *matrix, const RMatrix *src);
float r_bin_cross_entropy(const RMatrix *pred, const RMatrix *real);
float r_cat_cross_entropy(const RMatrix *pred, const RMatrix *real);
float r_mse_loss(const RMatrix *pred, const RMatrix *real);
float r_mae_loss(const RMatrix *pred, const RMatrix *real);
float r_cat_focal_loss(const RMatrix *pred, const RMatrix *real, float gamma);
float r_bin_focal_loss(const RMatrix *pred, const RMatrix *real, float gamma,
		       float alpha);

#endif
