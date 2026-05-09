#include <rc/r_types.h>
#include <rc/r_matrix.h>

float r_cross_entropy(const RNONNULL RMatrix *matrix, const RNONNULL RMatrix *src);
float r_mse_loss(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real);
float r_mae_loss(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real);

