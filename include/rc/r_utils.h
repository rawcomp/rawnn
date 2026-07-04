#ifndef RC_R_UTILS_H
#define RC_R_UTILS_H

#include <rc/r_matrix.h>

/**
 * r_has_same_shape() - Check if two matrices have identical dimensions.
 * @lhs: First matrix.
 * @rhs: Second matrix.
 *
 * Return: 1 if dimensions match exactly, 0 otherwise or if NULL.
 */
static inline int r_has_same_shape(const RMatrix *lhs, const RMatrix *rhs)
{
	if (!lhs || !rhs)
		return 0;
	return lhs->rows == rhs->rows && lhs->cols == rhs->cols;
}

#endif
