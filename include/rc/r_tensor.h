#ifndef RC_R_TENSOR_H
#define RC_R_TENSOR_H

#include <rc/r_types.h>

/**
 * struct r_tensor_t - N-dimensional tensor descriptor.
 * @data: Contiguous data buffer for tensor elements.
 * @n_dim: Number of dimensions.
 * @shape: Array of dimension sizes with length @n_dim.
 * @stride: Array of strides with length @n_dim.
 * @size: Total number of elements.
 */
typedef struct r_tensor_t
{
	float *data;
	size_t n_dim;
	size_t *shape;
	size_t *stride;	
	size_t size;
} RTensorND;



#endif
