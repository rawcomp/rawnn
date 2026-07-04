#ifndef RC_R_VECTOR_H
#define RC_R_VECTOR_H

#include <rc/r_matrix.h>
#include <rc/r_types.h>
#include <stdio.h>

/**
 * struct r_vector_t - Simple vector container.
 * @data: Contiguous data buffer of size @size.
 * @size: Number of elements.
 */
typedef struct r_vector_t {
	float *data;
	size_t size;
} RVector;

RVector *r_create_vector(size_t size);
void r_free_vector(RVector *vector);
void r_print_vector(FILE *stream, const RVector *vector, const char *name);
float r_vector_dot(const RVector *vector1, const RVector *vector2);
void r_vector_add_bias(RVector *vector, const RVector *bias);

RVector *r_mat_vec_mul(const RMatrix *matrix, const RVector *vector);

#endif
