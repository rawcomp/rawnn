#ifndef RC_R_VECTOR_H
#define RC_R_VECTOR_H

#include <rc/r_types.h>
#include <rc/r_matrix.h>

/**
 * struct r_vector_t - Simple vector container.
 * @data: Contiguous data buffer of size @size.
 * @size: Number of elements.
 */
typedef struct r_vector_t
{
    float *data;
    size_t size;
} RVector;

RVector *r_create_vector(size_t size);
void r_free_vector(RNONNULL RVector *vector);
void r_print_vector(RNONNULL RVector *vector, const RNONNULL char *name);
float r_vec_dot(const RNONNULL RVector *vector1, const RNONNULL RVector *vector2);
void r_add_bias(RNONNULL RVector *vector, float bias);

RVector *r_mat_vec_mul(const RNONNULL RMatrix *matrix, const RNONNULL RVector *vector);

#endif
