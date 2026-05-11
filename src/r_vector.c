#include <rc/r_matrix.h>
#include <rc/r_types.h>
#include <rc/r_vector.h>

RVector *r_create_vector(size_t size)
{
    RVector *vector = malloc(sizeof(RVector));

    vector->size = size;
    vector->data = malloc(sizeof(float) * vector->size);

    return vector;
}

void r_free_vector(RNONNULL RVector *vector)
{
    free(vector->data);
    vector->size = 0;
    free(vector);
}

RVector *r_mat_vec_mul(const RNONNULL RMatrix *matrix, const RNONNULL RVector *vector)
{
    if (matrix->cols != vector->size)
    {
        printf("[ERROR]: The number of columns in the matrix should be the same as the vector size\n");
        return NULL;
    }

    RVector *result = malloc(sizeof(RVector));
    result->size = matrix->rows;
    result->data = malloc(sizeof(float) * result->size);

    for (size_t i = 0; i < matrix->rows; i++)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < matrix->cols; j++)
        {
            sum += matrix->data[RMatrixIDX(i, j, matrix->cols)] * vector->data[j];
        }
        result->data[i] = sum;
    }

    return result;
}

float r_vec_dot(const RNONNULL RVector *vector1, const RNONNULL RVector *vector2)
{
    if (vector1->size != vector2->size)
    {
        printf("[ERROR]: The vectors should have the same size\n");
        exit(1);
    }

    float result = 0.0f;
    for (size_t i = 0; i < vector1->size; i++)
    {
        result += vector1->data[i] * vector2->data[i];
    }

    return result;
}

void r_add_bias(const RNONNULL RVector *vector, float bias)
{
    for (size_t i = 0; i < vector->size; i++)
    {
        vector->data[i] += bias;
    }
}

void r_print_vector(RNONNULL RVector *vector, const RNONNULL char *name)
{
    printf("%s = [", name);
    for (size_t i = 0; i < vector->size; i++)
    {
        printf("%.2f", vector->data[i]);
        if (i < vector->size - 1)
            printf(", ");
    }
    printf("]\n");
}
