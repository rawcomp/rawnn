#include <rc/r_vector.h>

/**
 * r_create_vector() - Allocate a vector.
 * @size: Number of elements.
 *
 * Allocates the vector structure and a contiguous data buffer of @size
 * floats. The data buffer is not initialized.
 * Return: Pointer to the newly allocated vector.
 */
RVector *r_create_vector(size_t size)
{
    RVector *vector = malloc(sizeof(RVector));

    vector->size = size;
    vector->data = malloc(sizeof(float) * vector->size);

    return vector;
}

/**
 * r_free_vector() - Free a vector and its data buffer.
 * @vector: Vector to free.
 *
 * Releases the vector data buffer, clears its size, and frees the
 * vector structure.
 * Return: Nothing.
 */
void r_free_vector(RNONNULL RVector *vector)
{
    free(vector->data);
    vector->size = 0;
    free(vector);
}

/**
 * r_mat_vec_mul() - Multiply a matrix by a vector.
 * @matrix: Input matrix.
 * @vector: Input vector.
 *
 * Computes matrix-vector multiplication when @matrix->cols equals
 * @vector->size. On size mismatch, prints an error and returns NULL.
 * Return: Newly allocated result vector, or NULL on size mismatch.
 */
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

/**
 * r_vec_dot() - Compute dot product of two vectors.
 * @vector1: First vector.
 * @vector2: Second vector.
 *
 * Computes the dot product when sizes match. On size mismatch, prints
 * an error and returns 0.0f.
 * Return: Dot product of the vectors, or 0.0f on size mismatch.
 */
float r_vec_dot(const RNONNULL RVector *vector1, const RNONNULL RVector *vector2)
{
    if (vector1->size != vector2->size)
    {
        printf("[ERROR]: The vectors should have the same size\n");
        return 0.0f;
    }

    float result = 0.0f;
    for (size_t i = 0; i < vector1->size; i++)
    {
        result += vector1->data[i] * vector2->data[i];
    }

    return result;
}

/**
 * r_add_bias() - Add a bias term to each element of a vector.
 * @vector: Vector updated in place.
 * @bias: Bias value to add.
 *
 * Adds @bias to every element in @vector.
 * Return: Nothing.
 */
void r_add_bias(RNONNULL RVector *vector, float bias)
{
    for (size_t i = 0; i < vector->size; i++)
    {
        vector->data[i] += bias;
    }
}

/**
 * r_print_vector() - Print a vector with a label.
 * @vector: Vector to print.
 * @name: Label to print before the vector.
 *
 * Outputs the vector values to stdout using a fixed two-decimal format.
 * Return: Nothing.
 */
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
