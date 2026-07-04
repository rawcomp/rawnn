#include <rc/r_vector.h>
#include <stdint.h>

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
	RVector *vector;

	if (size > SIZE_MAX / sizeof(float))
		return NULL;

	vector = malloc(sizeof(*vector));
	if (!vector)
		return NULL;

	vector->size = size;

	if (size == 0) {
		vector->data = NULL;
		return vector;
	}

	vector->data = malloc(sizeof(*vector->data) * size);
	if (!vector->data) {
		free(vector);
		return NULL;
	}

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
void r_free_vector(RVector *vector)
{
	if (!vector)
		return;

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
RVector *r_mat_vec_mul(const RMatrix *matrix, const RVector *vector)
{
	RVector *result;
	size_t i;
	size_t j;

	if (!matrix || !vector || matrix->cols != vector->size) {
		fprintf(stderr, "[ERROR]: The number of columns in the matrix "
				"should be the same as the vector size\n");
		return NULL;
	}

	result = r_create_vector(matrix->rows);
	if (!result)
		return NULL;

	for (i = 0; i < matrix->rows; i++) {
		double sum = 0.0;
		const float *row = &matrix->data[i * matrix->cols];

		for (j = 0; j < matrix->cols; j++)
			sum += (double)row[j] * vector->data[j];

		result->data[i] = (float)sum;
	}

	return result;
}

/**
 * r_vector_dot() - Compute dot product of two vectors.
 * @vector1: First vector.
 * @vector2: Second vector.
 *
 * Computes the dot product when sizes match. On size mismatch, prints
 * an error and returns NAN.
 * Return: Dot product of the vectors, or NAN on size mismatch.
 */
float r_vector_dot(const RVector *vector1, const RVector *vector2)
{
	double result = 0.0;
	size_t i;

	if (!vector1 || !vector2 || vector1->size != vector2->size) {
		fprintf(stderr,
			"[ERROR]: The vectors should have the same size\n");
		return NAN;
	}

	for (i = 0; i < vector1->size; i++)
		result += (double)vector1->data[i] * vector2->data[i];

	return (float)result;
}

/**
 * r_vector_add_bias() - Add a bias vector to each element of a vector.
 * @vector: Vector updated in place.
 * @bias: Bias vector to add.
 *
 * Adds @bias to every element in @vector.
 * Return: Nothing.
 */
void r_vector_add_bias(RVector *vector, const RVector *bias)
{
	size_t i;

	if (!vector || !bias || vector->size != bias->size) {
		fprintf(stderr,
			"[ERROR]: The vectors should have the same size\n");
		return;
	}

	for (i = 0; i < vector->size; i++)
		vector->data[i] += bias->data[i];
}

/**
 * r_print_vector(stdout, ) - Print a vector with a label.
 * @stream: Output stream.
 * @vector: Vector to print.
 * @name: Label to print before the vector.
 *
 * Outputs the vector values to @stream using a fixed two-decimal format.
 * Return: Nothing.
 */
void r_print_vector(FILE *stream, const RVector *vector, const char *name)
{
	size_t i;

	if (!stream || !vector || !name)
		return;

	fprintf(stream, "%s = [", name);
	for (i = 0; i < vector->size; i++) {
		fprintf(stream, "%.2f", vector->data[i]);
		if (i < vector->size - 1)
			fprintf(stream, ", ");
	}
	fprintf(stream, "]\n");
}
