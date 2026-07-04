#include <rc/r_tensor.h>
#include <string.h>

/**
 * r_create_tensor() - Allocate a tensor from a shape descriptor.
 * @n_dim: Number of dimensions.
 * @shape: Array of dimension sizes.
 *
 * Allocates the tensor object, copies the shape, computes the total
 * element count, derives the row-major strides, and allocates storage
 * for the tensor data.
 * Return: Newly allocated tensor, or NULL on allocation failure.
 */
RTensorND *r_create_tensor(size_t n_dim, const size_t *shape)
{
	RTensorND *tensor;
	size_t current_stride = 1;
	size_t i;
	size_t index;

	if (n_dim == 0 || !shape)
		return NULL;

	tensor = malloc(sizeof(*tensor));
	if (!tensor)
		return NULL;

	tensor->n_dim = n_dim;
	tensor->size = 1;

	tensor->shape = malloc(sizeof(*tensor->shape) * n_dim);
	if (!tensor->shape)
		goto out_free_tensor;

	tensor->stride = malloc(sizeof(*tensor->stride) * n_dim);
	if (!tensor->stride)
		goto out_free_shape;

	for (i = 0; i < n_dim; i++) {
		if (shape[i] == 0)
			goto out_free_stride;
		tensor->shape[i] = shape[i];
		tensor->size *= shape[i];
	}

	for (i = 0; i < n_dim; i++) {
		index = n_dim - 1 - i;
		tensor->stride[index] = current_stride;
		current_stride *= tensor->shape[index];
	}

	tensor->data = malloc(sizeof(*tensor->data) * tensor->size);
	if (!tensor->data)
		goto out_free_stride;

	return tensor;

out_free_stride:
	free(tensor->stride);
out_free_shape:
	free(tensor->shape);
out_free_tensor:
	free(tensor);
	return NULL;
}

/**
 * r_create_tensor_from_data() - Allocate a tensor and copy data into it.
 * @n_dim: Number of dimensions.
 * @shape: Array of dimension sizes.
 * @data: Source data buffer to copy into the tensor.
 *
 * Allocates the tensor object, copies the shape, computes the total
 * element count, derives the row-major strides, allocates storage for
 * the tensor data, and copies the provided elements into the new buffer.
 * Return: Newly allocated tensor, or NULL on allocation failure.
 */
RTensorND *r_create_tensor_from_data(size_t n_dim, const size_t *shape,
				     float *data)
{
	RTensorND *tensor;
	size_t current_stride = 1;
	size_t i;
	size_t index;

	if (n_dim == 0 || !shape || !data)
		return NULL;

	tensor = malloc(sizeof(*tensor));
	if (!tensor)
		return NULL;

	tensor->n_dim = n_dim;
	tensor->size = 1;

	tensor->shape = malloc(sizeof(*tensor->shape) * n_dim);
	if (!tensor->shape)
		goto out_free_tensor;

	tensor->stride = malloc(sizeof(*tensor->stride) * n_dim);
	if (!tensor->stride)
		goto out_free_shape;

	for (i = 0; i < n_dim; i++) {
		if (shape[i] == 0)
			goto out_free_stride;
		tensor->shape[i] = shape[i];
		tensor->size *= shape[i];
	}

	for (i = 0; i < n_dim; i++) {
		index = n_dim - 1 - i;
		tensor->stride[index] = current_stride;
		current_stride *= tensor->shape[index];
	}

	tensor->data = malloc(sizeof(*tensor->data) * tensor->size);
	if (!tensor->data)
		goto out_free_stride;

	memcpy(tensor->data, data, sizeof(*tensor->data) * tensor->size);

	return tensor;

out_free_stride:
	free(tensor->stride);
out_free_shape:
	free(tensor->shape);
out_free_tensor:
	free(tensor);
	return NULL;
}

/**
 * r_free_tensor() - Free a tensor and its owned buffers.
 * @tensor: Tensor to free.
 *
 * Releases the shape, stride, and data buffers before freeing the tensor
 * object itself.
 * Return: Nothing.
 */
void r_free_tensor(RTensorND *tensor)
{
	if (!tensor)
		return;

	tensor->size = 0;
	tensor->n_dim = 0;
	free(tensor->shape);
	free(tensor->stride);
	free(tensor->data);
	free(tensor);
}
