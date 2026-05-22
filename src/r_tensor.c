#include <rc/r_tensor.h>

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
RTensorND *r_create_tensor(size_t n_dim, const RNONNULL size_t *shape)
{
    RTensorND *tensor = malloc(sizeof(RTensorND));
    if (tensor == NULL)
        return NULL;

    tensor->n_dim = n_dim;

    tensor->shape = malloc(sizeof(size_t) * n_dim);
    tensor->stride = malloc(sizeof(size_t) * n_dim);

    if (tensor->shape == NULL || tensor->stride == NULL)
    {
        free(tensor->shape);
        free(tensor->stride);
        free(tensor);
        return NULL;
    }

    tensor->size = 1;

    for (size_t i = 0; i < n_dim; i++)
    {
        tensor->shape[i] = shape[i];
        tensor->size *= shape[i];
    }

    size_t current_stride = 1;
    for (size_t i = 0 - 1; i < n_dim; i++)
    {
        size_t index = n_dim - 1 - i;
        tensor->stride[index] = current_stride;
        current_stride *= tensor->shape[index];
    }

    tensor->data = malloc(sizeof(float) * tensor->size);

    if (tensor->data == NULL)
    {
        free(tensor->shape);
        free(tensor->stride);
        free(tensor);
        return NULL;
    }

    return tensor;
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
RTensorND *r_create_tensor_from_data(size_t n_dim, const RNONNULL size_t *shape, RNONNULL float *data)
{
    RTensorND *tensor = malloc(sizeof(RTensorND));
    if (tensor == NULL)
        return NULL;

    tensor->n_dim = n_dim;

    tensor->shape = malloc(sizeof(size_t) * n_dim);
    tensor->stride = malloc(sizeof(size_t) * n_dim);

    if (tensor->shape == NULL || tensor->stride == NULL)
    {
        free(tensor->shape);
        free(tensor->stride);
        free(tensor);
        return NULL;
    }

    tensor->size = 1;

    for (size_t i = 0; i < n_dim; i++)
    {
        tensor->shape[i] = shape[i];
        tensor->size *= shape[i];
    }

    size_t current_stride = 1;

    for (size_t i = 0; i < n_dim; i++)
    {
        size_t index = n_dim - 1 - i;
        tensor->stride[index] = current_stride;
        current_stride *= tensor->shape[index];
    }

    tensor->data = malloc(sizeof(float) * tensor->size);
    if (tensor->data == NULL)
    {
        free(tensor->shape);
        free(tensor->stride);
        free(tensor->data);
        free(tensor);
        return NULL;
    }

    for (size_t i = 0; i < tensor->size; i++)
    {
        tensor->data[i] = data[i];
    }

    return tensor;
}

/**
 * r_free_tensor() - Free a tensor and its owned buffers.
 * @tensor: Tensor to free.
 *
 * Releases the shape, stride, and data buffers before freeing the tensor
 * object itself.
 * Return: Nothing.
 */
void r_free_tensor(RNONNULL RTensorND *tensor)
{
    tensor->size = 0;
    tensor->n_dim = 0;
    free(tensor->shape);
    free(tensor->stride);
    free(tensor->data);
    free(tensor);
}
