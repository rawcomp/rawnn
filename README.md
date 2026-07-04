# RawCompute - Neural Networks from Scratch in C

A lightweight, educational framework for building neural networks in pure C, designed to understand the fundamentals of deep learning without external dependencies.

## Project Overview

RawCompute demonstrates how neural networks are built from first principles, providing low-level implementations of:
- **Matrix operations** (multiplication, transposition, indexing)
- **Vector operations** (dot product, vector-matrix multiplication, bias addition)
- **Dense layers** (fully connected layers with weights and biases)
- **Activation functions** (ReLU, Leaky ReLU, Softmax, Swish, GELU)
- **Loss functions** (MSE, Cross-Entropy)
- **Optimization** (SGD, Adam)

This project is perfect for learning how neural network computations work at the C level, before moving to high-level frameworks like TensorFlow or PyTorch.

## Dependencies
- Standard C99 compiler (e.g. `gcc` or `clang`)
- GNU `make`
- `ar` (for archiving static library)
- `libm` (Standard math library)
- `valgrind` (Optional, for memory leak testing)
- `clang-format` (Optional, for formatting)

## Project Status

🚧 **Work in Progress** - This project is actively being developed as I work through [Neural Networks from Scratch in Python](https://nnfs.io/). Each chapter introduces new concepts that are being translated and implemented in C to deepen understanding of neural network fundamentals.

**Current Progress:**
- ✅ Basic tensor operations (matrices and vectors)
- ✅ Dense layers with forward propagation
- ✅ Activation functions
- ✅ Loss functions
- ✅ Optimization
- ⏳ Backpropagation (Full graph)
- ⏳ Additional layer types

## Core Components

### 1. Types & Macros (`r_types.h/c`)
Basic macros for versioning and utility definitions like `EPSILON` and matrix indexing `R_MATRIX_IDX`.
The umbrella header for the library is `<rc/rc.h>`.

### 2. Matrix Operations (`r_matrix.h/c`)
Implements a 2D matrix structure and operations: `r_create_matrix`, `r_free_matrix`, `r_matrix_mul`, `r_matrix_transpose`, `r_print_matrix`.

### 3. Vector Operations (`r_vector.h/c`)
Implements a 1D vector structure and operations: `r_create_vector`, `r_free_vector`, `r_vector_dot`, `r_vector_add_bias`, `r_mat_vec_mul`.

### 4. Tensor Operations (`r_tensor.h/c`)
N-dimensional tensor support for multi-dimensional operations: `r_create_tensor`, `r_create_tensor_from_data`, `r_free_tensor`.

### 5. Dense Layer (`r_layer_dense.h/c`)
Implements a fully connected layer: `r_create_layer`, `r_free_layer`, `r_layer_forward`.

### 6. Activations (`r_activation.h/c`)
Functions for non-linear activations: `r_activation_relu`, `r_activation_softmax`, `r_activation_gelu`, `r_activation_swish`.

### 7. Activation Gradients (`r_activation_grad.h/c`)
Gradient calculations for backpropagation of activations.

### 8. Loss (`r_loss.h/c`)
Functions for error measurement: `r_mse_loss`, `r_cross_entropy`, `r_focal_loss`.

### 9. Loss Gradients (`r_loss_grad.h/c`)
Gradient calculations for loss functions.

### 10. Optimization (`r_optimization.h/c`)
Algorithms for updating weights: `r_optimization_sgd`, `r_optimization_adam`.

## Building and Running

### Build
```bash
make
```

### Run Tests
```bash
make test
```

### Run Valgrind
```bash
make valgrind
```

## Usage Example

```c
#include <rc/rc.h>

// Create a dense layer with 4 inputs and 3 neurons
RLayerDense *layer = r_create_layer(4, 3);

// Create batch of 2 samples with 4 features each
RMatrix *input = r_create_matrix(2, 4);
// ... populate input data ...

// Forward pass
RMatrix *output = r_layer_forward(layer, input);

// Print results
r_print_matrix(stdout, output, "Layer Output");

// Cleanup
r_free_matrix(input);
r_free_matrix(output);
r_free_layer(layer);
```

## Limitations and Future Improvements

Currently planned features being implemented:
- Backpropagation implementation
- Support for multiple layer types (Convolutional, Recurrent, etc.)
- Multi-threading support (OpenMP)
- Batched operations across multiple layers

**Inspiration:** This project follows the curriculum and concepts from [Neural Networks from Scratch in Python](https://nnfs.io/) but implements them in C for educational purposes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
