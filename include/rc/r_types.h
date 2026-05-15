#ifndef RC_R_TYPES_H
#define RC_R_TYPES_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>

#define RC_VERSION_MAJOR 0
#define RC_VERSION_MINOR 1
#define RC_VERSION_PATCH 0

#define RC_VERSION_STRING "0.1.0"

#define RC_VERSION_ENCODE(major, minor, patch) \
    ((major) * 10000 + (minor) * 100 + (patch))

#define RC_VERSION \
    RC_VERSION_ENCODE(RC_VERSION_MAJOR, RC_VERSION_MINOR, RC_VERSION_PATCH)


#define RNONNULL
#define RNULLABLE

#define RMatrixIDX(i, j, cols) ((i) * (cols) + (j))
#define MatrixSize(matrix) ((matrix->rows) * (matrix->cols))

#define EPSILON 1e-7f

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

#endif
