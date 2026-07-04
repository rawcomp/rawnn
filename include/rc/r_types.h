#ifndef RC_R_TYPES_H
#define RC_R_TYPES_H

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * RC_VERSION_MAJOR - Major version number.
 */
#define RC_VERSION_MAJOR 0
/**
 * RC_VERSION_MINOR - Minor version number.
 */
#define RC_VERSION_MINOR 1
/**
 * RC_VERSION_PATCH - Patch version number.
 */
#define RC_VERSION_PATCH 0

/**
 * RC_VERSION_STRING - String form of the semantic version.
 */
#define RC_VERSION_STRING "0.1.0"

/**
 * RC_VERSION_ENCODE() - Encode semantic version into an integer.
 * @major: Major version number.
 * @minor: Minor version number.
 * @patch: Patch version number.
 */
#define RC_VERSION_ENCODE(major, minor, patch)                                 \
	((major) * 10000 + (minor) * 100 + (patch))

/**
 * RC_VERSION - Encoded version integer using RC_VERSION_*.
 */
#define RC_VERSION                                                             \
	RC_VERSION_ENCODE(RC_VERSION_MAJOR, RC_VERSION_MINOR, RC_VERSION_PATCH)

/**
 * - Annotation macro for non-null pointers.
 */
#define RNONNULL
/**
 * - Annotation macro for nullable pointers.
 */
#define RNULLABLE

/**
 * R_MATRIX_IDX() - Compute row-major index for a matrix element.
 * @i: Row index.
 * @j: Column index.
 * @cols: Number of columns in the matrix.
 */
#define R_MATRIX_IDX(i, j, cols) ((i) * (cols) + (j))
/**
 * R_MATRIX_SIZE() - Compute number of elements in a matrix.
 * @matrix: Matrix pointer.
 */
#define R_MATRIX_SIZE(matrix) (((matrix)->rows) * ((matrix)->cols))

/**
 * EPSILON - Small constant for numerical stability.
 */
#define EPSILON 1e-7f

#ifndef M_PI
/**
 * M_PI - Pi constant if not provided by the system headers.
 */
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT2
/**
 * M_SQRT2 - Square root of 2 if not provided by the system headers.
 */
#define M_SQRT2 1.41421356237309504880
#endif

#endif
