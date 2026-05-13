#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <rc/rc.h>

#define DEFAULT_ABS_TOL 1e-5f
#define DEFAULT_REL_TOL 1e-5f

static RMatrix *make_matrix(size_t rows, size_t cols, const float *values)
{
    RMatrix *matrix = r_create_matrix(rows, cols);
    const size_t count = rows * cols;
    if (count > 0 && values != NULL)
    {
        memcpy(matrix->data, values, sizeof(float) * count);
    }
    return matrix;
}

static void assert_float_close(float actual, float expected, float abs_tol, float rel_tol, const char *msg)
{
    const float diff = fabsf(actual - expected);
    const float scale = fmaxf(fabsf(expected), 1.0f);
    const float tol = fmaxf(abs_tol, rel_tol * scale);

    if (diff > tol)
    {
        fprintf(stderr, "ASSERT_FLOAT_CLOSE failed: %s (actual=%.8f expected=%.8f diff=%.8f tol=%.8f)\n", msg,
                actual, expected, diff, tol);
    }

    assert(diff <= tol);
}

static void assert_true(int condition, const char *msg)
{
    if (!condition)
    {
        fprintf(stderr, "ASSERT_TRUE failed: %s\n", msg);
    }
    assert(condition);
}

static float row_sum(const RMatrix *matrix, size_t row)
{
    float sum = 0.0f;
    for (size_t j = 0; j < matrix->cols; j++)
    {
        sum += matrix->data[RMatrixIDX(row, j, matrix->cols)];
    }
    return sum;
}

static void test_softmax_basic_values(void)
{
    const float input_vals[] = {1.0f, 2.0f, 3.0f};
    const float expected_vals[] = {0.09003057f, 0.24472847f, 0.66524096f};

    RMatrix *input = make_matrix(1, 3, input_vals);
    RMatrix *output = r_activation_softmax(input);

    for (size_t j = 0; j < 3; j++)
    {
        char msg[96];
        snprintf(msg, sizeof(msg), "softmax_basic col=%zu", j);
        assert_float_close(output->data[j], expected_vals[j], DEFAULT_ABS_TOL, DEFAULT_REL_TOL, msg);
    }

    r_free_matrix(input);
    r_free_matrix(output);
}

static void test_softmax_row_sums_and_bounds(void)
{
    const float input_vals[] = {1.0f, 2.0f, -3.0f, 0.5f, 0.1f, 0.2f, 0.3f, 0.4f};

    RMatrix *input = make_matrix(2, 4, input_vals);
    RMatrix *output = r_activation_softmax(input);

    for (size_t i = 0; i < output->rows; i++)
    {
        char msg[96];
        snprintf(msg, sizeof(msg), "softmax_row_sum row=%zu", i);
        assert_float_close(row_sum(output, i), 1.0f, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, msg);

        for (size_t j = 0; j < output->cols; j++)
        {
            const float value = output->data[RMatrixIDX(i, j, output->cols)];
            snprintf(msg, sizeof(msg), "softmax_bounds row=%zu col=%zu", i, j);
            assert_true(isfinite(value) && value >= 0.0f && value <= 1.0f, msg);
        }
    }

    r_free_matrix(input);
    r_free_matrix(output);
}

static void test_softmax_shift_invariance(void)
{
    const float input_vals[] = {1.0f, -1.0f, 0.5f, 2.5f};
    const float shifted_vals[] = {101.0f, 99.0f, 100.5f, 102.5f};

    RMatrix *input = make_matrix(1, 4, input_vals);
    RMatrix *shifted = make_matrix(1, 4, shifted_vals);
    RMatrix *output = r_activation_softmax(input);
    RMatrix *output_shifted = r_activation_softmax(shifted);

    for (size_t j = 0; j < 4; j++)
    {
        char msg[96];
        snprintf(msg, sizeof(msg), "softmax_shift_invariance col=%zu", j);
        assert_float_close(output_shifted->data[j], output->data[j], DEFAULT_ABS_TOL, DEFAULT_REL_TOL, msg);
    }

    r_free_matrix(input);
    r_free_matrix(shifted);
    r_free_matrix(output);
    r_free_matrix(output_shifted);
}

static void test_softmax_uniform_for_equal_inputs(void)
{
    const float input_vals[] = {2.0f, 2.0f, 2.0f, 2.0f};
    const float expected = 0.25f;

    RMatrix *input = make_matrix(1, 4, input_vals);
    RMatrix *output = r_activation_softmax(input);

    for (size_t j = 0; j < 4; j++)
    {
        char msg[96];
        snprintf(msg, sizeof(msg), "softmax_uniform col=%zu", j);
        assert_float_close(output->data[j], expected, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, msg);
    }

    r_free_matrix(input);
    r_free_matrix(output);
}

static void test_softmax_single_element(void)
{
    const float input_vals[] = {5.0f, -2.0f, 0.0f};

    RMatrix *input = make_matrix(3, 1, input_vals);
    RMatrix *output = r_activation_softmax(input);

    for (size_t i = 0; i < output->rows; i++)
    {
        char msg[96];
        snprintf(msg, sizeof(msg), "softmax_single_element row=%zu", i);
        assert_float_close(output->data[RMatrixIDX(i, 0, output->cols)], 1.0f, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, msg);
    }

    r_free_matrix(input);
    r_free_matrix(output);
}

static void test_softmax_large_values_stability(void)
{
    const float input_vals[] = {1000.0f, 1001.0f, 1002.0f};
    const float expected_vals[] = {0.09003057f, 0.24472847f, 0.66524096f};

    RMatrix *input = make_matrix(1, 3, input_vals);
    RMatrix *output = r_activation_softmax(input);

    for (size_t j = 0; j < 3; j++)
    {
        char msg[96];
        snprintf(msg, sizeof(msg), "softmax_large_values col=%zu", j);
        assert_true(isfinite(output->data[j]), msg);
        assert_float_close(output->data[j], expected_vals[j], DEFAULT_ABS_TOL, DEFAULT_REL_TOL, msg);
    }

    assert_true(output->data[2] > output->data[1] && output->data[1] > output->data[0],
                "softmax_large_values preserves ordering");
    assert_float_close(row_sum(output, 0), 1.0f, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "softmax_large_values sum");

    r_free_matrix(input);
    r_free_matrix(output);
}

static void test_softmax_zero_rows(void)
{
    RMatrix *input = r_create_matrix(0, 3);
    RMatrix *output = r_activation_softmax(input);

    assert_true(output->rows == 0, "softmax_zero_rows preserves row count");
    assert_true(output->cols == 3, "softmax_zero_rows preserves col count");

    r_free_matrix(input);
    r_free_matrix(output);
}

int main(void)
{
    test_softmax_basic_values();
    test_softmax_row_sums_and_bounds();
    test_softmax_shift_invariance();
    test_softmax_uniform_for_equal_inputs();
    test_softmax_single_element();
    test_softmax_large_values_stability();
    test_softmax_zero_rows();

    return 0;
}
