#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <rc/rc.h>

#define DEFAULT_ABS_TOL 1e-5f
#define DEFAULT_REL_TOL 1e-5f
#define NUM_GRAD_ABS_TOL 1e-3f
#define NUM_GRAD_REL_TOL 1e-2f

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

static RMatrix *clone_matrix(const RMatrix *src)
{
    return make_matrix(src->rows, src->cols, src->data);
}

static void assert_float_close(float actual, float expected, float abs_tol, float rel_tol, const char *msg)
{
    const float diff = fabsf(actual - expected);
    const float scale = fmaxf(fabsf(expected), 1.0f);
    const float tol = fmaxf(abs_tol, rel_tol * scale);

    if (diff > tol)
    {
        fprintf(stderr, "ASSERT_FLOAT_CLOSE failed: %s (actual=%.8f expected=%.8f diff=%.8f tol=%.8f)\n", msg, actual,
                expected, diff, tol);
    }

    assert(diff <= tol);
}

static float numeric_bin_focal_grad(const RMatrix *pred, const RMatrix *real, float gamma, float alpha, size_t idx)
{
    const float delta = 1e-4f;

    RMatrix *pred_plus = clone_matrix(pred);
    RMatrix *pred_minus = clone_matrix(pred);

    pred_plus->data[idx] += delta;
    pred_minus->data[idx] -= delta;

    const float loss_plus = r_bin_focal_loss(pred_plus, real, gamma, alpha);
    const float loss_minus = r_bin_focal_loss(pred_minus, real, gamma, alpha);

    r_free_matrix(pred_plus);
    r_free_matrix(pred_minus);

    return (loss_plus - loss_minus) / (2.0f * delta);
}

static float numeric_cat_focal_grad(const RMatrix *pred, const RMatrix *real, float gamma, size_t idx)
{
    const float delta = 1e-4f;

    RMatrix *pred_plus = clone_matrix(pred);
    RMatrix *pred_minus = clone_matrix(pred);

    pred_plus->data[idx] += delta;
    pred_minus->data[idx] -= delta;

    const float loss_plus = r_cat_focal_loss(pred_plus, real, gamma);
    const float loss_minus = r_cat_focal_loss(pred_minus, real, gamma);

    r_free_matrix(pred_plus);
    r_free_matrix(pred_minus);

    return (loss_plus - loss_minus) / (2.0f * delta);
}

static void test_cross_entropy_grad_basic(void)
{
    const float pred_vals[] = {0.7f, 0.2f, 0.1f, 0.1f, 0.8f, 0.1f};
    const float real_vals[] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};

    RMatrix *pred = make_matrix(2, 3, pred_vals);
    RMatrix *real = make_matrix(2, 3, real_vals);
    RMatrix *grad = r_create_matrix(2, 3);

    r_cross_entropy_grad(pred, real, grad);

    for (size_t i = 0; i < pred->rows; i++)
    {
        for (size_t j = 0; j < pred->cols; j++)
        {
            const size_t idx = RMatrixIDX(i, j, pred->cols);
            const float p = pred->data[idx];
            const float y = real->data[idx];
            const float expected = -y / (2.0f * (p + EPSILON));

            char msg[96];
            snprintf(msg, sizeof(msg), "cross_entropy_grad idx=%zu", idx);
            assert_float_close(grad->data[idx], expected, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, msg);
        }
    }

    r_free_matrix(pred);
    r_free_matrix(real);
    r_free_matrix(grad);
}

static void test_cross_entropy_grad_zero_prediction(void)
{
    const float pred_vals[] = {0.0f};
    const float real_vals[] = {1.0f};

    RMatrix *pred = make_matrix(1, 1, pred_vals);
    RMatrix *real = make_matrix(1, 1, real_vals);
    RMatrix *grad = r_create_matrix(1, 1);

    r_cross_entropy_grad(pred, real, grad);

    const float expected = -1.0f / (1.0f * (0.0f + EPSILON));
    assert_float_close(grad->data[0], expected, 1e-2f, 1e-5f, "cross_entropy_grad handles p=0 via EPSILON");

    r_free_matrix(pred);
    r_free_matrix(real);
    r_free_matrix(grad);
}

static void test_cat_cross_entropy_grad_matches_cross_entropy(void)
{
    const float pred_vals[] = {0.6f, 0.3f, 0.1f, 0.2f, 0.7f, 0.1f};
    const float real_vals[] = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f};

    RMatrix *pred = make_matrix(2, 3, pred_vals);
    RMatrix *real = make_matrix(2, 3, real_vals);
    RMatrix *grad_ce = r_create_matrix(2, 3);
    RMatrix *grad_cat = r_create_matrix(2, 3);

    r_cross_entropy_grad(pred, real, grad_ce);
    r_cat_cross_entropy_grad(pred, real, grad_cat);

    for (size_t i = 0; i < pred->rows; i++)
    {
        for (size_t j = 0; j < pred->cols; j++)
        {
            const size_t idx = RMatrixIDX(i, j, pred->cols);
            char msg[96];
            snprintf(msg, sizeof(msg), "cat_vs_cross_entropy_grad idx=%zu", idx);
            assert_float_close(grad_cat->data[idx], grad_ce->data[idx], DEFAULT_ABS_TOL, DEFAULT_REL_TOL, msg);
        }
    }

    r_free_matrix(pred);
    r_free_matrix(real);
    r_free_matrix(grad_ce);
    r_free_matrix(grad_cat);
}

static void test_bin_cross_entropy_grad_basic(void)
{
    const float pred_vals[] = {0.9f, 0.1f, 0.2f, 0.8f};
    const float real_vals[] = {1.0f, 0.0f, 0.0f, 1.0f};

    RMatrix *pred = make_matrix(2, 2, pred_vals);
    RMatrix *real = make_matrix(2, 2, real_vals);
    RMatrix *grad = r_create_matrix(2, 2);

    r_bin_cross_entropy_grad(pred, real, grad);

    const float norm = 4.0f;
    for (size_t i = 0; i < pred->rows; i++)
    {
        for (size_t j = 0; j < pred->cols; j++)
        {
            const size_t idx = RMatrixIDX(i, j, pred->cols);
            const float p = pred->data[idx];
            const float y = real->data[idx];

            const float num_pos = -y / (p + EPSILON);
            const float num_neg = (1.0f - y) / (1.0f - p + EPSILON);
            const float expected = (num_pos + num_neg) / norm;

            char msg[96];
            snprintf(msg, sizeof(msg), "bin_cross_entropy_grad idx=%zu", idx);
            assert_float_close(grad->data[idx], expected, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, msg);
        }
    }

    r_free_matrix(pred);
    r_free_matrix(real);
    r_free_matrix(grad);
}

static void test_mse_loss_grad_basic(void)
{
    const float pred_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float real_vals[] = {1.0f, 1.0f, 2.0f, 2.0f};

    RMatrix *pred = make_matrix(2, 2, pred_vals);
    RMatrix *real = make_matrix(2, 2, real_vals);
    RMatrix *grad = r_create_matrix(2, 2);

    r_mse_loss_grad(pred, real, grad);

    const float expected_vals[] = {0.0f, 0.5f, 0.5f, 1.0f};
    for (size_t i = 0; i < 4; i++)
    {
        char msg[96];
        snprintf(msg, sizeof(msg), "mse_grad idx=%zu", i);
        assert_float_close(grad->data[i], expected_vals[i], DEFAULT_ABS_TOL, DEFAULT_REL_TOL, msg);
    }

    r_free_matrix(pred);
    r_free_matrix(real);
    r_free_matrix(grad);
}

static void test_mae_loss_grad_basic(void)
{
    const float pred_vals[] = {2.0f, 1.0f, 3.0f};
    const float real_vals[] = {1.0f, 2.0f, 3.0f};

    RMatrix *pred = make_matrix(1, 3, pred_vals);
    RMatrix *real = make_matrix(1, 3, real_vals);
    RMatrix *grad = r_create_matrix(1, 3);

    r_mae_loss_grad(pred, real, grad);

    const float expected_vals[] = {1.0f / 3.0f, -1.0f / 3.0f, 0.0f};
    for (size_t i = 0; i < 3; i++)
    {
        char msg[96];
        snprintf(msg, sizeof(msg), "mae_grad idx=%zu", i);
        assert_float_close(grad->data[i], expected_vals[i], DEFAULT_ABS_TOL, DEFAULT_REL_TOL, msg);
    }

    r_free_matrix(pred);
    r_free_matrix(real);
    r_free_matrix(grad);
}

static void test_bin_focal_loss_grad_numeric(void)
{
    const float pred_vals[] = {0.2f, 0.7f};
    const float real_vals[] = {1.0f, 0.0f};
    const float gamma = 2.0f;
    const float alpha = 0.25f;

    RMatrix *pred = make_matrix(1, 2, pred_vals);
    RMatrix *real = make_matrix(1, 2, real_vals);
    RMatrix *grad = r_create_matrix(1, 2);

    r_bin_focal_loss_grad(pred, real, gamma, alpha, grad);

    for (size_t i = 0; i < 2; i++)
    {
        const float expected = numeric_bin_focal_grad(pred, real, gamma, alpha, i);
        char msg[96];
        snprintf(msg, sizeof(msg), "bin_focal_grad_numeric idx=%zu", i);
        assert_float_close(grad->data[i], expected, NUM_GRAD_ABS_TOL, NUM_GRAD_REL_TOL, msg);
    }

    r_free_matrix(pred);
    r_free_matrix(real);
    r_free_matrix(grad);
}

static void test_cat_focal_loss_grad_numeric(void)
{
    const float pred_vals[] = {0.7f, 0.2f, 0.1f};
    const float real_vals[] = {1.0f, 0.0f, 0.0f};
    const float gamma = 2.0f;

    RMatrix *pred = make_matrix(1, 3, pred_vals);
    RMatrix *real = make_matrix(1, 3, real_vals);
    RMatrix *grad = r_create_matrix(1, 3);

    r_cat_focal_loss_grad(pred, real, gamma, grad);

    for (size_t i = 0; i < 3; i++)
    {
        const float expected = numeric_cat_focal_grad(pred, real, gamma, i);
        char msg[96];
        snprintf(msg, sizeof(msg), "cat_focal_grad_numeric idx=%zu", i);
        assert_float_close(grad->data[i], expected, NUM_GRAD_ABS_TOL, NUM_GRAD_REL_TOL, msg);
    }

    r_free_matrix(pred);
    r_free_matrix(real);
    r_free_matrix(grad);
}

static void test_nan_propagation_mse_grad(void)
{
    const float pred_vals[] = {NAN};
    const float real_vals[] = {0.0f};

    RMatrix *pred = make_matrix(1, 1, pred_vals);
    RMatrix *real = make_matrix(1, 1, real_vals);
    RMatrix *grad = r_create_matrix(1, 1);

    r_mse_loss_grad(pred, real, grad);

    assert(isnan(grad->data[0]) && "mse_grad should propagate NaN inputs");

    r_free_matrix(pred);
    r_free_matrix(real);
    r_free_matrix(grad);
}

static void test_zero_sized_matrix_no_crash(void)
{
    RMatrix *pred = r_create_matrix(0, 0);
    RMatrix *real = r_create_matrix(0, 0);
    RMatrix *grad = r_create_matrix(0, 0);

    assert(pred->rows == 0 && pred->cols == 0 && "zero-sized matrix constructed");

    r_cross_entropy_grad(pred, real, grad);
    r_cat_cross_entropy_grad(pred, real, grad);

    r_free_matrix(pred);
    r_free_matrix(real);
    r_free_matrix(grad);
}

int main(void)
{
    test_cross_entropy_grad_basic();
    test_cross_entropy_grad_zero_prediction();
    test_cat_cross_entropy_grad_matches_cross_entropy();
    test_bin_cross_entropy_grad_basic();
    test_mse_loss_grad_basic();
    test_mae_loss_grad_basic();
    test_bin_focal_loss_grad_numeric();
    test_cat_focal_loss_grad_numeric();
    test_nan_propagation_mse_grad();
    test_zero_sized_matrix_no_crash();

    printf("All loss gradient tests passed.\n");
    return 0;
}
