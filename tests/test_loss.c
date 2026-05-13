#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <rc/rc.h>

#define DEFAULT_ABS_TOL 1e-6f
#define DEFAULT_REL_TOL 1e-6f

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
        fprintf(stderr,
                "ASSERT_FLOAT_CLOSE failed: %s (actual=%.8f expected=%.8f diff=%.8f tol=%.8f)\n",
                msg, actual, expected, diff, tol);
    }

    assert(diff <= tol);
}

static void assert_isfinitef(float value, const char *msg)
{
    if (!isfinite(value))
    {
        fprintf(stderr, "ASSERT_FINITE failed: %s (value=%.8f)\n", msg, value);
    }
    assert(isfinite(value));
}

static void assert_isnanf(float value, const char *msg)
{
    if (!isnan(value))
    {
        fprintf(stderr, "ASSERT_NAN failed: %s (value=%.8f)\n", msg, value);
    }
    assert(isnan(value));
}

static void test_cross_entropy_basic(void)
{
    const float pred_vals[] = {0.7f, 0.2f, 0.1f, 0.1f, 0.8f, 0.1f};
    const float real_vals[] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};

    RMatrix *pred = make_matrix(2, 3, pred_vals);
    RMatrix *real = make_matrix(2, 3, real_vals);

    const float expected = -(logf(0.7f + EPSILON) + logf(0.8f + EPSILON)) / 2.0f;
    const float actual = r_cross_entropy(pred, real);

    assert_float_close(actual, expected, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "cross_entropy basic");

    r_free_matrix(pred);
    r_free_matrix(real);
}

static void test_cross_entropy_matches_cat_cross_entropy(void)
{
    const float pred_vals[] = {0.4f, 0.6f, 0.3f, 0.7f};
    const float real_vals[] = {0.0f, 1.0f, 1.0f, 0.0f};

    RMatrix *pred = make_matrix(2, 2, pred_vals);
    RMatrix *real = make_matrix(2, 2, real_vals);

    const float ce = r_cross_entropy(pred, real);
    const float cat = r_cat_cross_entropy(pred, real);

    assert_float_close(cat, ce, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "cat_cross_entropy matches cross_entropy");

    r_free_matrix(pred);
    r_free_matrix(real);
}

static void test_cross_entropy_bounds_finite(void)
{
    const float pred_vals[] = {0.0f, 1.0f, 0.0f};
    const float real_vals[] = {1.0f, 0.0f, 0.0f};

    RMatrix *pred = make_matrix(1, 3, pred_vals);
    RMatrix *real = make_matrix(1, 3, real_vals);

    const float loss = r_cross_entropy(pred, real);

    assert_isfinitef(loss, "cross_entropy finite with p=0/1");
    assert(loss >= 0.0f && "cross_entropy should be non-negative");

    r_free_matrix(pred);
    r_free_matrix(real);
}

static void test_bin_cross_entropy_basic(void)
{
    const float pred_vals[] = {0.9f, 0.1f, 0.2f, 0.8f};
    const float real_vals[] = {1.0f, 0.0f, 0.0f, 1.0f};

    RMatrix *pred = make_matrix(2, 2, pred_vals);
    RMatrix *real = make_matrix(2, 2, real_vals);

    const float expected =
        -(logf(0.9f + EPSILON) + logf(1.0f - 0.1f + EPSILON) +
          logf(1.0f - 0.2f + EPSILON) + logf(0.8f + EPSILON)) / 4.0f;

    const float actual = r_bin_cross_entropy(pred, real);

    assert_float_close(actual, expected, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "bin_cross_entropy basic");

    r_free_matrix(pred);
    r_free_matrix(real);
}

static void test_bin_cross_entropy_bounds_finite(void)
{
    const float pred_vals[] = {0.0f, 1.0f};
    const float real_vals[] = {1.0f, 0.0f};

    RMatrix *pred = make_matrix(1, 2, pred_vals);
    RMatrix *real = make_matrix(1, 2, real_vals);

    const float loss = r_bin_cross_entropy(pred, real);

    assert_isfinitef(loss, "bin_cross_entropy finite with p=0/1");
    assert(loss >= 0.0f && "bin_cross_entropy should be non-negative");

    r_free_matrix(pred);
    r_free_matrix(real);
}

static void test_mse_loss_basic(void)
{
    const float pred_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float real_vals[] = {1.0f, 1.0f, 2.0f, 2.0f};

    RMatrix *pred = make_matrix(2, 2, pred_vals);
    RMatrix *real = make_matrix(2, 2, real_vals);

    const float expected = 1.5f;
    const float actual = r_mse_loss(pred, real);

    assert_float_close(actual, expected, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "mse_loss basic");

    r_free_matrix(pred);
    r_free_matrix(real);
}

static void test_mae_loss_basic(void)
{
    const float pred_vals[] = {2.0f, 1.0f, 3.0f};
    const float real_vals[] = {1.0f, 2.0f, 3.0f};

    RMatrix *pred = make_matrix(1, 3, pred_vals);
    RMatrix *real = make_matrix(1, 3, real_vals);

    const float expected = 2.0f / 3.0f;
    const float actual = r_mae_loss(pred, real);

    assert_float_close(actual, expected, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "mae_loss basic");

    r_free_matrix(pred);
    r_free_matrix(real);
}

static void test_bin_focal_gamma_zero_matches_scaled_bce(void)
{
    const float pred_vals[] = {0.9f, 0.1f, 0.2f, 0.8f};
    const float real_vals[] = {1.0f, 0.0f, 0.0f, 1.0f};

    RMatrix *pred = make_matrix(2, 2, pred_vals);
    RMatrix *real = make_matrix(2, 2, real_vals);

    const float bce = r_bin_cross_entropy(pred, real);
    const float focal = r_bin_focal_loss(pred, real, 0.0f, 0.5f);

    assert_float_close(focal, 0.5f * bce, DEFAULT_ABS_TOL, DEFAULT_REL_TOL,
                       "bin_focal gamma=0 alpha=0.5 equals 0.5*bce");

    r_free_matrix(pred);
    r_free_matrix(real);
}

static void test_cat_focal_gamma_zero_matches_cat_ce(void)
{
    const float pred_vals[] = {0.6f, 0.3f, 0.1f, 0.2f, 0.7f, 0.1f};
    const float real_vals[] = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f};

    RMatrix *pred = make_matrix(2, 3, pred_vals);
    RMatrix *real = make_matrix(2, 3, real_vals);

    const float ce = r_cat_cross_entropy(pred, real);
    const float focal = r_cat_focal_loss(pred, real, 0.0f);

    assert_float_close(focal, ce, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "cat_focal gamma=0 equals cat_ce");

    r_free_matrix(pred);
    r_free_matrix(real);
}

static void test_cat_focal_all_zero_targets_is_zero(void)
{
    const float pred_vals[] = {0.5f, 0.3f, 0.2f, 0.1f, 0.2f, 0.7f};
    const float real_vals[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    RMatrix *pred = make_matrix(2, 3, pred_vals);
    RMatrix *real = make_matrix(2, 3, real_vals);

    const float loss = r_cat_focal_loss(pred, real, 2.0f);

    assert_float_close(loss, 0.0f, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "cat_focal all zero targets");

    r_free_matrix(pred);
    r_free_matrix(real);
}

static void test_bin_focal_bounds_finite(void)
{
    const float pred_vals[] = {0.0f, 1.0f};
    const float real_vals[] = {1.0f, 0.0f};

    RMatrix *pred = make_matrix(1, 2, pred_vals);
    RMatrix *real = make_matrix(1, 2, real_vals);

    const float loss = r_bin_focal_loss(pred, real, 2.0f, 0.25f);

    assert_isfinitef(loss, "bin_focal finite with p=0/1");
    assert(loss >= 0.0f && "bin_focal should be non-negative");

    r_free_matrix(pred);
    r_free_matrix(real);
}

static void test_nan_propagation_mse_mae(void)
{
    const float pred_vals[] = {NAN, 1.0f};
    const float real_vals[] = {0.0f, 0.0f};

    RMatrix *pred = make_matrix(1, 2, pred_vals);
    RMatrix *real = make_matrix(1, 2, real_vals);

    const float mse = r_mse_loss(pred, real);
    const float mae = r_mae_loss(pred, real);

    assert_isnanf(mse, "mse_loss should propagate NaN");
    assert_isnanf(mae, "mae_loss should propagate NaN");

    r_free_matrix(pred);
    r_free_matrix(real);
}

static void test_zero_sized_matrices_return_nan(void)
{
    RMatrix *pred = r_create_matrix(0, 0);
    RMatrix *real = r_create_matrix(0, 0);

    assert_isnanf(r_cross_entropy(pred, real), "cross_entropy on empty");
    assert_isnanf(r_cat_cross_entropy(pred, real), "cat_cross_entropy on empty");
    assert_isnanf(r_bin_cross_entropy(pred, real), "bin_cross_entropy on empty");
    assert_isnanf(r_mse_loss(pred, real), "mse_loss on empty");
    assert_isnanf(r_mae_loss(pred, real), "mae_loss on empty");
    assert_isnanf(r_bin_focal_loss(pred, real, 2.0f, 0.25f), "bin_focal_loss on empty");
    assert_isnanf(r_cat_focal_loss(pred, real, 2.0f), "cat_focal_loss on empty");

    r_free_matrix(pred);
    r_free_matrix(real);
}

int main(void)
{
    test_cross_entropy_basic();
    test_cross_entropy_matches_cat_cross_entropy();
    test_cross_entropy_bounds_finite();
    test_bin_cross_entropy_basic();
    test_bin_cross_entropy_bounds_finite();
    test_mse_loss_basic();
    test_mae_loss_basic();
    test_bin_focal_gamma_zero_matches_scaled_bce();
    test_cat_focal_gamma_zero_matches_cat_ce();
    test_cat_focal_all_zero_targets_is_zero();
    test_bin_focal_bounds_finite();
    test_nan_propagation_mse_mae();
    test_zero_sized_matrices_return_nan();

    printf("All loss tests passed.\n");
    return 0;
}