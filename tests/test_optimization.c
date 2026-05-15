#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <rc/r_optimization.h>
#include <rc/rc.h>

#define DEFAULT_ABS_TOL 1e-5f
#define DEFAULT_REL_TOL 1e-5f

static RMatrix *make_matrix(size_t rows, size_t cols, const float *values)
{
    RMatrix *m = r_create_matrix(rows, cols);
    const size_t n = rows * cols;
    if (n > 0 && values != NULL)
        memcpy(m->data, values, sizeof(float) * n);
    return m;
}

static RMatrix *make_zeros(size_t rows, size_t cols)
{
    RMatrix *m = r_create_matrix(rows, cols);
    memset(m->data, 0, sizeof(float) * rows * cols);
    return m;
}

static void assert_float_close(float actual, float expected, float abs_tol, float rel_tol, const char *msg)
{
    const float diff = fabsf(actual - expected);
    const float scale = fmaxf(fabsf(expected), 1.0f);
    const float tol = fmaxf(abs_tol, rel_tol * scale);

    if (diff > tol)
    {
        fprintf(stderr,
                "ASSERT_FLOAT_CLOSE failed: %s "
                "(actual=%.8f expected=%.8f diff=%.8f tol=%.8f)\n",
                msg, actual, expected, diff, tol);
    }
    assert(diff <= tol);
}

static void assert_matrix_close(const RMatrix *m, const float *expected, size_t n, const char *label)
{
    for (size_t i = 0; i < n; i++)
    {
        char msg[128];
        snprintf(msg, sizeof(msg), "%s[%zu]", label, i);
        assert_float_close(m->data[i], expected[i], DEFAULT_ABS_TOL, DEFAULT_REL_TOL, msg);
    }
}

static void test_sgd_basic(void)
{
    const float theta_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float grad_vals[] = {0.1f, 0.2f, 0.3f, 0.4f};
    const float eta = 0.5f;

    RMatrix *theta = make_matrix(2, 2, theta_vals);
    RMatrix *grad = make_matrix(2, 2, grad_vals);

    r_optimization_sgd(theta, grad, eta);

    const float expected[] = {
        1.0f - 0.5f * 0.1f,
        2.0f - 0.5f * 0.2f,
        3.0f - 0.5f * 0.3f,
        4.0f - 0.5f * 0.4f,
    };
    assert_matrix_close(theta, expected, 4, "sgd_basic");

    r_free_matrix(theta);
    r_free_matrix(grad);
}

static void test_sgd_negative_grad(void)
{
    const float theta_vals[] = {1.0f};
    const float grad_vals[] = {-1.0f};

    RMatrix *theta = make_matrix(1, 1, theta_vals);
    RMatrix *grad = make_matrix(1, 1, grad_vals);

    r_optimization_sgd(theta, grad, 0.1f);

    assert_float_close(theta->data[0], 1.1f, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "sgd_negative_grad");

    r_free_matrix(theta);
    r_free_matrix(grad);
}

static void test_sgd_zero_grad(void)
{
    const float theta_vals[] = {3.14f, -2.71f};
    const float grad_vals[] = {0.0f, 0.0f};

    RMatrix *theta = make_matrix(1, 2, theta_vals);
    RMatrix *grad = make_matrix(1, 2, grad_vals);

    r_optimization_sgd(theta, grad, 1.0f);

    assert_float_close(theta->data[0], 3.14f, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "sgd_zero_grad[0]");
    assert_float_close(theta->data[1], -2.71f, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "sgd_zero_grad[1]");

    r_free_matrix(theta);
    r_free_matrix(grad);
}

static void test_sgd_zero_eta(void)
{
    const float theta_vals[] = {5.0f, -3.0f};
    const float grad_vals[] = {1.0f, 1.0f};

    RMatrix *theta = make_matrix(1, 2, theta_vals);
    RMatrix *grad = make_matrix(1, 2, grad_vals);

    r_optimization_sgd(theta, grad, 0.0f);

    assert_float_close(theta->data[0], 5.0f, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "sgd_zero_eta[0]");
    assert_float_close(theta->data[1], -3.0f, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "sgd_zero_eta[1]");

    r_free_matrix(theta);
    r_free_matrix(grad);
}

static void test_sgd_nan_grad_propagates(void)
{
    const float theta_vals[] = {1.0f};
    const float grad_vals[] = {NAN};

    RMatrix *theta = make_matrix(1, 1, theta_vals);
    RMatrix *grad = make_matrix(1, 1, grad_vals);

    r_optimization_sgd(theta, grad, 0.1f);

    assert(isnan(theta->data[0]) && "sgd: NaN in gradient should propagate to theta");

    r_free_matrix(theta);
    r_free_matrix(grad);
}

static void test_sgd_empty_matrix(void)
{
    RMatrix *theta = r_create_matrix(0, 0);
    RMatrix *grad = r_create_matrix(0, 0);

    r_optimization_sgd(theta, grad, 0.01f);

    r_free_matrix(theta);
    r_free_matrix(grad);
}

static void test_sgdm_basic_zero_velocity(void)
{
    const float theta_vals[] = {1.0f, 2.0f};
    const float grad_vals[] = {0.5f, -0.5f};
    const float eta = 0.1f;
    const float beta = 0.9f;

    RMatrix *theta = make_matrix(1, 2, theta_vals);
    RMatrix *grad = make_matrix(1, 2, grad_vals);
    RMatrix *velocity = make_zeros(1, 2);

    r_optimization_sgdm(theta, grad, velocity, eta, beta);

    const float v0 = (1.0f - beta) * 0.5f;
    const float v1 = (1.0f - beta) * (-0.5f);

    assert_float_close(velocity->data[0], v0, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "sgdm_velocity[0]");
    assert_float_close(velocity->data[1], v1, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "sgdm_velocity[1]");
    assert_float_close(theta->data[0], 1.0f - eta * v0, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "sgdm_theta[0]");
    assert_float_close(theta->data[1], 2.0f - eta * v1, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "sgdm_theta[1]");

    r_free_matrix(theta);
    r_free_matrix(grad);
    r_free_matrix(velocity);
}

static void test_sgdm_beta_zero_equals_sgd(void)
{
    const float theta_vals[] = {3.0f, 1.0f};
    const float grad_vals[] = {0.2f, 0.4f};
    const float eta = 0.5f;

    RMatrix *theta_sgdm = make_matrix(1, 2, theta_vals);
    RMatrix *theta_sgd = make_matrix(1, 2, theta_vals);
    RMatrix *grad = make_matrix(1, 2, grad_vals);
    RMatrix *velocity = make_zeros(1, 2);

    r_optimization_sgdm(theta_sgdm, grad, velocity, eta, 0.0f);
    r_optimization_sgd(theta_sgd, grad, eta);

    for (size_t i = 0; i < 2; i++)
    {
        char msg[64];
        snprintf(msg, sizeof(msg), "sgdm_beta0_vs_sgd[%zu]", i);
        assert_float_close(theta_sgdm->data[i], theta_sgd->data[i], DEFAULT_ABS_TOL, DEFAULT_REL_TOL, msg);
    }

    r_free_matrix(theta_sgdm);
    r_free_matrix(theta_sgd);
    r_free_matrix(grad);
    r_free_matrix(velocity);
}

static void test_sgdm_beta_one_ignores_grad(void)
{
    const float theta_vals[] = {2.0f};
    const float grad_vals[] = {10.0f};

    RMatrix *theta = make_matrix(1, 1, theta_vals);
    RMatrix *grad = make_matrix(1, 1, grad_vals);
    RMatrix *velocity = make_zeros(1, 1);

    r_optimization_sgdm(theta, grad, velocity, 0.1f, 1.0f);

    assert_float_close(theta->data[0], 2.0f, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "sgdm_beta1_no_change");

    r_free_matrix(theta);
    r_free_matrix(grad);
    r_free_matrix(velocity);
}

static void test_sgdm_velocity_accumulates(void)
{
    const float theta_v[] = {0.0f};
    const float grad_v[] = {1.0f};
    const float eta = 0.1f;
    const float beta = 0.9f;

    RMatrix *theta = make_matrix(1, 1, theta_v);
    RMatrix *grad = make_matrix(1, 1, grad_v);
    RMatrix *velocity = make_zeros(1, 1);

    r_optimization_sgdm(theta, grad, velocity, eta, beta);
    const float v1 = (1.0f - beta) * 1.0f;

    r_optimization_sgdm(theta, grad, velocity, eta, beta);
    const float v2 = beta * v1 + (1.0f - beta) * 1.0f;

    assert_float_close(velocity->data[0], v2, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "sgdm_velocity_step2");

    r_free_matrix(theta);
    r_free_matrix(grad);
    r_free_matrix(velocity);
}

static void test_sgdm_zero_eta(void)
{
    const float theta_v[] = {5.0f};
    const float grad_v[] = {2.0f};

    RMatrix *theta = make_matrix(1, 1, theta_v);
    RMatrix *grad = make_matrix(1, 1, grad_v);
    RMatrix *velocity = make_zeros(1, 1);

    r_optimization_sgdm(theta, grad, velocity, 0.0f, 0.9f);

    assert_float_close(theta->data[0], 5.0f, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "sgdm_zero_eta_no_change");

    r_free_matrix(theta);
    r_free_matrix(grad);
    r_free_matrix(velocity);
}

static void test_rmsprop_basic(void)
{
    const float theta_v[] = {2.0f};
    const float grad_v[] = {0.5f};
    const float eta = 0.01f, rho = 0.9f, eps = 1e-8f;

    RMatrix *theta = make_matrix(1, 1, theta_v);
    RMatrix *grad = make_matrix(1, 1, grad_v);
    RMatrix *cache = make_zeros(1, 1);

    r_optimization_rmsprop(theta, grad, cache, eta, rho, eps);

    const float g = 0.5f;
    const float c = (1.0f - rho) * g * g;
    const float step = (eta / (sqrtf(c) + eps)) * g;

    assert_float_close(cache->data[0], c, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "rmsprop_cache");
    assert_float_close(theta->data[0], 2.0f - step, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "rmsprop_theta");

    r_free_matrix(theta);
    r_free_matrix(grad);
    r_free_matrix(cache);
}

static void test_rmsprop_rho_zero(void)
{
    const float theta_v[] = {1.0f};
    const float grad_v[] = {2.0f};
    const float g = 2.0f, eta = 0.1f, eps = 1e-8f;

    RMatrix *theta = make_matrix(1, 1, theta_v);
    RMatrix *grad = make_matrix(1, 1, grad_v);
    RMatrix *cache = make_zeros(1, 1);

    r_optimization_rmsprop(theta, grad, cache, eta, 0.0f, eps);

    const float c = g * g;
    const float step = (eta / (sqrtf(c) + eps)) * g;

    assert_float_close(cache->data[0], c, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "rmsprop_rho0_cache");
    assert_float_close(theta->data[0], 1.0f - step, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "rmsprop_rho0_theta");

    r_free_matrix(theta);
    r_free_matrix(grad);
    r_free_matrix(cache);
}

static void test_rmsprop_zero_grad(void)
{
    const float theta_v[] = {3.0f};
    const float grad_v[] = {0.0f};
    const float cache_init_v[] = {0.5f};

    RMatrix *theta = make_matrix(1, 1, theta_v);
    RMatrix *grad = make_matrix(1, 1, grad_v);
    RMatrix *cache = make_matrix(1, 1, cache_init_v);

    r_optimization_rmsprop(theta, grad, cache, 0.01f, 0.9f, 1e-8f);

    assert_float_close(cache->data[0], 0.45f, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "rmsprop_zero_grad_cache");
    assert_float_close(theta->data[0], 3.0f, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "rmsprop_zero_grad_theta");

    r_free_matrix(theta);
    r_free_matrix(grad);
    r_free_matrix(cache);
}

static void test_rmsprop_eps_prevents_div_zero(void)
{
    const float theta_v[] = {1.0f};
    const float grad_v[] = {0.0f};
    const float eps = 1.0f;

    RMatrix *theta = make_matrix(1, 1, theta_v);
    RMatrix *grad = make_matrix(1, 1, grad_v);
    RMatrix *cache = make_zeros(1, 1);

    r_optimization_rmsprop(theta, grad, cache, 0.01f, 0.9f, eps);

    assert(!isnan(theta->data[0]) && "rmsprop: theta should not be NaN");
    assert(!isinf(theta->data[0]) && "rmsprop: theta should not be Inf");

    r_free_matrix(theta);
    r_free_matrix(grad);
    r_free_matrix(cache);
}

static void test_rmsprop_cache_accumulates(void)
{
    const float theta_v[] = {0.0f};
    const float grad_v[] = {1.0f};
    const float rho = 0.9f, eta = 0.01f, eps = 1e-8f;

    RMatrix *theta = make_matrix(1, 1, theta_v);
    RMatrix *grad = make_matrix(1, 1, grad_v);
    RMatrix *cache = make_zeros(1, 1);

    r_optimization_rmsprop(theta, grad, cache, eta, rho, eps);
    const float c1 = (1.0f - rho) * 1.0f;

    r_optimization_rmsprop(theta, grad, cache, eta, rho, eps);
    const float c2 = rho * c1 + (1.0f - rho) * 1.0f;

    assert_float_close(cache->data[0], c2, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "rmsprop_cache_step2");

    r_free_matrix(theta);
    r_free_matrix(grad);
    r_free_matrix(cache);
}

static void test_adam_basic_t1(void)
{
    const float g = 0.5f;
    const float theta_v[] = {1.0f};
    const float grad_v[] = {g};
    const float eta = 0.001f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;

    RMatrix *theta = make_matrix(1, 1, theta_v);
    RMatrix *grad = make_matrix(1, 1, grad_v);
    RMatrix *m = make_zeros(1, 1);
    RMatrix *v = make_zeros(1, 1);

    r_optimization_adam(theta, grad, m, v, eta, beta1, beta2, eps, 1);

    const float m_new = (1.0f - beta1) * g;
    const float v_new = (1.0f - beta2) * g * g;
    const float m_hat = m_new / (1.0f - powf(beta1, 1.0f));
    const float v_hat = v_new / (1.0f - powf(beta2, 1.0f));
    const float step = (eta / (sqrtf(v_hat) + eps)) * m_hat;

    assert_float_close(m->data[0], m_new, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "adam_m_t1");
    assert_float_close(v->data[0], v_new, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "adam_v_t1");
    assert_float_close(theta->data[0], 1.0f - step, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "adam_theta_t1");

    r_free_matrix(theta);
    r_free_matrix(grad);
    r_free_matrix(m);
    r_free_matrix(v);
}

static void test_adam_zero_grad(void)
{
    const float theta_v[] = {3.0f};
    const float grad_v[] = {0.0f};

    RMatrix *theta = make_matrix(1, 1, theta_v);
    RMatrix *grad = make_matrix(1, 1, grad_v);
    RMatrix *m = make_zeros(1, 1);
    RMatrix *v = make_zeros(1, 1);

    r_optimization_adam(theta, grad, m, v, 0.001f, 0.9f, 0.999f, 1e-8f, 1);

    assert_float_close(theta->data[0], 3.0f, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "adam_zero_grad_theta");
    assert_float_close(m->data[0], 0.0f, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "adam_zero_grad_m");

    r_free_matrix(theta);
    r_free_matrix(grad);
    r_free_matrix(m);
    r_free_matrix(v);
}

static void test_adam_zero_eta(void)
{
    const float theta_v[] = {7.0f};
    const float grad_v[] = {1.0f};

    RMatrix *theta = make_matrix(1, 1, theta_v);
    RMatrix *grad = make_matrix(1, 1, grad_v);
    RMatrix *m = make_zeros(1, 1);
    RMatrix *v = make_zeros(1, 1);

    r_optimization_adam(theta, grad, m, v, 0.0f, 0.9f, 0.999f, 1e-8f, 1);

    assert_float_close(theta->data[0], 7.0f, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "adam_zero_eta_theta");

    r_free_matrix(theta);
    r_free_matrix(grad);
    r_free_matrix(m);
    r_free_matrix(v);
}

static void test_adam_two_steps(void)
{
    const float theta_v[] = {0.0f};
    const float grad_v[] = {1.0f};
    const float eta = 0.001f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    const float g = 1.0f;

    RMatrix *theta = make_matrix(1, 1, theta_v);
    RMatrix *grad = make_matrix(1, 1, grad_v);
    RMatrix *m = make_zeros(1, 1);
    RMatrix *v = make_zeros(1, 1);

    r_optimization_adam(theta, grad, m, v, eta, beta1, beta2, eps, 1);
    const float m1 = (1.0f - beta1) * g;
    const float v1 = (1.0f - beta2) * g * g;

    r_optimization_adam(theta, grad, m, v, eta, beta1, beta2, eps, 2);
    const float m2 = beta1 * m1 + (1.0f - beta1) * g;
    const float v2 = beta2 * v1 + (1.0f - beta2) * g * g;

    assert_float_close(m->data[0], m2, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "adam_m_step2");
    assert_float_close(v->data[0], v2, DEFAULT_ABS_TOL, DEFAULT_REL_TOL, "adam_v_step2");

    r_free_matrix(theta);
    r_free_matrix(grad);
    r_free_matrix(m);
    r_free_matrix(v);
}

static void test_adam_large_t(void)
{
    const float theta_v[] = {1.0f};
    const float grad_v[] = {0.5f};

    RMatrix *theta = make_matrix(1, 1, theta_v);
    RMatrix *grad = make_matrix(1, 1, grad_v);
    RMatrix *m = make_zeros(1, 1);
    RMatrix *v = make_zeros(1, 1);

    r_optimization_adam(theta, grad, m, v, 0.001f, 0.9f, 0.999f, 1e-8f, 10000);

    assert(!isnan(theta->data[0]) && "adam_large_t: theta must not be NaN");
    assert(!isinf(theta->data[0]) && "adam_large_t: theta must not be Inf");

    r_free_matrix(theta);
    r_free_matrix(grad);
    r_free_matrix(m);
    r_free_matrix(v);
}

static void test_adam_nan_grad_propagates(void)
{
    const float theta_v[] = {1.0f};
    const float grad_v[] = {NAN};

    RMatrix *theta = make_matrix(1, 1, theta_v);
    RMatrix *grad = make_matrix(1, 1, grad_v);
    RMatrix *m = make_zeros(1, 1);
    RMatrix *v = make_zeros(1, 1);

    r_optimization_adam(theta, grad, m, v, 0.001f, 0.9f, 0.999f, 1e-8f, 1);

    assert(isnan(theta->data[0]) && "adam: NaN in gradient should propagate to theta");

    r_free_matrix(theta);
    r_free_matrix(grad);
    r_free_matrix(m);
    r_free_matrix(v);
}

static void test_all_optimizers_empty_matrix(void)
{
    RMatrix *theta = r_create_matrix(0, 0);
    RMatrix *grad = r_create_matrix(0, 0);
    RMatrix *aux1 = r_create_matrix(0, 0);
    RMatrix *aux2 = r_create_matrix(0, 0);

    r_optimization_sgd(theta, grad, 0.01f);
    r_optimization_sgdm(theta, grad, aux1, 0.01f, 0.9f);
    r_optimization_rmsprop(theta, grad, aux1, 0.01f, 0.9f, 1e-8f);
    r_optimization_adam(theta, grad, aux1, aux2, 0.001f, 0.9f, 0.999f, 1e-8f, 1);

    r_free_matrix(theta);
    r_free_matrix(grad);
    r_free_matrix(aux1);
    r_free_matrix(aux2);
}

int main(void)
{
    test_sgd_basic();
    test_sgd_negative_grad();
    test_sgd_zero_grad();
    test_sgd_zero_eta();
    test_sgd_nan_grad_propagates();
    test_sgd_empty_matrix();

    test_sgdm_basic_zero_velocity();
    test_sgdm_beta_zero_equals_sgd();
    test_sgdm_beta_one_ignores_grad();
    test_sgdm_velocity_accumulates();
    test_sgdm_zero_eta();

    test_rmsprop_basic();
    test_rmsprop_rho_zero();
    test_rmsprop_zero_grad();
    test_rmsprop_eps_prevents_div_zero();
    test_rmsprop_cache_accumulates();

    test_adam_basic_t1();
    test_adam_zero_grad();
    test_adam_zero_eta();
    test_adam_two_steps();
    test_adam_large_t();
    test_adam_nan_grad_propagates();

    test_all_optimizers_empty_matrix();

    printf("All optimization tests passed.\n");
    return 0;
}