#pragma once
#ifndef __OPTIM_NEWTON_CG_HPP__
#define __OPTIM_NEWTON_CG_HPP__

/// @cond
template <typename fp_t>
struct cgParams
{
    int max_iter;
    fp_t tol;
};

template <typename fp_t>
int cg(const Mat<fp_t> &A, const Mat<fp_t> &b,
       Mat<fp_t> &x, const cgParams<fp_t> &param = {-1, 1e-10})
{
    BMO_SET_ZERO(x);
    const int max_iter =
        param.max_iter < 0 ? BMO_SIZE(x) : param.max_iter;
    Mat<fp_t> r = b, p = r, Ap;
    fp_t r_nrm2 = BMO_SQUARE_NORM(r), r_nrm2_new, alpha;
    const fp_t r0_nrm = std::sqrt(r_nrm2);
    for (int iter = 0; iter < max_iter; iter++)
    {
        Ap = A * p;
        alpha = r_nrm2 / BMO_MAT_DOT_PROD(p, Ap);
        if (alpha <= 0.)
            return -1;
        x += alpha * p;
        r -= alpha * Ap;
        r_nrm2_new = BMO_SQUARE_NORM(r);
        if (std::sqrt(r_nrm2_new) / r0_nrm < param.tol)
            return 0;
        p = (r_nrm2_new / r_nrm2) * p + r;
        r_nrm2 = r_nrm2_new;
    }
    return 1;
}

template <typename forward_fn, typename fp_t>
int cg(const forward_fn &&apply_A,
       const Mat<fp_t> &b, Mat<fp_t> &x,
       const cgParams<fp_t> &param)
{
    BMO_SET_ZERO(x);
    const int max_iter =
        param.max_iter < 0 ? BMO_SIZE(x) : param.max_iter;
    Mat<fp_t> r = b, p = r, Ap;
    fp_t r_nrm2 = BMO_SQUARE_NORM(r), r_nrm2_new, alpha;
    const fp_t r0_nrm = std::sqrt(r_nrm2);
    for (int iter = 0; iter < max_iter; iter++)
    {
        apply_A(p, Ap);
        alpha = r_nrm2 / BMO_MAT_DOT_PROD(p, Ap);
        if (alpha <= 0.)
            return -1;
        x += alpha * p;
        r -= alpha * Ap;
        r_nrm2_new = BMO_SQUARE_NORM(r);
        if (std::sqrt(r_nrm2_new) / r0_nrm < param.tol)
            return 0;
        p = (r_nrm2_new / r_nrm2) * p + r;
        r_nrm2 = r_nrm2_new;
    }
    return 1;
}

/// @endcond
#endif