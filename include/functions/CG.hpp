#pragma once
#ifndef __OPTIM_NEWTON_CG_HPP__
#define __OPTIM_NEWTON_CG_HPP__

/// @cond 
template <typename fp_t>
struct cgParams
{
    int max_iter;
    fp_t tol;
    int status = 0;
};

template <typename fp_t>
void cg(const Mat<fp_t> &H, const Mat<fp_t> &g,
        Mat<fp_t> &d, cgParams<fp_t> &param)
{
    BMO_SET_ZERO(d);
    Mat<fp_t> r = g, p = r, Hp;
    fp_t r_nrm2 = BMO_SQUARE_NORM(r), r_nrm2_new, alpha;
    const fp_t r0_nrm = std::sqrt(r_nrm2);
    for (int iter = 0; iter < param.max_iter; iter++)
    {
        Hp = H * p;
        alpha = r_nrm2 / BMO_MAT_DOT_PROD(p, Hp);
        if (alpha <= 0.)
        {
            param.status = -1;
            return;
        }
        d += alpha * p;
        r -= alpha * Hp;
        r_nrm2_new = BMO_SQUARE_NORM(r);
        if (std::sqrt(r_nrm2_new) / r0_nrm < param.tol)
        {
            param.status = 0;
            return;
        }
        p = r + (r_nrm2_new / r_nrm2) * p;
        r_nrm2 = r_nrm2_new;
    }
    param.status = 1;
}

/// @endcond
#endif