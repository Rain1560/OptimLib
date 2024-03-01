#include "unconstrained/newton/NewtonCG.hpp"

namespace optim::internal::NewtonCG
{
    template <typename fp_t>
    int cg(const Mat<fp_t> &H, const Mat<fp_t> &g,
           Mat<fp_t> &d, int max_iter, fp_t tol)
    {
        BMO_SET_ZERO(d);
        Mat<fp_t> r = g, p = r, Hp;
        fp_t r_nrm2 = BMO_SQUARE_NORM(r), r_nrm2_new, alpha;
        const fp_t r0_nrm = std::sqrt(r_nrm2);
        for (int iter = 0; iter < max_iter; iter++)
        {
            Hp = H * p;
            alpha = r_nrm2 / BMO_MAT_DOT_PROD(p, Hp);
            if (alpha <= 0.)
                return -1;
            d += alpha * p;
            r -= alpha * Hp;
            r_nrm2_new = BMO_SQUARE_NORM(r);
            if (std::sqrt(r_nrm2_new) / r0_nrm < tol)
                return 0;
            p = r + (r_nrm2_new / r_nrm2) * p;
            r_nrm2 = r_nrm2_new;
        }
        return 1;
    }
}