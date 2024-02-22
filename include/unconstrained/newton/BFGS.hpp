#pragma once
#ifndef _OPTIM_BFGS_HPP__
#define _OPTIM_BFGS_HPP__

#include "line_search/More_Thuente.hpp"

namespace optim
{
    template <typename fp_t,
              template <typename> class LS = MTLineSearch>
    class BFGS final : public BaseSolver<fp_t>
    {
    public:
        using Problem = GradProblem<fp_t>;
        using Constant = OptimConst<fp_t>;
        using BaseSolver<fp_t>::iter;
        using LineSearchImp = MTLineSearch<fp_t>;

    private:
        Problem *prob;

    public:
        LineSearchImp ls;
        int max_iter = 100; ///< max number of iterations
        fp_t xtol = 1e-6;   ///< stop if |x_{k+1} - x_k| < xtol
        fp_t ftol = 1e-6;   ///< stop if |f_{k+1} - f_k| < ftol
        fp_t gtol = 1e-4;   ///< stop if |g_{k}| < gtol
        fp_t step = 1e-4;   ///< initial step size
        int status;         // wether solve successfully

    public:
        BFGS(Problem &prob)
        {
            this->prob = &prob;
        };

        fp_t solve(Mat<fp_t> &cur_x) override
        {
            const size_t n = BMO_ROWS(cur_x),
                         k = BMO_COLS(cur_x),
                         nk = n * k;
            Mat<fp_t> prev_x = cur_x,
                      cur_g(n, k), prev_g(n, k),
                      s(nk, 1), y(nk, 1), d(nk, 1),
                      Hy(nk, 1),
                      H = BMO_IDENTITY(Mat<fp_t>, nk, nk);
            // H: approximate inverse Hessian
            // I_yst: I - y * s^T / (y^T * s)
            fp_t prev_loss = prob->loss(cur_x), cur_loss,
                 ls_step = step, sTy,
                 g_nrm, x_diff_nrm, f_diff;
            ls.init(this->prob, prev_loss);
            prob->grad(cur_x, prev_g);
            g_nrm = ls.line_search(
                ls_step, prev_g,
                prev_x, cur_x,
                prev_loss, cur_loss,
                prev_g, cur_g);
            for (iter = 1; iter < max_iter; ++iter)
            {
                BMO_RESIZE(cur_x, nk, 1), BMO_RESIZE(prev_x, nk, 1);
                BMO_RESIZE(cur_g, nk, 1), BMO_RESIZE(prev_g, nk, 1);
                s = cur_x - prev_x, y = cur_g - prev_g;
                BMO_RESIZE(cur_x, n, k), BMO_RESIZE(prev_x, n, k);
                BMO_RESIZE(cur_g, n, k), BMO_RESIZE(prev_g, n, k);
                sTy = BMO_MAT_DOT_PROD(s, y), Hy = H * y;
                // update H
                H -= (Hy * BMO_TRANSPOSE(s)) / sTy;
                Hy = BMO_TRANSPOSE(H) * y;
                H -= s * BMO_TRANSPOSE(Hy) / sTy;
                H += (s * BMO_TRANSPOSE(s)) / sTy;
                d = H * cur_g;
                // update prev values
                prev_x = std::move(cur_x);
                prev_g = std::move(cur_g);
                prev_loss = cur_loss;
                ls_step = fp_t(1);
                ls.line_search(
                    ls_step, d,
                    prev_x, cur_x,
                    prev_loss, cur_loss,
                    prev_g, cur_g);
                x_diff_nrm = BMO_NORM(d) * ls_step;
                g_nrm = BMO_NORM(cur_g);
                f_diff = std::abs(cur_loss - prev_loss);
                if (g_nrm < gtol && x_diff_nrm < xtol && f_diff < ftol)
                {
                    status = 0;
                    return cur_loss;
                }
            }
            return cur_loss;
        }
    };
}

#endif