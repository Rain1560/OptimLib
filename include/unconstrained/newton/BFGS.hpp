#pragma once
#ifndef _OPTIM_BFGS_HPP__
#define _OPTIM_BFGS_HPP__

#include "line_search/More_Thuente.hpp"

namespace optim
{
    template <typename fp_t>
    class BFGS final : public BaseSolver<fp_t>
    {
    public:
        using Problem = GradProblem<fp_t>;
        using Constant = OptimConst<fp_t>;
        using BaseSolver<fp_t>::iter;
        using LineSearchImp = LineSearch<fp_t, false>;

    private:
        Problem *prob;

    public:
        LineSearchImp *ls;
        int max_iter = 100; ///< max number of iterations
        fp_t xtol = 1e-6;   ///< stop if |x_{k+1} - x_k| < xtol
        fp_t ftol = 1e-6;   ///< stop if |f_{k+1} - f_k| < ftol
        fp_t gtol = 1e-4;   ///< stop if |g_{k}| < gtol
        fp_t step = 1e-4;   ///< initial step size
        int status;         // wether solve successfully

    public:
        BFGS(Problem &prob, LineSearchImp &ls)
        {
            this->prob = &prob;
            this->ls = &ls;
        };

        fp_t solve(Mat<fp_t> &x) override
        {
            const size_t n = BMO_ROWS(x),
                         k = BMO_COLS(x),
                         nk = n * k;
            LineSearchArgs<fp_t, false> arg(x);
            Mat<fp_t> s(n, k), y(n, k), Hy(nk, 1),
                H = BMO_IDENTITY(Mat<fp_t>, nk, nk);
            // H: approximate inverse Hessian
            fp_t sTy, g_nrm, x_diff_nrm, f_diff;
            arg.step = step;
            arg.step_forward(this->prob);
            arg.update_cur_loss(this->prob);
            arg.update_cur_grad(this->prob);
            arg.direction = -arg.cur_grad;
            ls->init(this->prob, arg);
            arg.flush();
            ls->line_search(arg);
            for (iter = 1; iter < max_iter; ++iter)
            {
                s = arg.cur_x - arg.prev_x;
                y = arg.cur_grad - arg.prev_grad;
                // check convergence condition
                x_diff_nrm = BMO_FRO_NORM(s);
                g_nrm = BMO_FRO_NORM(arg.cur_grad);
                f_diff = std::abs(arg.cur_loss - arg.prev_loss) /
                         (std::abs(arg.cur_loss) + fp_t(1));
                if (g_nrm < gtol && x_diff_nrm < xtol && f_diff < ftol)
                {
                    status = 0;
                    x = std::move(arg.cur_x);
                    return arg.cur_loss;
                }
                BMO_RESIZE(s, nk, 1), BMO_RESIZE(y, nk, 1);
                sTy = BMO_MAT_DOT_PROD(s, y), Hy = H * y;
                // update H
                H -= (Hy * BMO_TRANSPOSE(s)) / sTy;
                Hy = BMO_TRANSPOSE(H) * y;
                H -= s * BMO_TRANSPOSE(Hy) / sTy;
                H += (s * BMO_TRANSPOSE(s)) / sTy;
                arg.direction = -H * arg.cur_grad;
                BMO_RESIZE(arg.direction, n, k);
                BMO_RESIZE(s, n, k), BMO_RESIZE(y, n, k);
                // update prev values
                arg.flush();
                arg.step = fp_t(1);
                ls->line_search(arg);
            }
            return arg.cur_loss;
        }
    };
}

#endif