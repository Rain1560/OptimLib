#pragma once
#ifndef _OPTIM_NEWTON_CG_HPP_
#define _OPTIM_NEWTON_CG_HPP_

#include "line_search/More_Thuente.hpp"
#include "CG.ipp"

namespace optim
{
    template <typename fp_t = double>
    class NewtonCG final
        : public BaseSolver<fp_t>
    {
    public:
        using Constant = OptimConst<fp_t>;
        using Problem = HessProblem<fp_t>;
        using BaseSolver<fp_t>::iter;
        using LineSearchImp = LineSearch<fp_t, false>;

    private:
        Problem *prob;

    public:
        LineSearchImp *ls;
        int max_iter = 100;
        fp_t xtol = 1e-10;
        fp_t ftol = 1e-10;
        fp_t gtol = 1e-6;
        int cg_max_iter = -1;
        int status;

        explicit NewtonCG(
            Problem &prob, LineSearchImp &ls)
        {
            this->prob = &prob;
            this->ls = &ls;
        }

        fp_t solve(Mat<fp_t> &x) override
        {
            using namespace internal::NewtonCG;
            const int n = BMO_ROWS(x),
                      k = BMO_COLS(x),
                      nk = n * k;
            Mat<fp_t> H(nk, nk);
            LineSearchArgs<fp_t, false> arg(x);
            fp_t f_diff, x_diff_nrm, g_nrm;
            cgParams<fp_t> cg_arg;
            if (cg_max_iter < 0 || cg_max_iter > nk)
                cg_arg.max_iter = nk;
            cg_arg.tol = gtol;
            arg.update_cur_loss(prob);
            arg.update_cur_grad(prob);
            ls->init(prob, arg);
            for (iter = 1; iter <= max_iter; iter++)
            {
                arg.flush();
                prob->hess(arg.prev_x, H);
                BMO_RESIZE(arg.direction, nk, 1);
                BMO_RESIZE(arg.prev_grad, nk, 1);
                // solve H * d = -g
                cg(H, arg.prev_grad, arg.direction, cg_arg);
                BMO_RESIZE(arg.direction, n, k);
                BMO_RESIZE(arg.prev_grad, n, k);
                arg.step = 1.;
                arg.direction *= -1.;
                ls->line_search(arg);
                g_nrm = BMO_FRO_NORM(arg.cur_grad);
                x_diff_nrm = BMO_FRO_NORM(arg.direction) * arg.step;
                f_diff = std::abs(arg.cur_loss - arg.prev_loss) /
                         (std::abs(arg.cur_loss) + fp_t(1));
                if ((g_nrm < gtol && x_diff_nrm < xtol && f_diff < ftol) || g_nrm < Constant::eps)
                { // check stop criteria
                    status = 0;
                    x = std::move(arg.cur_x);
                    return arg.cur_loss;
                }
            }
            // TODO : log
            status = 1;
            x = std::move(arg.cur_x);
            return arg.cur_loss;
        }
    };
}

#endif