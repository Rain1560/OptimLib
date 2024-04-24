#pragma once
#ifndef _OPTIM_NEWTON_BASE_HPP__
#define _OPTIM_NEWTON_BASE_HPP__

#include "base/BaseSolver.hpp"
#include "line_search/More_Thuente.hpp"
#include "functions/functions.hpp"

namespace optim
{
    template <typename fp_t>
    class NewtonMethod final
        : public LSBaseSolver<fp_t>
    {
    public:
        using Constant = OptimConst<fp_t>;
        using Problem = HessProblem<fp_t>;
        using LineSearchImp = typename LSBaseSolver<fp_t, false>::LineSearchImp;

        using BaseSolver<fp_t>::iter;
        using LSBaseSolver<fp_t>::ls;

    private:
        std::shared_ptr<Problem> prob;

    public:
        int max_iter = 100;
        fp_t xtol = 1e-10;
        fp_t ftol = 1e-6;
        fp_t gtol = 1e-6;
        int status;

        explicit NewtonMethod(std::shared_ptr<Problem> prob)
        {
            this->prob = prob;
            this->ls = std::make_shared<MTLS<fp_t, false>>();
        }

        explicit NewtonMethod(
            std::shared_ptr<Problem> prob,
            std::shared_ptr<LineSearchImp> ls)
        {
            this->prob = prob;
            this->ls = ls;
        }

        fp_t solve(Mat<fp_t> &x) override
        {
            const Index n = BMO_ROWS(x),
                        k = BMO_COLS(x),
                        nk = n * k;
            Mat<fp_t> H(nk, nk);
            LineSearchArgs<fp_t, false> arg(x);
            fp_t f_diff, x_diff_nrm, g_nrm;
            arg.update_cur_loss(prob.get());
            arg.update_cur_grad(prob.get());
            ls->init(prob, arg);
            for (iter = 1; iter <= max_iter; iter++)
            {
                arg.flush();
                prob->hess_backward(-arg.prev_grad, arg.direction);
                arg.step = 1.;
                ls->line_search(arg);
                g_nrm = BMO_FRO_NORM(arg.cur_grad);
                x_diff_nrm = BMO_FRO_NORM(arg.direction) * arg.step;
                f_diff = std::abs(arg.cur_loss - arg.prev_loss) /
                         (std::abs(arg.cur_loss) + fp_t(1));
                if ((g_nrm < gtol && x_diff_nrm < xtol && f_diff < ftol) ||
                    g_nrm < Constant::eps)
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