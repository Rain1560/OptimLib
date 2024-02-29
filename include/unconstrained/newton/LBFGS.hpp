#pragma once
#ifndef _OPTIM_NEWTON_LBFGS_HPP_
#define _OPTIM_NEWTON_LBFGS_HPP_

#include "line_search/More_Thuente.hpp"
#include "LBFGS.ipp"

namespace optim
{
    template <typename fp_t>
    class LBFGS final : public BaseSolver<fp_t>
    {
        using Storage = internal::LBFGS::Storage<fp_t>;

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
        fp_t step = 1e-2;   ///< initial step size
        int m = 4;          ///< number of memory
        int status;         // wether solve successfully

    private:
        OPTIM_STRONG_INLINE void
        // assign gradient to direction
        update_direction(LineSearchArgs<fp_t, use_prox> &arg)
        {
            arg.direction = -arg.cur_grad;
        }

        OPTIM_INLINE void
        update_sy(
            Storage &sy,
            LineSearchArgs<fp_t, use_prox> &arg)
        {
            sy.s = arg.cur_x - arg.prev_x;
            sy.y = arg.cur_grad - arg.prev_grad;
            // compute bb_step1
            const double sTy = BMO_MAT_DOT_PROD(sy.s, sy.y);
            const double y_nrm2 = BMO_SQUARE_NORM(sy.y);
            if (y_nrm2 < Constant::eps || sTy < 0)
                OPTIM_UNLIKELY
                { // too small to storage!
                    arg.step = step;
                    return;
                }
            arg.step = sTy / y_nrm2;
            // force bb_step to be in [min_step, max_step]
            arg.step = std::min(
                std::max(ls->min_step, arg.step), ls->max_step);
            sy.rho = 1. / sTy;
        }

    public:
        explicit LBFGS(
            Problem &prob,
            LineSearchImp &ls)
        {
            this->prob = &prob;
            this->ls = &ls;
        };

        fp_t solve(Mat<fp_t> &x) override
        {
            CircularArray<Storage> memory(m);
            Storage sy;
            LineSearchArgs<fp_t, use_prox> arg(x);
            fp_t g_nrm, x_diff_nrm, f_diff;
            arg.update_loss(this->prob);
            arg.update_grad(this->prob);
            ls->init(this->prob, arg);
            arg.step = step;
            // make step 0
            update_direction(arg);
            arg.flush();
            ls->line_search(arg);
            // update bb_step to arg.step
            update_sy(sy, arg);
            memory.push_back(std::move(sy));
            // begin main loop
            for (iter = 1; iter <= max_iter; iter++)
            {
                update_direction(arg);
                // compute H * -grad
                lbfgs_update_direction(arg.step, memory, arg.direction);
                // update prev_x, prev_g, prev_loss and memory
                arg.step = fp_t(1), arg.flush();
                ls->line_search(arg);
                update_sy(sy, arg);
                // check stop criteria
                g_nrm = BMO_FRO_NORM(arg.cur_grad);
                x_diff_nrm = BMO_FRO_NORM(sy.s); // = norm(d) * step;
                f_diff = std::abs(arg.cur_loss - arg.prev_loss) /
                         std::abs(arg.cur_loss);
                if (g_nrm < gtol && x_diff_nrm < xtol && f_diff < ftol)
                { // check stop criteria and resize back
                    status = 0;
                    x = std::move(arg.cur_x);
                    return arg.cur_loss;
                }
                // update memory
                memory.push_back(std::move(sy));
            }
            status = 1; // TODO: log
            x = std::move(arg.cur_x);
            return arg.cur_loss;
        };
    };
}

// #ifdef OPTIM_HEADER_ONLY
// #include "unconstrained/newton/LBFGS.cpp"
// #endif
#endif