#pragma once
#ifndef _OPTIM_BB_HPP_
#define _OPTIM_BB_HPP_

#include "line_search/Zhang_Hager.hpp"

namespace optim
{
    template <typename fp_t = double,
              bool use_prox = false>
    struct BarzilarBorwein final
        : public LSBaseSolver<fp_t, use_prox>
    {
        using Problem = ProxWrapper<
            fp_t, GradProblem, use_prox>;
        using Constant = OptimConst<fp_t>;
        using LineSearchImp = typename LSBaseSolver<fp_t, use_prox>::LineSearchImp;

        using BaseSolver<fp_t>::iter;
        using LSBaseSolver<fp_t, use_prox>::ls;

    private:
        std::shared_ptr<Problem> prob;

    public:
        int max_iter = 100; /// max number of iterations
        fp_t xtol = 1e-4;   /// stop if |x_{k+1} - x_k| < xtol
        fp_t ftol = 1e-4;   /// stop if |f_{k+1} - f_k| < ftol
        fp_t gtol = 1e-4;   /// stop if |g_{k}| < gtol
        fp_t step = 1e-2;   /// initial step length, used when bb step is not available

    public:
        explicit BarzilarBorwein(
            std::shared_ptr<Problem> prob)
        {
            this->prob = prob;
            this->ls.reset(new ZHLS<fp_t, use_prox>());
        }

        explicit BarzilarBorwein(
            std::shared_ptr<Problem> prob,
            std::shared_ptr<LineSearchImp> ls)
        {
            this->prob = prob;
            this->ls = ls;
        }

        fp_t solve(Mat<fp_t> &x) override
        { // init the first points
            const size_t n = BMO_ROWS(x),
                         m = BMO_COLS(x);
            LineSearchArgs<fp_t, use_prox> args(x);
            Mat<fp_t> s(n, m), y(n, m);
            fp_t g_nrm, diff_x_nrm, diff_abs_f;
            // first step
            args.step = this->step;
            args.update_cur_loss(prob.get());
            args.update_cur_grad(prob.get());
            args.direction = -args.cur_grad;
            ls->init(prob, args);
            args.flush(); // move cur to prev
            ls->line_search(args);
            for (iter = 1; iter <= max_iter; iter++)
            {
                // update s and y
                s = args.cur_x - args.prev_x;
                y = args.cur_grad - args.prev_grad;
                diff_x_nrm = BMO_FRO_NORM(s);
                if constexpr (use_prox)
                {
                    args.update_cur_grad_map(prob.get());
                    g_nrm = BMO_FRO_NORM(args.cur_grad_map);
                }
                else
                    g_nrm = BMO_FRO_NORM(args.cur_grad);
                // check stop criteria
                diff_abs_f = std::abs(args.cur_loss - args.prev_loss) /
                             (std::abs(args.prev_loss) + 1.);
                if ((diff_x_nrm < xtol &&
                     diff_abs_f < ftol) ||
                    g_nrm < gtol)
                    goto success_end;
                const double sTy = std::abs(BMO_MAT_DOT_PROD(s, y));
                if (sTy > 0)
                { // use bb step1 or step2
                    if (iter % 2 == 1)
                        args.step = sTy / BMO_SQUARE_NORM(y);
                    else
                        args.step = BMO_SQUARE_NORM(s) / sTy;
                    // force bb_step to be in [min_step, max_step]
                    args.step = std::min(
                        std::max(ls->min_step, args.step), ls->max_step);
                }
                else // sTy == 0
                    args.step = step;
                // "copy" cur to prev and record loss
                args.direction = -args.cur_grad;
                args.flush();
                // (Zhang & Hager) line search
                ls->line_search(args);
                if (iter % 5 == 0)
                    logger.info("[BB] iter: {:<5d}| loss: {:<16g}| bb_step: {:<10g}\n|g_nrm: {:<12g}| diff_x_nrm: {:<10g}| diff_abs_f: {:<10g}",
                                iter, args.cur_loss, args.step, g_nrm, diff_x_nrm, diff_abs_f);
                else
                    logger.trace("[BB] iter: {:<5d}| loss: {:<16g}| bb_step: {:<10g}\n|g_nrm: {:<12g}| diff_x_nrm: {:<10g}| diff_abs_f: {:<10g}", iter, args.cur_loss, args.step, g_nrm, diff_x_nrm, diff_abs_f);
            }
            if (diff_x_nrm < xtol)
            {
                if (diff_abs_f < ftol)
                { // g_nrm not converge
                    if (g_nrm >= gtol)
                        logger.warn("BB solver failed to converge at gradient tolerance: {}", g_nrm);
                }
                else
                    // diff_abs_f not converge
                    logger.warn("BB solver failed to converge at function tolerance: {}", diff_abs_f);
            }
            else
            {
                logger.error("BB solver fails to converge! Maybe you should check your gradient function.");
            }
        success_end:
            x = std::move(args.cur_x);
            return args.cur_loss;
        }
    };

    template <typename fp_t = double>
    using BB = BarzilarBorwein<fp_t, false>;

    template <typename fp_t = double>
    using ProxBB = BarzilarBorwein<fp_t, true>;

}

// #ifdef OPTIM_HEADER_ONLY
// #include "unconstrained/gradient/Barzilar_Borwein.cpp"
// #endif
#endif // _OPTIM_BB_HPP_