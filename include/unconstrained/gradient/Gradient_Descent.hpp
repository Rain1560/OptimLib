#pragma once
#ifndef _OPTIM_PROXIMAL_GRADIENT_DESCENT_HPP_
#define _OPTIM_PROXIMAL_GRADIENT_DESCENT_HPP_

#include "line_search/More_Thuente.hpp"
#include "line_search/Zhang_Hager.hpp"
#include "base/BaseSolver.hpp"
#include "Accelerator.hpp"
#include "Step_Sceduler.hpp"
/// @cond
namespace optim
{

    template <typename fp_t, bool use_prox = false>
    class GradientDescent
        : public LSBaseSolver<fp_t, use_prox>
    {
    public:
        using Problem = ProxWrapper<
            fp_t, GradProblem, use_prox>;
        using Accel = GDAccelerator<fp_t>;
        using lrScheduler = StepScheduler<fp_t>;
        using LineSearchImp = typename LSBaseSolver<fp_t, use_prox>::LineSearchImp;

        using BaseSolver<fp_t>::iter;
        using LSBaseSolver<fp_t, use_prox>::ls;

    private:
        std::shared_ptr<Problem> prob;

    public:
        std::shared_ptr<Accel> accelerator;
        std::shared_ptr<lrScheduler> lr_scheduler;

        int max_iter = 1000;
        fp_t step = 1e-2;
        fp_t xtol = 1e-6,
             ftol = 1e-4,
             gtol = 1e-4;

    public:
        GradientDescent(std::shared_ptr<Problem> p)
        {
            prob = p;
            accelerator.reset(new GDAccelerator<fp_t>());
            ls.reset(new LineSearch<fp_t, use_prox>());
            lr_scheduler.reset(new StepScheduler<fp_t>());
        };

        fp_t solve(Mat<fp_t> &x) override
        {
            LineSearchArgs<fp_t, use_prox> arg(x);
            Mat<fp_t> prev_x = x;
            fp_t diff_x_nrm, diff_abs_f, g_nrm;
            ls->update_cur_grad = false;
            // first step
            arg.step = this->step;
            arg.update_cur_loss(prob.get());
            arg.update_cur_grad(prob.get());
            arg.direction = -arg.cur_grad;
            ls->init(prob, arg);
            arg.flush(); // move cur to prev
            ls->line_search(arg);
            accelerator->init(arg); // initialize the accelerator
            for (iter = 1; iter <= max_iter; iter++)
            {
                // update direction, cur_x and cur_grad
                auto grad_f = [this](const Mat<fp_t> &in_x,
                                     Mat<fp_t> &out_grad)
                { prob->grad(in_x, out_grad); };
                accelerator->update(iter, grad_f, arg);
                // update step size
                lr_scheduler->update(iter, arg);
                // line search
                BMO_SWAP(arg.prev_x, prev_x);
                arg.flush(); // move cur to prev
                ls->line_search(arg);
                // check stop criteria
                prev_x -= arg.cur_x, diff_x_nrm = BMO_FRO_NORM(prev_x);
                diff_abs_f = std::abs(arg.cur_loss - arg.prev_loss);
                if constexpr (use_prox)
                {
                    arg.update_cur_grad_map(prob.get());
                    g_nrm = BMO_FRO_NORM(arg.cur_grad_map);
                }
                else
                    g_nrm = BMO_FRO_NORM(arg.cur_grad);
                if (iter % 5 == 0)
                    logger.info("[GD] iter: {:<5d}| loss: {:<16g}| step: {:<10g}\n|g_nrm: {:<12g}| diff_x_nrm: {:<10g}| diff_abs_f: {:<10g}",
                                iter, arg.cur_loss, arg.step, g_nrm, diff_x_nrm, diff_abs_f);
                else
                    logger.trace("[GD] iter: {:<5d}| loss: {:<16g}| step: {:<10g}\n|g_nrm: {:<12g}| diff_x_nrm: {:<10g}| diff_abs_f: {:<10g}", iter, arg.cur_loss, arg.step, g_nrm, diff_x_nrm, diff_abs_f);
                if (g_nrm < gtol || (diff_x_nrm < xtol && diff_abs_f < ftol))
                    goto over;
            }
        over:
            accelerator->release();
            lr_scheduler->release();
            x = std::move(arg.cur_x);
            return arg.cur_loss;
        };
    };

    template <typename T>
    using GD = GradientDescent<T, false>;
    template <typename T>
    using ProxGD = GradientDescent<T, true>;
};
/// @endcond
#endif // !_OPTIM_PROXIMAL_GRADIENT_DESCENT_HPP_