#pragma once
#ifndef _OPTIM_LINE_SEARCH_BASE_HPP__
#define _OPTIM_LINE_SEARCH_BASE_HPP__

#include "base/BaseSolver.hpp"

namespace optim
{
    template <typename, bool>
    struct LineSearchArgs;

    template <typename fp_t>
    struct LineSearchArgs<fp_t, false>
    {
        using Problem = GradProblem<fp_t>;

        fp_t step, prev_loss, cur_loss;
        Mat<fp_t> direction;
        Mat<fp_t> prev_x, cur_x;
        Mat<fp_t> prev_grad, cur_grad;

        LineSearchArgs(Mat<fp_t> &x)
        {
            const Index n = BMO_ROWS(x),
                        m = BMO_COLS(x);
            cur_x = x;
            BMO_RESIZE(prev_x, n, m);
            BMO_RESIZE(prev_grad, n, m);
            BMO_RESIZE(cur_grad, n, m);
            BMO_RESIZE(direction, n, m);
        }

        OPTIM_STRONG_INLINE
        void step_forward(Problem *)
        {
            cur_x = prev_x + step * direction;
        }

        OPTIM_STRONG_INLINE
        void update_cur_loss(Problem *prob)
        {
            cur_loss = prob->loss(cur_x);
        }

        OPTIM_STRONG_INLINE
        void update_cur_grad(Problem *prob)
        {
            prob->grad(cur_x, cur_grad);
        }

        OPTIM_STRONG_INLINE
        void flush()
        {
            BMO_SWAP(prev_x, cur_x);
            BMO_SWAP(prev_grad, cur_grad);
            prev_loss = cur_loss;
        }
    };

    template <typename fp_t>
    struct LineSearchArgs<fp_t, true>
    {
        using Problem = internal::ProxOperator<fp_t, GradProblem>;

        fp_t step, prev_loss, cur_loss;
        Mat<fp_t> direction;
        Mat<fp_t> prev_x, cur_x;
        Mat<fp_t> prev_grad, cur_grad;
        Mat<fp_t> prev_grad_map, cur_grad_map;
        fp_t prev_sm_loss, prev_nsm_loss;
        fp_t cur_sm_loss, cur_nsm_loss;
        Mat<fp_t> tmp; // storage tmp value

    public:
        LineSearchArgs(Mat<fp_t> &x)
        {
            const Index n = BMO_ROWS(x),
                        m = BMO_COLS(x);
            cur_x = x;
            BMO_RESIZE(prev_x, n, m);
            BMO_RESIZE(prev_grad, n, m);
            BMO_RESIZE(cur_grad, n, m);
            BMO_RESIZE(direction, n, m);
            BMO_RESIZE(prev_grad_map, n, m);
            BMO_RESIZE(cur_grad_map, n, m);
            BMO_RESIZE(tmp, n, m);
        }

        OPTIM_STRONG_INLINE void
        step_forward(Problem *prob)
        {
            cur_x = prev_x + step * direction;
            prob->prox(step, cur_x, tmp);
            BMO_SWAP(cur_x, tmp);
        }

        OPTIM_STRONG_INLINE void
        update_cur_loss(Problem *prob)
        {
            cur_sm_loss = prob->sm_loss(cur_x);
            cur_nsm_loss = prob->nsm_loss(cur_x);
            cur_loss = cur_sm_loss + cur_nsm_loss;
        }

        OPTIM_STRONG_INLINE void
        update_cur_grad(Problem *prob)
        {
            prob->grad(cur_x, cur_grad);
            cur_grad_map = cur_x - cur_grad;
            prob->prox(1, cur_grad_map, tmp);
            cur_grad_map = (cur_x - tmp) / step;
        }

        OPTIM_STRONG_INLINE void
        update_prev_grad_map(Problem *prob)
        {
            prev_grad_map = prev_x - step * prev_grad;
            prob->prox(step, prev_grad_map, tmp);
            prev_grad_map = (prev_x - tmp) / step;
        }

        OPTIM_STRONG_INLINE void
        update_cur_grad_map(Problem *prob)
        {
            cur_grad_map = cur_x - step * cur_grad;
            prob->prox(step, cur_grad_map, tmp);
            cur_grad_map = (cur_x - tmp) / step;
        }

        OPTIM_STRONG_INLINE void
        flush()
        {
            BMO_SWAP(prev_x, cur_x);
            BMO_SWAP(prev_grad, cur_grad);
            BMO_SWAP(prev_grad_map, cur_grad_map);
            prev_sm_loss = cur_sm_loss;
            prev_nsm_loss = cur_nsm_loss;
            prev_loss = cur_loss;
        }
    };

    template <typename fp_t,
              bool use_prox = false>
    struct LineSearch
    {
        using Problem = ProxWrapper<
            fp_t, GradProblem, use_prox>;
        using Constant = OptimConst<fp_t>;
        using Args = LineSearchArgs<fp_t, use_prox>;

    protected:
        Problem *prob;

    public:
        fp_t min_step = Constant::sqrt_eps;
        fp_t max_step = fp_t(1e2);

        virtual bool success() const { return true; }

        virtual void init(Problem *p, Args &)
        {
            this->prob = p;
        }

        virtual void line_search(Args &arg)
        {
            arg.step_forward(prob);
            arg.update_cur_grad(prob);
        }

        virtual ~LineSearch() = default;
    };
}

#endif