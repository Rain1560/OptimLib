#ifndef __OPTIM_LINE_SEARCH_ARMIJO_HPP__
#define __OPTIM_LINE_SEARCH_ARMIJO_HPP__

#include "base.hpp"

namespace optim
{
    template <typename fp_t, bool use_prox = false>
    struct ArmijoLineSearch final
        : public LineSearch<fp_t, use_prox>
    {
        using Problem = typename LineSearch<fp_t, use_prox>::Problem;
        using Args = LineSearchArgs<fp_t, use_prox>;

        int max_iter = 10;
        fp_t armijo_c = 0.95;
        fp_t decay_rate = 0.1;

        void line_search(Args &arg)
        {
            optim_assert(arg.step > 0, "step must be positive.");
            int iter = 0;
            fp_t dTg;
            for (iter = 1; iter <= max_iter; iter++)
            {
                arg.step_forward(this->prob);
                arg.update_cur_loss(this->prob);
                if constexpr (use_prox)
                {
                    arg.update_prev_grad_map(this->prob);
                    arg.tmp = arg.cur_x - arg.prev_x;
                    dTg = BMO_MAT_DOT_PROD(
                              arg.tmp, arg.prev_grad_map) /
                          arg.step;
                }
                else
                    dTg = BMO_MAT_DOT_PROD(arg.direction, arg.prev_grad);
                optim_assert(dTg < 0, "dTg must be negtive.");
                if (arg.cur_loss <= arg.prev_loss + arg.step * dTg ||
                    arg.step == this->min_step)
                    goto over;
                arg.step *= decay_rate;
                // force step larger than min_step
                if (arg.step < this->min_step)
                    arg.step = this->min_step;
            }
            arg.step = this->min_step;
            arg.step_forward(this->prob);
            arg.update_cur_loss(this->prob);
        over:
            arg.update_cur_grad(this->prob);
        }
    };

} // namespace optim

#endif