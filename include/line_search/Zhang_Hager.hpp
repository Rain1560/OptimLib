#pragma once
#ifndef _OPTIM_ZHANG_HAGER_HPP_
#define _OPTIM_ZHANG_HAGER_HPP_

/**

*/

#include "base.hpp"

namespace optim
{
    /// @brief Zhang-Hager line search
    /// @details find a step \[\alpha_k\] such that \[f(x_k + \alpha_k d) \leq C_k + \rho \alpha_k \nabla f(x_k)^T d\], where \[C_k = \frac{\gamma_k p_k C_{k-1} + f(x_k)}{\gamma_k + 1}, C_0 = f(x_)\].
    template <typename fp_t = double,
              bool use_prox = false>
    class ZHLineSearch final
        : public LineSearch<fp_t, use_prox>
    {
    public:
        using Problem = LineSearch<fp_t, use_prox>::Problem;
        using Args = LineSearchArgs<fp_t, use_prox>;

    private:
        int iter = 0;
        fp_t pQ = 0, Q = 1, Cval;

    public:
        int max_iter = 10; ///< max iter of line search
        fp_t pho = 1e-3;   ///< sufficient decrease parameter in line search
        fp_t gamma = 0.85;
        fp_t decay_rate = 0.1; ///< decay rate of step length in line search

    private:
        int status = 0;

    public:
        ZHLineSearch() {}

        explicit ZHLineSearch(Problem &prob)
        {
            this->prob = &prob;
        }

        void init(Problem *p, Args &arg)
        {
            this->prob = p;
            this->Cval = arg.cur_loss;
        }

        /// @brief 
        void line_search(Args &arg) override
        {
            optim_assert(arg.step > 0, "step must be positive.");
            iter = 0; // reset iter
            fp_t dTg; // d.dot(g) in sm prob or
            // or (cur_x - prev_x) / step .dot(g_grad_map) in nsm prob
            // begin line search
            for (iter = 1; iter <= max_iter; iter++)
            {
                arg.step_forward(this->prob);
                arg.update_loss(this->prob);
                if constexpr (use_prox)
                {
                    arg.update_prev_gradmap(this->prob);
                    arg.tmp = arg.cur_x - arg.prev_x;
                    dTg = BMO_MAT_DOT_PROD(
                              arg.tmp, arg.prev_grad_map) /
                          arg.step;
                }
                else
                    dTg = BMO_MAT_DOT_PROD(arg.direction, arg.prev_grad);
                optim_assert(dTg < 0, "dTg must be negtive.");
                if (arg.cur_loss <= Cval - pho * arg.step * dTg ||
                    arg.step == this->min_step)
                {
                    arg.update_grad(this->prob);
                    // update pQ, Q, Cval
                    pQ = Q, Q = gamma * Q + 1;
                    Cval = (gamma * pQ * Cval + arg.cur_loss) / Q;
                    return;
                }
                arg.step *= decay_rate;
                // force step larger than min_step
                if (arg.step < this->min_step)
                    arg.step = this->min_step;
            }
            // if line search failed, use the min step
            // and give a warning(TODO)
            arg.step = this->min_step;
            arg.step_forward(this->prob);
            arg.update_loss(this->prob);
            arg.update_grad(this->prob);
            return;
        };

        bool success() const override
        {
            return status == 0;
        }

        int n_iter() const
        {
            return iter;
        }
    };

    template <typename fp_t = double, bool use_prox = false>
    using ZHLS = ZHLineSearch<fp_t, use_prox>;
}

// #ifdef OPTIM_HEADER_ONLY
// #include "line_search/Zhang_Hager.hpp"
// #endif

#endif