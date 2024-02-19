#pragma once
#ifndef _OPTIM_ZHANG_HAGER_HPP_
#define _OPTIM_ZHANG_HAGER_HPP_

#include "LineSearcher.hpp"

namespace optim
{
    /// @brief Zhang-Hager line search
    template <typename fp_t = double,
              bool use_prox = false>
    class ZHLineSearch final
        : public LineSearcher<fp_t, use_prox>
    {
    public:
        using Problem = LineSearcher<fp_t>::Problem;

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

        explicit ZHLineSearch(
            ZHLineSearch::Problem &prob)
        {
            this->prob = &prob;
        }

        void init(Problem *p, fp_t f)
        {
            this->prob = p;
            this->Cval = f;
        }

        /// @brief
        /// @return square norm of the gradient(positive!)
        fp_t line_search(
            fp_t &step, const Mat<fp_t> &direc,
            const Mat<fp_t> &in_x, Mat<fp_t> &out_x,
            const fp_t &in_loss, fp_t &out_loss,
            const Mat<fp_t> &in_grad, Mat<fp_t> &out_grad)
        {
            const fp_t dTg = BMO_MAT_DOT_PROD(direc, in_grad);
            const fp_t neg_dTg_sgn = dTg > 0 ? fp_t(-1) : fp_t(1);
            const fp_t dTg_abs = dTg * neg_dTg_sgn;
            // line search
            for (iter = 0; iter < max_iter; iter++)
            {
                out_x = in_x + neg_dTg_sgn * step * direc;
                out_loss = this->prob->loss(out_x);
                if (out_loss <= Cval - pho * step * dTg_abs ||
                    step == this->min_step)
                {
                    this->prob->grad(out_x, out_grad);
                    constexpr if (use_prox)
                    {
                        Mat<fp_t> prox_x = out_x;
                        this->prob->prox(step, out_x, prox_x);
                        out_x = std::move(prox_x);
                    }
                    // update pQ, Q, Cval
                    pQ = Q, Q = gamma * Q + 1;
                    Cval = (gamma * pQ * Cval + out_loss) / Q;
                    return std::sqrt(dTg);
                }
                step *= decay_rate;
                // force step larger than min_step
                if (step < this->min_step)
                    step = this->min_step;
            }
            // if line search failed, use the default step
            // TODO: give a warning
            out_x = in_x + neg_dTg_sgn * this->min_step * direc;
            out_loss = this->prob->loss(out_x);
            this->prob->grad(out_x, out_grad);
            return dTg;
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
}

// #ifdef OPTIM_HEADER_ONLY
// #include "line_search/Zhang_Hager.hpp"
// #endif

#endif