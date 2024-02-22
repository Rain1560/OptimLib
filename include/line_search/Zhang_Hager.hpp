#pragma once
#ifndef _OPTIM_ZHANG_HAGER_HPP_
#define _OPTIM_ZHANG_HAGER_HPP_

#include "base.hpp"

namespace optim
{
    /// @brief Zhang-Hager line search
    template <typename fp_t = double,
              bool use_prox = false>
    class ZHLineSearch final
        : public LineSearch<fp_t, use_prox>
    {
    public:
        using Problem = LineSearch<fp_t, use_prox>::Problem;

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
            LineSearch<fp_t, use_prox>::Problem &prob)
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
            fp_t dTg_abs = -dTg * neg_dTg_sgn;
            Mat<fp_t> prox_x; // for proximal operator
            if constexpr (use_prox)
            {
                const int m = BMO_ROWS(in_x),
                          n = BMO_COLS(in_x);
                prox_x = Mat<fp_t>(m, n);
            }
            // line search
            for (iter = 1; iter <= max_iter; iter++)
            {
                out_x = in_x + neg_dTg_sgn * step * direc;
                if constexpr (use_prox)
                {
                    this->prob->prox(step, out_x, prox_x);
                    out_x = std::move(prox_x);
                    // temporary use to store x_{k+1} - x_k
                    prox_x = out_x - in_x;
                    dTg_abs = BMO_SQUARE_NORM(prox_x) / fp_t(2);
                }
                out_loss = this->prob->loss(out_x);
                if (out_loss <= Cval - pho * step * dTg_abs ||
                    step == this->min_step)
                {
                    this->prob->grad(out_x, out_grad);
                    // update pQ, Q, Cval
                    pQ = Q, Q = gamma * Q + 1;
                    Cval = (gamma * pQ * Cval + out_loss) / Q;
                    return dTg_abs;
                }
                step *= decay_rate;
                // force step larger than min_step
                if (step < this->min_step)
                    step = this->min_step;
            }
            // if line search failed, use the min step
            // and give a warning(TODO)
            out_x = in_x + neg_dTg_sgn * this->min_step * direc;
            out_loss = this->prob->loss(out_x);
            this->prob->grad(out_x, out_grad);
            return dTg_abs;
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