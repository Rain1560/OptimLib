#pragma once
#ifndef _OPTIMLIB_LINE_SEARCH_MORE_THUENTE_H_
#define _OPTIMLIB_LINE_SEARCH_MORE_THUENTE_H_

#include "More_Thuente.ipp"

namespace optim
{
    /*!
     @class LinearSearchMT
     @brief More-Thuente line search
     */
    template <typename fp_t = double,
              bool use_prox = false>
    class MTLineSearch final
        : public LineSearch<fp_t, use_prox>
    {
        friend class BaseSolver<fp_t>;

    public:
        using Problem = LineSearch<fp_t>::Problem;
        using Args = LineSearchArgs<fp_t, use_prox>;

    private:
        int iter = 0;

    public:
        int max_iter = 10;
        fp_t wolfe_c1 = 1e-3;
        fp_t wolfe_c2 = 0.9;
        fp_t tol = 1e-4;

    private:
        int status = 0; // wether ls success

    private:
        bool check_wolfe_condition(
            fp_t step,
            fp_t f_0, fp_t g_0,
            fp_t f_t, fp_t g_t)
        {
            using std::abs;
            return (f_t <= f_0 + step * wolfe_c1 * g_0) &&
                   (abs(g_t) <= wolfe_c2 * abs(g_0));
        };

    public:
        MTLineSearch() {}

        explicit MTLineSearch(Problem &p)
        {
            this->prob = &p;
        }

        /// @brief More-Thuente line search Algorithm implementation for efficiently invoke by taking less computing on f(x0),\f$\nabla f(x0)\f$
        void line_search(Args &arg) override
        {
            using std::abs;
            using std::max;
            using std::min;
            using namespace internal::MTLineSearch;
            optim_assert(arg.step > 0, "step must be positive.");
            iter = 0;      // reset iter
            fp_t g_0, g_t; // = grad_0.T * d;
            // check wolfe condition
            arg.step_forward(this->prob);
            arg.update_loss(this->prob);
            arg.update_grad(this->prob);
            // update subgradient if needed
            if constexpr (use_prox)
            {
                arg.update_prev_gradmap(this->prob);
                arg.update_cur_gradmap(this->prob);
                arg.tmp = arg.cur_x - arg.prev_x;
                g_0 = BMO_MAT_DOT_PROD(arg.prev_grad_map, arg.tmp) /
                      arg.step;
                g_t = BMO_MAT_DOT_PROD(arg.cur_grad_map, arg.tmp) /
                      arg.step;
            }
            else
            {
                g_0 = BMO_MAT_DOT_PROD(arg.prev_grad, arg.direction);
                g_t = BMO_MAT_DOT_PROD(arg.cur_grad, arg.direction);
            }
            optim_assert(g_0 < 0, "g_0 must be negtive.");
            if (check_wolfe_condition(
                    arg.step, arg.prev_loss, g_0, arg.cur_loss, g_t))
                return;
            bool use_auxiliary = 1, brackt = 0;
            fp_t step_max = arg.step * 10,
                 step_min = this->min_step;
            fp_t width = this->max_step - this->min_step,
                 width_old = 2. * width;
            FuncVal<fp_t, use_prox>
                a_l(0, arg.prev_loss, g_0),
                a_t(arg.step, arg.cur_loss, g_t),
                a_u(0, arg.prev_loss, g_0);
            const int n = BMO_ROWS(arg.cur_x),
                      m = BMO_COLS(arg.cur_x);
            Mat<fp_t> best_x(n, m), best_grad(n, m), best_gmap;
            if constexpr (use_prox)
                BMO_RESIZE(best_gmap, n, m);
            g_t = wolfe_c1 * g_0; // gtest in origin code
            logger.info("[MTLS] start line-search.");
            for (iter = 1; iter <= max_iter; iter++)
            {
                if (use_auxiliary)
                {
                    a_l.phi_to_psi(g_t);
                    a_t.phi_to_psi(g_t);
                    a_u.phi_to_psi(g_t);
                    arg.step = select_trial_value<fp_t>(
                        brackt, a_l, a_u, a_t, step_min, step_max);
                    // Reset the function and derivative values for f.
                    a_l.psi_to_phi(g_t);
                    a_t.psi_to_phi(g_t);
                    a_u.psi_to_phi(g_t);
                }
                else
                    arg.step = select_trial_value(
                        brackt, a_l, a_u, a_t, step_min, step_max);
                // Force the step to be within the bounds
                arg.step = std::min(std::max(arg.step, this->min_step), this->max_step);
                // ----------------- Updating Method -----------------
                if (a_t.val > a_l.val)
                    // case U1 & a
                    a_u.swap(a_t); // shrink interval
                else
                {
                    if (std::signbit(a_t.deriv) !=
                        std::signbit(a_l.deriv))
                        // a_t and a_l should be endpoints
                        a_u.swap(a_l);
                    a_l.swap(a_t); // case U2 & b
                    best_x = std::move(arg.cur_x);
                    best_grad = std::move(arg.cur_grad);
                    if constexpr (use_prox)
                        best_gmap = std::move(arg.cur_grad_map);
                }
                if (check_wolfe_condition(
                        a_l.arg, arg.prev_loss, g_0, a_l.val, a_l.deriv))
                { // that's what we want!
                    status = 0;
                    goto over;
                }
                // update the 'interval of uncertainty'
                if (brackt)
                {
                    if (abs(a_u.arg - a_l.arg) > 0.66 * width_old) [[unlikely]]
                        // Decide if a bisection step is needed.
                        arg.step = (a_l.arg + a_u.arg) / 2;
                    width_old = width;
                    width = abs(a_u.arg - a_l.arg);
                    if (step_max < tol) [[unlikely]]
                    { // minimal is too close to origin point!
                        optim_assert(a_l.val <= arg.prev_loss,
                                     "f_t should be less than f_0. Please check your loss and grad func.");
                        status = -1;
                        goto over;
                    }
                    step_max = max(a_l.arg, a_u.arg);
                    step_min = min(a_l.arg, a_u.arg);
                    if ((arg.step > step_max || arg.step < step_min) ||
                        step_max - step_min <= 1e-4 * step_max) [[unlikely]]
                        arg.step = (step_max + step_min) / 2;
                }
                else
                { // step too small, expand interval.
                    step_min = arg.step + 1.1 * (arg.step - a_l.arg);
                    step_max = arg.step + 5.0 * (arg.step - a_l.arg);
                }
                // log
                if (iter % 10 == 0)
                {
                    logger.info("[MTLS] iter: {:<3d}  brackt:{:<5}  use_auxilary: {:<5}\nstep_min: {:<10g} | a_l: {:<10g} | a_t: {:<10g} | a_u: {:<10g} | step_max: {:<10g}", iter, brackt, use_auxiliary, step_min, a_l.arg, a_t.arg, a_u.arg, step_max);
                }
                else
                {
                    // f_t <= f_0 + step * wolfe_c1 * g_0
                    logger.trace("[MTLS] iter: {:<3d}  brackt:{:<5}  use_auxilary: {:<5}\n| a_l: {:<10g} | a_t: {:<10g} | a_u: {:<10g} |\n| f_l: {:<10g} | f_t: {:<10g} | f_u: {:<10g} |\n| g_l: {:<10g} | g_t: {:<10g} | g_u: {:<10g} |", iter, brackt, use_auxiliary, a_l.arg, a_t.arg, a_u.arg, a_l.val, a_t.val, a_u.val, a_l.deriv, a_t.deriv, a_u.deriv);
                }
                // compute new step
                a_t.update_val(this->prob, arg);
                // check if switch to "Modified Updating Method"
                if (use_auxiliary &&
                    a_t.val <= arg.prev_loss + g_t && // psi(a_t) <= 0
                    a_t.deriv >= 0)                   // phi'(a_t) >= 0
                    use_auxiliary = 0;
            }
            status = 1;
            logger.warn("[MTLS] reaches max_iter: {}.", max_iter);
            optim_assert(a_l.val <= arg.prev_loss,
                         "cur_loss should be less than prev_loss. Please check your loss, grad func.");
        over:
            arg.step = a_l.arg;
            arg.cur_loss = a_l.val;
            arg.cur_x = std::move(best_x);
            arg.cur_grad = std::move(best_grad);
            if constexpr (use_prox)
                arg.cur_grad_map = std::move(best_gmap);
        }

        bool success() const override { return status == 0; }

        int n_iter() const { return iter; };
    };

    template <typename fp_t = double, bool use_prox = false>
    using MTLS = MTLineSearch<fp_t, use_prox>;
}

#endif