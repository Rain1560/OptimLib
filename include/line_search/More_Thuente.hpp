#pragma once
#ifndef _OPTIMLIB_LINE_SEARCH_MORE_THUENTE_H_
#define _OPTIMLIB_LINE_SEARCH_MORE_THUENTE_H_

#include "More_Thuente.ipp"

namespace optim
{
    /*!
     @class LinearSearchMT
     @brief More-Thuente line search which using gradient and loss value to find the step meets Wolfe condition. 
     @tparam fp_t float point type
     */
    template <typename fp_t = double>
    class MTLineSearch final
        : public LineSearcher<fp_t>
    {
        friend class BaseSolver<fp_t>;

    public:
        using Problem = LineSearcher<fp_t>::Problem;

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

        void init(Problem *p, fp_t) override
        {
            this->prob = p;
        }

        /// @brief More-Thuente line search Algorithm implementation for efficiently invoke by taking less computing on f(x0),\f$\nabla f(x0)\f$
        /// @param x0 the initial point
        /// @param step the initial step, or learning rate, must be positive!
        /// @param direc the search direction, usually the negative gradient. But to avoid constructing a new vec, g and -g are both acceptable.
        /// @param grad in: \f$\nabla f(x0)\f$ out: \f$\nabla f(x0+step*direc)\f$
        /// @return dot(out_grad, direc)
        fp_t line_search(
            fp_t &step, const Mat<fp_t> &d,
            const Mat<fp_t> &x_0, Mat<fp_t> &x_t,
            const fp_t &f_0, fp_t &f_t,
            const Mat<fp_t> &grad_0, Mat<fp_t> &grad_t)
        {
            using std::abs;
            using std::max;
            using std::min;
            using namespace internal::MTLineSearch;
            const Index rows = BMO_ROWS(x_0),
                        cols = BMO_COLS(x_0);
            iter = 0; // reset iter
            fp_t g_0 = BMO_MAT_DOT_PROD(grad_0, d);
            // check if we need to reflect direction
            const fp_t sgn_d = g_0 > 0 ? fp_t(-1) : fp_t(1);
            // check wolfe condition
            x_t = x_0 + step * sgn_d * d;
            f_t = this->prob->loss(x_t);
            this->prob->grad(x_t, grad_t);
            optim_assert(BMO_ROWS(grad_t) == rows && BMO_COLS(grad_t) == cols,
                         "Size of gradient should be equal to size of x.");
            fp_t g_t = BMO_MAT_DOT_PROD(grad_t, d);
            if (check_wolfe_condition(step, f_0, g_0, f_t, g_t))
                return g_t;
            bool use_auxiliary = 1, brackt = 0;
            fp_t step_max = step * 10,
                 step_min = this->min_step;
            fp_t width = this->max_step - this->min_step,
                 width_old = 2. * width;
            FuncVal<fp_t> a_l(0, rows, cols), a_t(step, rows, cols), a_u(0, rows, cols);
            a_l.val = f_0, a_l.deriv = g_0 * sgn_d;
            a_t.val = f_t, a_t.deriv = g_t * sgn_d;
            a_u.val = f_0, a_u.deriv = g_0 * sgn_d;
            g_t = wolfe_c1 * g_0; // gtest in origin code
            logger.info("MT-line-search started.");
            for (iter = 1; iter <= max_iter; iter++)
            {
                if (use_auxiliary)
                {
                    a_l.phi_to_psi(g_t);
                    a_t.phi_to_psi(g_t);
                    a_u.phi_to_psi(g_t);
                    step = select_trial_value<fp_t>(
                        brackt, a_l, a_u, a_t, step_min, step_max);
                    // Reset the function and derivative values for f.
                    a_l.psi_to_phi(g_t);
                    a_t.psi_to_phi(g_t);
                    a_u.psi_to_phi(g_t);
                }
                else
                {
                    step = select_trial_value(
                        brackt, a_l, a_u, a_t, step_min, step_max);
                }
                // Force the step to be within the bounds
                step = std::min(std::max(step, this->min_step), this->max_step);
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
                }
                if (check_wolfe_condition(
                        a_l.arg, f_0, g_0, a_l.val, a_l.deriv))
                { // that's what we want!
                    status = 0;
                    goto over;
                }
                // update the 'interval of uncertainty'
                if (brackt)
                {
                    if (abs(a_u.arg - a_l.arg) > 0.66 * width_old) [[unlikely]]
                        // Decide if a bisection step is needed.
                        step = (a_l.arg + a_u.arg) / 2;
                    width_old = width;
                    width = abs(a_u.arg - a_l.arg);
                    if (step_max < tol) [[unlikely]]
                    { // minimal is too close to origin point!
                        optim_assert(a_l.val <= f_0,
                                     "f_t should be less than f_0. Please check your loss and grad func.");
                        status = 1;
                        return NAN;
                    }
                    step_max = max(a_l.arg, a_u.arg);
                    step_min = min(a_l.arg, a_u.arg);
                    if ((step > step_max || step < step_min) ||
                        step_max - step_min <= 1e-4 * step_max) [[unlikely]]
                        step = (step_max + step_min) / 2;
                }
                else
                { // step too small, expand interval.
                    step_min = step + 1.1 * (step - a_l.arg);
                    step_max = step + 5.0 * (step - a_l.arg);
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
                a_t.arg = step;
                a_t.update_val(this->prob, x_0, d, sgn_d);
                // check if switch to "Modified Updating Method"
                if (use_auxiliary &&
                    a_t.val <= f_0 + g_t && // psi(a_t) <= 0
                    a_t.deriv >= 0)         // phi'(a_t) >= 0
                    use_auxiliary = 0;
            }
            logger.warn("[MTLS] reaches max_iter: {}.", max_iter);
            optim_assert(a_l.val <= f_0,
                         "f_t should be less than f_0. Please check your loss and grad func.");
        over:
            step = a_l.arg;
            x_t = std::move(a_l.pos);
            f_t = std::move(a_l.val);
            grad_t = a_l.grad;
            return g_0 > 0 ? g_0 : -g_0;
        }

        bool success() const override { return status == 0; }

        int n_iter() const { return iter; };
    };

    // extern MTLineSearch line_search_mt;
}

#endif