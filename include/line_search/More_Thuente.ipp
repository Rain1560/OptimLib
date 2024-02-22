#include "base.hpp"

// reference: https://ftp.mcs.anl.gov/pub/MINPACK-2/csrch
namespace optim::internal::MTLineSearch
{
    template <typename fp_t>
    struct FuncVal
    {
        fp_t arg;       ///< argument
        fp_t val;       ///< function value(loss at pos)
        fp_t deriv;     ///< derivative = grad^T * direc
        Mat<fp_t> pos;  ///< position
        Mat<fp_t> grad; ///< gradient

        FuncVal(fp_t argument, size_t rows, size_t cols)
            : arg(argument)
        {
            BMO_RESIZE(this->pos, rows, cols);
            BMO_RESIZE(this->grad, rows, cols);
        };

        void swap(FuncVal &other)
        {
            fp_t tmp;
            // swap arg and val
            tmp = this->arg;
            this->arg = other.arg;
            other.arg = tmp;
            tmp = this->val;
            this->val = other.val;
            other.val = tmp;
            // swap pos, grad and deriv
            BMO_SWAP(this->pos, other.pos);
            BMO_SWAP(this->grad, other.grad);
            tmp = this->deriv;
            this->deriv = other.deriv;
            other.deriv = tmp;
        };

        void update_val(
            LineSearch<fp_t>::Problem *prob,
            const Mat<fp_t> &x0,
            const Mat<fp_t> &d,
            const fp_t sgn_d)
        { // arg update but val and grad not
            this->pos = x0 + this->arg * sgn_d * d;
            this->val = prob->loss(this->pos);
            prob->grad(this->pos, this->grad);
            this->deriv = BMO_MAT_DOT_PROD(this->grad, d) * sgn_d;
        };

        void phi_to_psi(const fp_t g_test)
        {
            this->val = this->val - g_test * this->arg;
            this->deriv = this->deriv - g_test;
        };

        void psi_to_phi(const fp_t g_test)
        {
            this->val = this->val + g_test * this->arg;
            this->deriv = this->deriv + g_test;
        };
    };

    // minimizer of the cubic function that interpolates f(a), f'(a), f(b), f'(b) within the given interval.
    template <typename fp_t>
    OPTIM_INLINE fp_t find_ac(
        const FuncVal<fp_t> &f_a,
        const FuncVal<fp_t> &f_b)
    {
        using std::abs;
        using std::max;
        using std::sqrt;
        const fp_t theta = fp_t(3) * (f_a.val - f_b.val) / (f_b.arg - f_a.arg) + f_b.deriv + f_a.deriv;
        const fp_t s = max({abs(theta), abs(f_a.deriv), abs(f_b.deriv)});
        fp_t gamma = s * sqrt((theta / s) * (theta / s) - (f_a.deriv / s) * (f_b.deriv / s));
        if (f_b.arg > f_a.arg)
            gamma = -gamma;
        const fp_t p = gamma - f_b.deriv + theta;
        const fp_t q = gamma - f_b.deriv + gamma + f_a.deriv;
        const fp_t r = p / q;
        return f_b.arg + r * (f_a.arg - f_b.arg);
    }

    // minimizer of the cubic function that interpolates f(a), f'(a), f(b), f'(b) within the given interval.
    // specifically for case 3
    template <typename fp_t>
    OPTIM_INLINE fp_t
    find_ac(
        const FuncVal<fp_t> &f_a,
        const FuncVal<fp_t> &f_b,
        const fp_t step_min,
        const fp_t step_max)
    {
        using std::abs;
        using std::max;
        using std::sqrt;
        const fp_t theta = 3 * (f_a.val - f_b.val) / (f_b.arg - f_a.arg) + f_b.deriv + f_a.deriv;
        const fp_t s = max({abs(theta), abs(f_a.deriv), abs(f_b.deriv)});
        // The case gamma = 0 only arises if the cubic does not tend
        // to infinity in the direction of the step.
        fp_t gamma = s * sqrt(max((theta / s) * (theta / s) - (f_a.deriv / s) * (f_b.deriv / s), fp_t(0)));
        if (f_b.arg > f_a.arg)
            gamma = -gamma;
        const fp_t p = gamma - f_b.deriv + theta;
        const fp_t q = f_a.deriv - f_b.deriv + 2 * gamma;
        const fp_t r = p / q;
        if (r < 0 && gamma != 0)
            return f_b.arg + r * (f_a.arg - f_b.arg);
        else if (f_b.arg > f_a.arg)
            return step_max;
        else
            return step_min;
    }

    // minimizer of the quadratic function that interpolates f(a), f'(a), f(b) within the given interval
    template <typename fp_t>
    OPTIM_INLINE fp_t
    find_aq(
        const FuncVal<fp_t> &f_a,
        const FuncVal<fp_t> &f_b)
    {
        return f_a.arg +
               fp_t(0.5) * f_a.deriv /
                   ((f_a.val - f_b.val) / (f_b.arg - f_a.arg) + f_a.deriv) * (f_b.arg - f_a.arg);
    }

    // minimizer of the quadratic function that interpolates f'(a), f'(b) within the given interval
    // N.B. the function itself is undetermined since we miss information like f(a) or f(b); however the minimizer is well-defined
    template <typename fp_t>
    OPTIM_INLINE fp_t
    find_as(
        const FuncVal<fp_t> &f_a,
        const FuncVal<fp_t> &f_b)
    {
        return f_a.arg +
               f_a.deriv /
                   (f_a.deriv - f_b.deriv) *
                   (f_b.arg - f_a.arg);
    }

    template <typename fp_t>
    OPTIM_INLINE fp_t
    select_trial_value(
        bool &brackt,
        const FuncVal<fp_t> &f_l,
        const FuncVal<fp_t> &f_u,
        const FuncVal<fp_t> &f_t,
        const fp_t step_min,
        const fp_t step_max)
    {
        using std::abs;
        using std::max;
        using std::min;
        using std::sqrt;
        if (f_t.val > f_l.val)
        { // case 1
            brackt = true;
            const fp_t a_c = find_ac(f_l, f_t);
            const fp_t a_q = find_aq(f_l, f_t);
            if (abs(a_c - f_l.arg) < abs(a_q - f_l.arg))
                return a_c;
            else
                return (a_c + a_q) / 2;
        }
        // note that f_t.deriv * f_l.deriv may be overflow
        else if (std::signbit(f_t.deriv) !=
                 std::signbit(f_l.deriv))
        { // case 2
            brackt = true;
            const fp_t a_c = find_ac(f_l, f_t);
            const fp_t a_s = find_as(f_l, f_t);
            if (abs(a_c - f_t.arg) >= abs(a_s - f_t.arg))
                return a_c;
            else
                return a_s;
        }
        // case 3
        // The cubic step is computed only if the cubic tends to infinity
        // in the direction of the step or if the minimum of the cubic
        // is beyond stp. Otherwise the cubic step is defined to be the
        // secant step.
        else if (abs(f_t.deriv) <= abs(f_l.deriv))
        {
            fp_t trial_step;
            fp_t a_c = find_ac(f_l, f_t, step_min, step_max);
            fp_t a_s = find_as(f_l, f_t);
            if (brackt) // case 3, brackt
            {           // gamma not used, so we can use it to storage temp value
                // A minimizer has been bracketed. If the cubic step is
                // closer to stp than the secant step, the cubic step is
                // taken, otherwise the secant step is taken.
                if (abs(a_c - f_t.arg) < abs(a_s - f_t.arg))
                    trial_step = a_c;
                else
                    trial_step = a_s;
                if (f_t.arg > f_l.arg)
                    trial_step = min(f_t.arg + fp_t(0.66) * (f_u.arg - f_t.arg), trial_step);
                else
                    trial_step = max(f_t.arg + fp_t(0.66) * (f_u.arg - f_t.arg), trial_step);
                return trial_step;
            }
            else // case 3, but not brackt
            {
                // A minimizer has not been bracketed. If the cubic step is
                // farther from stp than the secant step, the cubic step is
                // taken, otherwise the secant step is taken.
                if (abs(a_c - f_t.arg) > abs(a_s - f_t.arg))
                    trial_step = a_c;
                else
                    trial_step = a_s;
                trial_step = min(step_max, trial_step);
                trial_step = max(step_min, trial_step);
                return trial_step;
            }
        }
        else // case 4
        {
            if (brackt)
                return find_ac(f_t, f_u);
            else if (f_t.arg > f_l.arg)
                return step_max;
            else
                return step_min;
        }
    }

} // namespace internal::MTLineSearch