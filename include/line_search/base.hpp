#pragma once
#ifndef _OPTIM_LINE_SEARCH_BASE_HPP__
#define _OPTIM_LINE_SEARCH_BASE_HPP__

#include "base/BaseSolver.hpp"

/**
 * \page line_search Line Search
 * \par
 * Below Line Search algorithms are trying to find a step \f$ t_k \f$ that satisfies different conditions. To make sure the optimization algorithms are converging, a proper Line Search algorithm is significant.
 * \par
 * We define 2 line search functions: \f$ \phi(\alpha) = f(x + \alpha d) \f$ and auxiliary function \f$ \psi(\alpha) = \phi(\alpha) - \phi(0) - \alpha \phi'(0) \f$.
 *
 * # Armijo Line Search
 * \copydoc ArmijoLineSearch
 * # Zhang-Hager Line Search
 * \copydoc ZHLineSearch
 * # More-Thuente Line Search
 * \copydoc MTLineSearch
 */

namespace optim
{
    template <typename fp_t>
    struct BaseLineSearchArgs
    {
        fp_t step; ///< step size

        fp_t prev_loss; ///< previous loss
        fp_t cur_loss;  ///< current loss

        Mat<fp_t> direction; ///< step direction

        Mat<fp_t> prev_x; ///< previous point
        Mat<fp_t> cur_x;  ///< current point

        Mat<fp_t> prev_grad; ///< previous gradient
        Mat<fp_t> cur_grad;  ///< current gradient
    };

    /// @cond
    template <typename fp_t, bool use_prox>
    struct LineSearchArgs;
    /// @endcond

    /// @brief Line search arguments
    /// @tparam fp_t floating-point type
    template <typename fp_t>
    struct LineSearchArgs<fp_t, false> final
        : public BaseLineSearchArgs<fp_t>
    {
        using Problem = GradProblem<fp_t>;

        /// @brief  malloc the memory for the line search arguments and assign x to the current point
        /// @param x initial point
        LineSearchArgs(Mat<fp_t> &x)
        {
            const Index n = BMO_ROWS(x),
                        m = BMO_COLS(x);
            this->cur_x = x;
            BMO_RESIZE(this->prev_x, n, m);
            BMO_RESIZE(this->prev_grad, n, m);
            BMO_RESIZE(this->cur_grad, n, m);
            BMO_RESIZE(this->direction, n, m);
        }

        /// @brief Step forward to the next point
        OPTIM_STRONG_INLINE
        void step_forward(Problem *)
        {
            this->cur_x = this->prev_x + this->step * this->direction;
        }

        /// @brief Update the current loss
        OPTIM_STRONG_INLINE
        void update_cur_loss(Problem *prob)
        {
            this->cur_loss = prob->loss(this->cur_x);
        }

        /// @brief Update the current gradient
        OPTIM_STRONG_INLINE
        void update_cur_grad(Problem *prob)
        {
            prob->grad(this->cur_x, this->cur_grad);
        }

        /// @brief Flush the current state to the previous state
        OPTIM_STRONG_INLINE
        void flush()
        {
            BMO_SWAP(this->prev_x, this->cur_x);
            BMO_SWAP(this->prev_grad, this->cur_grad);
            this->prev_loss = this->cur_loss;
        }
    };

    template <typename fp_t>
    struct LineSearchArgs<fp_t, true> final
        : public BaseLineSearchArgs<fp_t>
    {
        using Problem = internal::ProxOperator<fp_t, GradProblem>;

        Mat<fp_t> prev_grad_map; ///< previous gradient mapping
        Mat<fp_t> cur_grad_map;  ///< current gradient mapping

        fp_t prev_sm_loss;  ///< previous smooth loss
        fp_t prev_nsm_loss; ///< previous non-smooth loss

        fp_t cur_sm_loss;  ///< current smooth loss
        fp_t cur_nsm_loss; ///< current non-smooth loss

        Mat<fp_t> tmp; ///< storage tmp value

    public:
        /// @brief malloc the memory for the line search arguments and assign x to the current point
        /// @param x initial point
        LineSearchArgs(Mat<fp_t> &x)
        {
            const Index n = BMO_ROWS(x),
                        m = BMO_COLS(x);
            this->cur_x = x;
            BMO_RESIZE(this->prev_x, n, m);
            BMO_RESIZE(this->prev_grad, n, m);
            BMO_RESIZE(this->cur_grad, n, m);
            BMO_RESIZE(this->direction, n, m);
            BMO_RESIZE(prev_grad_map, n, m);
            BMO_RESIZE(cur_grad_map, n, m);
            BMO_RESIZE(tmp, n, m);
        }

        /// @brief  Step forward to the next point
        OPTIM_STRONG_INLINE void
        step_forward(Problem *prob)
        {
            this->cur_x = this->prev_x + this->step * this->direction;
            prob->prox(this->step, this->cur_x, tmp);
            BMO_SWAP(this->cur_x, tmp);
        }

        /// @brief Update the current loss
        OPTIM_STRONG_INLINE void
        update_cur_loss(Problem *prob)
        {
            cur_sm_loss = prob->sm_loss(this->cur_x);
            cur_nsm_loss = prob->nsm_loss(this->cur_x);
            this->cur_loss = cur_sm_loss + cur_nsm_loss;
        }

        /// @brief Update the current gradient
        OPTIM_STRONG_INLINE void
        update_cur_grad(Problem *prob)
        {
            prob->grad(this->cur_x, this->cur_grad);
            this->cur_grad_map = this->cur_x - this->cur_grad;
            prob->prox(1, this->cur_grad_map, this->tmp);
            this->cur_grad_map = (this->cur_x - this->tmp) / this->step;
        }

        /// @brief Update the previous gradient mapping
        OPTIM_STRONG_INLINE void
        update_prev_grad_map(Problem *prob)
        {
            this->prev_grad_map = this->prev_x - this->step * this->prev_grad;
            prob->prox(this->step, this->prev_grad_map, this->tmp);
            this->prev_grad_map = (this->prev_x - this->tmp) / this->step;
        }

        /// @brief Update the current gradient mapping
        OPTIM_STRONG_INLINE void
        update_cur_grad_map(Problem *prob)
        {
            cur_grad_map = this->cur_x - this->step * this->cur_grad;
            prob->prox(this->step, cur_grad_map, tmp);
            cur_grad_map = (this->cur_x - tmp) / this->step;
        }

        /// @brief Flush the current state to the previous state
        OPTIM_STRONG_INLINE void
        flush()
        {
            BMO_SWAP(this->prev_x, this->cur_x);
            BMO_SWAP(this->prev_grad, this->cur_grad);
            BMO_SWAP(prev_grad_map, cur_grad_map);
            prev_sm_loss = cur_sm_loss;
            prev_nsm_loss = cur_nsm_loss;
            this->prev_loss = this->cur_loss;
        }
    };

    /// @brief  Line search base class
    /// @details step forward without any line search.
    /// @tparam fp_t floating-point type
    /// @tparam use_prox whether to use proximal operator
    template <typename fp_t, bool use_prox = false>
    struct LineSearch
    {
        using Problem = ProxWrapper<
            fp_t, GradProblem, use_prox>;
        using Constant = OptimConst<fp_t>;
        using Args = LineSearchArgs<fp_t, use_prox>;

    protected:
        std::shared_ptr<Problem> prob;

    public:
        bool update_cur_grad = true;

        fp_t min_step = Constant::sqrt_eps; ///< minimum step size
        fp_t max_step = fp_t(1e2);          ///< maximum step size

        /// @brief whether the line search is successful
        /// @return true if successful
        virtual bool success() const { return true; }

        /// @brief initialize the line search
        /// @param p problem
        /// @param args Line Search arguments
        virtual void init(
            std::shared_ptr<Problem> p, Args &arg)
        {
            this->prob = p;
        }

        /// @brief line search with the given arguments
        /// @details Require: step, prev_x, direction, prev_loss, prev_grad. Line search will update cur_x, cur_loss. If you want line search to update cur_grad(and cur_grad_map in nsm problem), you should set `update_cur_grad` to true.
        /// @param arg Line Search arguments
        virtual void line_search(Args &arg)
        {
            arg.step_forward(prob.get());
            arg.update_cur_loss(prob.get());
            arg.update_cur_grad(prob.get());
        }

        virtual ~LineSearch() = default;
    };

    template <typename fp_t, bool use_prox = false>
    struct LSBaseSolver : public BaseSolver<fp_t>
    {
        using LineSearchImp = LineSearch<fp_t, use_prox>;

        std::shared_ptr<LineSearchImp> ls;
    };
}

#endif