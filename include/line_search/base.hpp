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
    /// @cond
    template <typename fp_t, bool use_prox>
    struct LineSearchArgs;
    /// @endcond

    /// @brief Line search arguments
    /// @tparam fp_t floating-point type
    template <typename fp_t>
    struct LineSearchArgs<fp_t, false>
    {
        using Problem = GradProblem<fp_t>;

        fp_t step; ///< step size

        fp_t prev_loss; ///< previous loss
        fp_t cur_loss;  ///< current loss

        Mat<fp_t> direction; ///< step direction

        Mat<fp_t> prev_x; ///< previous point
        Mat<fp_t> cur_x;  ///< current point

        Mat<fp_t> prev_grad; ///< previous gradient
        Mat<fp_t> cur_grad;  ///< current gradient

        /// @brief  malloc the memory for the line search arguments and assign x to the current point
        /// @param x initial point
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

        /// @brief Step forward to the next point
        OPTIM_STRONG_INLINE
        void step_forward(Problem *)
        {
            cur_x = prev_x + step * direction;
        }

        /// @brief Update the current loss
        OPTIM_STRONG_INLINE
        void update_cur_loss(Problem *prob)
        {
            cur_loss = prob->loss(cur_x);
        }

        /// @brief Update the current gradient
        OPTIM_STRONG_INLINE
        void update_cur_grad(Problem *prob)
        {
            prob->grad(cur_x, cur_grad);
        }

        /// @brief Flush the current state to the previous state
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

        fp_t step; ///< step size

        fp_t prev_loss; ///< previous loss
        fp_t cur_loss;  ///< current loss

        Mat<fp_t> direction; ///< step direction

        Mat<fp_t> prev_x; ///< previous point
        Mat<fp_t> cur_x;  ///< current point

        Mat<fp_t> prev_grad; ///< previous gradient
        Mat<fp_t> cur_grad;  ///< current gradient

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
            cur_x = x;
            BMO_RESIZE(prev_x, n, m);
            BMO_RESIZE(prev_grad, n, m);
            BMO_RESIZE(cur_grad, n, m);
            BMO_RESIZE(direction, n, m);
            BMO_RESIZE(prev_grad_map, n, m);
            BMO_RESIZE(cur_grad_map, n, m);
            BMO_RESIZE(tmp, n, m);
        }

        /// @brief  Step forward to the next point
        OPTIM_STRONG_INLINE void
        step_forward(Problem *prob)
        {
            cur_x = prev_x + step * direction;
            prob->prox(step, cur_x, tmp);
            BMO_SWAP(cur_x, tmp);
        }

        /// @brief Update the current loss
        OPTIM_STRONG_INLINE void
        update_cur_loss(Problem *prob)
        {
            cur_sm_loss = prob->sm_loss(cur_x);
            cur_nsm_loss = prob->nsm_loss(cur_x);
            cur_loss = cur_sm_loss + cur_nsm_loss;
        }

        /// @brief Update the current gradient
        OPTIM_STRONG_INLINE void
        update_cur_grad(Problem *prob)
        {
            prob->grad(cur_x, cur_grad);
            cur_grad_map = cur_x - cur_grad;
            prob->prox(1, cur_grad_map, tmp);
            cur_grad_map = (cur_x - tmp) / step;
        }

        /// @brief Update the previous gradient mapping
        OPTIM_STRONG_INLINE void
        update_prev_grad_map(Problem *prob)
        {
            prev_grad_map = prev_x - step * prev_grad;
            prob->prox(step, prev_grad_map, tmp);
            prev_grad_map = (prev_x - tmp) / step;
        }

        /// @brief Update the current gradient mapping
        OPTIM_STRONG_INLINE void
        update_cur_grad_map(Problem *prob)
        {
            cur_grad_map = cur_x - step * cur_grad;
            prob->prox(step, cur_grad_map, tmp);
            cur_grad_map = (cur_x - tmp) / step;
        }

        /// @brief Flush the current state to the previous state
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
        Problem *prob;

    public:
        fp_t min_step = Constant::sqrt_eps; ///< minimum step size
        fp_t max_step = fp_t(1e2);          ///< maximum step size

        /// @brief whether the line search is successful
        /// @return true if successful
        virtual bool success() const { return true; }

        /// @brief initialize the line search
        /// @param p problem
        /// @param args Line Search arguments
        virtual void init(Problem *p, Args &arg)
        {
            this->prob = p;
        }

        /// @brief line search with the given arguments
        /// @details All LineSearch classes will using previous loss and grad to find the proper step which meats different requirements. After finishing line search, it will update current loss and gradient at the new point.
        /// @param arg Line Search arguments
        virtual void line_search(Args &arg)
        {
            arg.step_forward(prob);
            arg.update_cur_grad(prob);
        }

        virtual ~LineSearch() = default;
    };

    template <typename fp_t, bool use_prox = false>
    struct LSBaseSolver : public BaseSolver<fp_t>
    {
        using LineSearchImp = LineSearch<fp_t, use_prox>;

    protected:
        std::shared_ptr<LineSearchImp> ls;

    public:
        void reset_ls(std::shared_ptr<LineSearchImp> ls)
        {
            this->ls = ls;
        }
    };
}

#endif