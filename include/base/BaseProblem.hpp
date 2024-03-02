#pragma once
#ifndef _OPTIMLIB_BASE_BASE_PROBLEM_HPP_
#define _OPTIMLIB_BASE_BASE_PROBLEM_HPP_

#include "macro/macro.h"

namespace optim
{
    /// @cond
    template <typename fp_t>
    struct OptimConst
    {
        static constexpr const fp_t eps = std::numeric_limits<fp_t>::epsilon();
        static constexpr const fp_t sqrt_eps = std::sqrt(eps);
        static constexpr const fp_t inf = std::numeric_limits<fp_t>::infinity();
        static constexpr const fp_t nan = std::numeric_limits<fp_t>::quiet_NaN();
    };
    /// @endcond

    /// @brief Base problem interface
    template <typename fp_t>
    struct BaseProblem
    {
        /// @brief loss function
        /// @param in_x the point to evaluate
        virtual fp_t loss(
            const Mat<fp_t> &in_x) = 0;
    };

    /// @brief Gradient problem interface
    template <typename fp_t>
    struct GradProblem
        : public BaseProblem<fp_t>
    {
        /// @brief gradient function
        /// @param in_x the point to evaluate
        /// @param out_x the output gradient
        virtual void grad(
            const Mat<fp_t> &in_x,
            Mat<fp_t> &out_x) = 0;
    };

    template <typename fp_t>
    struct HessProblem
        : public GradProblem<fp_t>
    {
        /// @brief Hessian function
        /// @details the in_x may not be a vector, so you need to treat a matrix as a vector. i.e. your Hessian matrix should be matched with resize(in_x,size(in_x),1).
        /// @param in_x  the point to evaluate
        /// @param out_x  the output Hessian
        virtual void hess(
            const Mat<fp_t> &in_x,
            Mat<fp_t> &out_x) = 0;
    };

    namespace internal
    {
        /// @brief Proximal operator interface
        /// @details when using proximal operator, you need to implement the proximal operator and the loss function(both smooth and non-smooth parts).
        template <typename fp_t,
                  template <typename> class Problem>
        struct ProxOperator
            : virtual public Problem<fp_t>
        {
            static_assert(std::is_base_of_v<BaseProblem<fp_t>, Problem<fp_t>>,
                          "ProxOperator requires a BaseProblem");
            /// @brief smooth part of the loss function
            /// @param x the point to evaluate
            /// @return  the smooth part of the loss
            virtual fp_t sm_loss(const Mat<fp_t> &x) = 0;

            /// @brief non-smooth part of the loss function
            /// @param x the point to evaluate
            /// @return  the non-smooth part of the loss
            virtual fp_t nsm_loss(const Mat<fp_t> &x) = 0;

            fp_t loss(const Mat<fp_t> &x) override
            {
                return sm_loss(x) + nsm_loss(x);
            }

            /// @brief proximal operator
            /// @param step the step size
            /// @param in_x the input point
            /// @param out_x the output point
            virtual std::enable_if_t<
                std::is_base_of_v<
                    GradProblem<fp_t>, Problem<fp_t>>,
                void>
            prox(fp_t step,
                 const Mat<fp_t> &in_x,
                 Mat<fp_t> &out_x) = 0;
        };

        template <typename fp_t,
                  template <typename> class Problem,
                  bool use_prox = false>
        struct ProxWrapper
        {
            static_assert(std::is_base_of_v<BaseProblem<fp_t>, Problem<fp_t>>,
                          "ProxWrapper requires a BaseProblem");
            using type = Problem<fp_t>;
        };

        template <typename fp_t,
                  template <typename> class Problem>
        struct ProxWrapper<fp_t, Problem, true>
        {
            static_assert(std::is_base_of_v<BaseProblem<fp_t>, Problem<fp_t>>,
                          "ProxWrapper requires a BaseProblem");
            using type = typename internal::ProxOperator<fp_t, Problem>;
        };
    }

    template <typename fp_t,
              template <typename> class Problem,
              bool use_prox = false>
    using ProxWrapper = typename internal::ProxWrapper<
        fp_t, Problem, use_prox>::type;
}

#endif