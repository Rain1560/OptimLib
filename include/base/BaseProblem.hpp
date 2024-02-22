#pragma once
#ifndef _OPTIMLIB_BASE_BASE_PROBLEM_HPP_
#define _OPTIMLIB_BASE_BASE_PROBLEM_HPP_

#include "macro/macro.h"

namespace optim
{

    template <typename fp_t>
    struct OptimConst
    {
        static constexpr const fp_t eps = std::numeric_limits<fp_t>::epsilon();
        static constexpr const fp_t sqrt_eps = std::sqrt(eps);
        static constexpr const fp_t inf = std::numeric_limits<fp_t>::infinity();
        static constexpr const fp_t nan = std::numeric_limits<fp_t>::quiet_NaN();
    };

    template <typename fp_t>
    struct BaseProblem
    {
        virtual fp_t loss(
            const Mat<fp_t> &in_x) = 0;
    };

    template <typename fp_t>
    struct StoGradProblem
        : public BaseProblem<fp_t>
    {
        virtual void grad(
            const Mat<fp_t> &in_x,
            const std::vector<int> &slice,
            Mat<fp_t> &out_x) = 0;
    };

    template <typename fp_t>
    struct GradProblem
        : public BaseProblem<fp_t>
    {
        virtual void grad(
            const Mat<fp_t> &in_x,
            Mat<fp_t> &out_x) = 0;
    };

    template <typename fp_t>
    struct HessProblem
        : public GradProblem<fp_t>
    {
        virtual void hess(
            const Mat<fp_t> &in_x,
            Mat<fp_t> &out_x) = 0;
    };

    namespace internal
    {
        template <typename fp_t,
                  template <typename> class Problem>
        struct ProxOperator
            : virtual public Problem<fp_t>
        {
            static_assert(std::is_base_of_v<BaseProblem<fp_t>, Problem<fp_t>>,
                          "ProxOperator requires a BaseProblem");
            virtual fp_t sm_loss(const Mat<fp_t> &x) = 0;

            virtual fp_t nsm_loss(const Mat<fp_t> &x) = 0;

            fp_t loss(const Mat<fp_t> &x) override
            {
                return sm_loss(x) + nsm_loss(x);
            }

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
            using type = ProxOperator<fp_t, Problem>;
        };
    }

    template <typename fp_t,
              template <typename> class Problem,
              bool use_prox = false>
    using ProxWrapper =
        internal::ProxWrapper<
            fp_t, Problem, use_prox>::type;
}

#endif