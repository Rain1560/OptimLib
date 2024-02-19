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
        : virtual public BaseProblem<fp_t>
    {
        virtual void grad(
            const Mat<fp_t> &in_x,
            const std::vector<int> &slice,
            Mat<fp_t> &out_x) = 0;
    };

    template <typename fp_t>
    struct GradProblem
        : virtual public BaseProblem<fp_t>
    {
        virtual void grad(
            const Mat<fp_t> &in_x,
            Mat<fp_t> &out_x) = 0;
    };

    template <typename fp_t>
    struct HessProblem
        : virtual public GradProblem<fp_t>
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
        {
            virtual std::enable_if_t<
                std::is_base_of_v<
                    BaseProblem<fp_t>, Problem<fp_t>>,
                fp_t>
            nsm_loss(const Mat<fp_t> &in_x) = 0;

            virtual std::enable_if_t<
                std::is_base_of_v<
                    GradProblem<fp_t>, Problem<fp_t>>,
                void>
            prox(const Mat<fp_t> &in_x,
                 Mat<fp_t> &out_x) = 0;
        };

        template <typename fp_t,
                  template <typename> class Problem,
                  bool use_prox = false>
        struct ProxWrapper
        {
            using type = Problem<fp_t>;
        };

        template <typename fp_t,
                  template <typename> class Problem>
        struct ProxWrapper<fp_t, Problem, true>
        {
            using type = internal::ProxOperator<fp_t, Problem>;
        };
    }

    template <typename fp_t,
              template <typename> class Problem,
              bool use_prox = false>
    using ProxWrapper = typename internal::ProxWrapper<fp_t, Problem, use_prox>::type;
}

#endif