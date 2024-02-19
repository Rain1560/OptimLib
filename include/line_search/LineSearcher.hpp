#pragma once
#ifndef _OPTIM_LINE_SEARCH_BASE_HPP__
#define _OPTIM_LINE_SEARCH_BASE_HPP__

#include "base/BaseSolver.hpp"

namespace optim
{
    template <typename fp_t>
    struct LineSearcher
    {
        using Problem = GradProblem<fp_t>;
        using Constant = OptimConst<fp_t>;

    protected:
        Problem *prob;

    public:
        fp_t min_step = Constant::sqrt_eps;
        fp_t max_step = fp_t(1e2);

        virtual bool success() const = 0;

        virtual void init(Problem *, fp_t) = 0;

        virtual fp_t line_search(
            fp_t &step, const Mat<fp_t> &direc,
            const Mat<fp_t> &in_x, Mat<fp_t> &out_x,
            const fp_t &in_loss, fp_t &out_loss,
            const Mat<fp_t> &in_grad, Mat<fp_t> &out_grad) = 0;

    };
}

#endif