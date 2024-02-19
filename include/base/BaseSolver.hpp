#pragma once
#ifndef _OPTIMLIB_BASE_BASE_SOLVER_HPP_
#define _OPTIMLIB_BASE_BASE_SOLVER_HPP_

#include "BaseProblem.hpp"
#include "misc/logger.hpp"
#include "Recorder.hpp"

namespace optim
{
    /// @brief Base class for all solvers.
    template <typename fp_t>
    struct BaseSolver
    {
    protected:
        int iter;

    public:
        int n_iter() const { return iter; };
        virtual fp_t solve(Mat<fp_t> &) = 0;
    };
};

#endif