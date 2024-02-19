#pragma once
#ifndef _OPTIMLIB_MACRO_MACRO_H_
#define _OPTIMLIB_MACRO_MACRO_H_

#include <bits/stdc++.h>

#if defined(__GNUC__) || defined(__GNUG__) || defined(__clang__)
#define OPTIM_INLINE inline
#define OPTIM_STRONG_INLINE __attribute__((always_inline))
#elif defined(_MSC_VER)
#define OPTIM_INLINE
#define OPTIM_STRONG_INLINE __forceinline
#else
#define OPTIM_INLINE
#define OPTIM_STRONG_INLINE
#endif

#ifdef OPTIM_USE_EIGEN
#include <Eigen/Eigen>
#elif OPTIM_USE_ARMA
#include <armadillo>
#else
#error "Eigen or Arma is required for this library. Add -DOPTIM_USE_EIGEN to compiler flags."
#endif

namespace optim
{
    // #ifndef OPTIM_FLOAT_TYPE
    //     using fp_t = double;
    // #else
    //     static_assert(std::is_floating_point<OPTIM_FLOAT_TYPE>::value,
    //                   "OPTIM_FLOAT_TYPE must be a floating point type.");
    //     using fp_t = OPTIM_FLOAT_TYPE;
    // #endif
    //     // 1e-8 for float, 1e-22 for double
    //     constexpr const fp_t eps = std::numeric_limits<fp_t>::epsilon();
    //     // when norm close to sqrt_eps, it should be stop.
    //     constexpr const fp_t sqrt_eps = std::sqrt(eps);
    //     constexpr const fp_t inf = std::numeric_limits<fp_t>::infinity();
    //     constexpr const fp_t nan = std::numeric_limits<fp_t>::quiet_NaN();

    // #define OPTIM_SMALL_NUM 1e-6

#if defined(OPTIM_USE_EIGEN)
    using Index = Eigen::Index;
    template <typename fp_t> //= double
    using Mat = Eigen::Matrix<fp_t, Eigen::Dynamic, Eigen::Dynamic>;
    template <typename fp_t> //= double
    using Row = Eigen::Matrix<fp_t, 1, Eigen::Dynamic>;
    template <typename fp_t> // = double
    using Col = Eigen::Matrix<fp_t, Eigen::Dynamic, 1>;
    template <typename fp_t> //= double
    using MapCol = Eigen::Map<Col<fp_t>>;
    template <typename fp_t> //= double
    using MapRow = Eigen::Map<Row<fp_t>>;
    template <typename fp_t> //= double
    using MapMat = Eigen::Map<Mat<fp_t>>;
#elif defined(OPTIM_USE_ARMA)
    // TODO: using template alias
    using Index = arma::uword;
    using Mat = arma::Mat<fp_t>;
    using Row = arma::Row<fp_t>;
    using Col = arma::Col<fp_t>;
    using MapCol = arma::Col<fp_t>;
    using MapRow = arma::Row<fp_t>;
    using MapMat = arma::Mat<fp_t>;
#endif
};

#ifndef OPTIM_NO_DEBUG
#ifndef OPTIM_NO_ASSERT
#define OPTIM_NO_ASSERT
#endif
#endif

#include "BasicMatrixOp.h"
#include "optim_assert.h"

#endif