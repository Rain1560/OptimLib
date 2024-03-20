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

#if __cplusplus >= 202002L
#define OPTIM_LIKELY [[likely]]
#define OPTIM_UNLIKELY [[unlikely]]
#else
#define OPTIM_LIKELY
#define OPTIM_UNLIKELY
#endif

#ifdef OPTIM_USE_EIGEN
#define BMO_USE_EIGEN
#include <Eigen/Eigen>
#elif OPTIM_USE_ARMA
#define BMO_USE_ARMA
#include <armadillo>
#else
#error "Eigen or Arma is required for this library. Add -DOPTIM_USE_EIGEN to compiler flags."
#endif

namespace optim
{
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
    template <typename fp_t>
    using Mat = arma::Mat<fp_t>;
    template <typename fp_t>
    using Row = arma::Row<fp_t>;
    template <typename fp_t>
    using Col = arma::Col<fp_t>;
    template <typename fp_t>
    using MapCol = arma::Col<fp_t>;
    template <typename fp_t>
    using MapRow = arma::Row<fp_t>;
    template <typename fp_t>
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