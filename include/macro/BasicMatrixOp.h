#pragma once
#ifndef __BASIC_MATRIX_OPERATIONS_HPP__
#define __BASIC_MATRIX_OPERATIONS_HPP__

/*-------------------Get Data------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_GET_DATA(X) (X).data()
#define BMO_GET_SP_DATA (X) (X).valuePtr()
#elif defined(BMO_USE_ARMA)
#define BMO_GET_DATA(X) (X).memptr()
#define BMO_GET_SP_DATA(X) (X).values
#endif
/*------------------- N cols-------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_COLS(X) (X).cols()
#elif defined(BMO_USE_ARMA)
#define BMO_COLS(X) (X).n_cols
#endif
/*------------------- N rows-------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_ROWS(X) (X).rows()
#elif defined(BMO_USE_ARMA)
#define BMO_ROWS(X) (X).n_rows
#endif
/*------------------- Size  -------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_SIZE(X) (X).size()
#elif defined(BMO_USE_ARMA)
#define BMO_SIZE(X) (X).n_elem
#endif
/*------------------- Resize -------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_RESIZE(X, rows, cols) (X).resize(rows, cols)
#elif defined(BMO_USE_ARMA)
#define BMO_RESIZE(X, rows, cols) (X).resize(rows, cols)
#endif
/*------------------- Zero init -------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_INIT_ZERO(Mat, rows, cols) Mat::Zero(rows, cols)
#elif defined(BMO_USE_ARMA)
#define BMO_INIT_ZERO(Mat, rows, cols) Mat(rows, cols, arma::fill::zeros)
#endif
/*------------------- Set Zero -------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_SET_ZERO(X) (X).setZero()
#elif defined(BMO_USE_ARMA)
#define BMO_SET_ZERO(X) (X).zeros()
#endif
/*----------------- Rand init ------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_INIT_RAND(Mat, rows, cols) Mat::Random(rows, cols)
#elif defined(BMO_USE_ARMA)
#define BMO_INIT_RAND(Mat, rows, cols) Mat(rows, cols, arma::fill::randn)
#endif
/*------------------- Identity -------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_IDENTITY(Mat, rows, cols) Mat::Identity(rows, cols)
#elif defined(BMO_USE_ARMA)
#define BMO_IDENTITY(Mat, rows, cols) Mat(rows, cols, arma::fill::eye)
#endif
/*--------------------As Diagnol-----------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_AS_DIAG(X) (X).asDiagonal()
#elif defined(BMO_USE_ARMA)
#define BMO_AS_DIAG(X) arma::diagmat(X)
#endif
/*--------------------Array  div------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_ARRAY_DIV(X, Y) ((X).array() / (Y).array()).matrix()
#elif defined(BMO_USE_ARMA)
#define BMO_ARRAY_DIV(X, Y) (X) / (Y)
#endif
/*--------------------Array  inv------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_ARRAY_INV(X) (X).array().inverse().matrix()
#elif defined(BMO_USE_ARMA)
#define BMO_ARRAY_INV(X) 1 / (X)
#endif
/*--------------------Array  mul------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_ARRAY_MUL(X, Y) ((X).array() * (Y).array()).matrix()
#elif defined(BMO_USE_ARMA)
#define BMO_ARRAY_MUL(X, Y) (X) % (Y)
#endif
/*--------------------Array  sqrt------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_ARRAY_SQRT(X) (X).array().sqrt().matrix()
#elif defined(BMO_USE_ARMA)
#define BMO_ARRAY_SQRT(X) arma::sqrt(X)
#endif
/*--------------------Array  exp------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_ARRAY_EXP(X) (X).array().exp().matrix()
#elif defined(BMO_USE_ARMA)
#define BMO_ARRAY_EXP(X) arma::exp(X)
#endif
/*--------------------Array  log------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_ARRAY_LOG(X) (X).array().log().matrix()
#elif defined(BMO_USE_ARMA)
#define BMO_ARRAY_LOG(X) arma::log(X)
#endif
/*--------------------Array + Scalar------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_ARRAY_ADD_SCALAR(X, Y) ((X).array() + (Y)).matrix()
#elif defined(BMO_USE_ARMA)
#define BMO_ARRAY_ADD_SCALAR(X, Y) (X) + (Y)
#endif
/*-----------------Mat Map init------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_INIT_MAP_MAT(M, X, n_rows, n_cols) M(X.data(), n_rows, n_cols)
#elif defined(BMO_USE_ARMA)
#define BMO_INIT_MAP_MAT(M, X, n_rows, n_cols) M(X.memptr(), n_rows, n_cols, false, true)
#endif
// /*------------------Col Map init-----------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_INIT_MAP_COL(M, X, n) M(X.data(), n)
#elif defined(BMO_USE_ARMA)
#define BMO_INIT_MAP_COL(M, X, n) M(X.memptr(), n, false, true)
#endif
/*-------------------  Swap  -------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_SWAP(X, Y) (X).swap(Y)
#elif defined(BMO_USE_ARMA)
#define BMO_SWAP(X, Y) (X).swap(Y)
#endif
/*------------------ Transpose ------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_TRANSPOSE(X) (X).transpose()
#elif defined(BMO_USE_ARMA)
#define BMO_TRANSPOSE(X) (X).t()
#endif
/*------------------ Inverse ------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_LDLT_SOLVE(X, Y) (X).ldlt().solve(Y)
#define BMO_INVERSE(X) (X).inverse()
#elif defined(BMO_USE_ARMA)
#define BMO_LDLT_SOLVE(X, Y) arma::solve(X, Y, arma::solve_opts::fast)
#define BMO_INVERSE(X) arma::inv(X)
#endif
/*-------------------- Sum --------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_SUM(X) (X).array().sum()
#elif defined(BMO_USE_ARMA)
#define BMO_SUM(X) arma::accu(X)
#endif
/*------------------ Abs ------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_ABS(X) (X).array().abs().matrix()
#elif defined(BMO_USE_ARMA)
#define BMO_ABS(X) arma::abs(X)
#endif
/*------------------ Sign ------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_SIGN(X) (X).array().sign()
#elif defined(BMO_USE_ARMA)
#define BMO_SIGN(X) arma::sign(X)
#endif
/*------------------ Determinant ------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_DET(X) (X).determinant()
#elif defined(BMO_USE_ARMA)
#define BMO_DET(X) arma::det(X)
#endif
/*------------------ Inner Product ------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_DOT_PROD(X, Y) (X).dot(Y)
#elif defined(BMO_USE_ARMA)
#define BMO_DOT_PROD(X, Y) arma::dot(X, Y)
#endif
#if defined(BMO_USE_EIGEN)
#define BMO_MAT_DOT_PROD(X, Y) ((X).array() * (Y).array()).sum()
#elif defined(BMO_USE_ARMA)
#define BMO_MAT_DOT_PROD(X, Y) arma::dot(X, Y)
#endif
/*------------------ Square (Frob)Norm ------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_SQUARE_NORM(X) (X).squaredNorm()
#elif defined(BMO_USE_ARMA)
#define BMO_SQUARE_NORM(X) arma::dot(X, X)
#endif
/*------------------ Inf Norm ------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_INF_NORM(X) (X).lpNorm<Infinity>()
#elif defined(BMO_USE_ARMA)
#define BMO_INF_NORM(X) arma::norm(X, "inf")
#endif
/*------------------ Lp Norm ------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_LP_NORM(X, p) (X).lpNorm<p>()
#elif defined(BMO_USE_ARMA)
#define BMO_LP_NORM(X, p) arma::norm(X, p)
#endif
/*------------------(Frob) Norm ------------------*/
#if defined(BMO_USE_EIGEN)
#define BMO_FRO_NORM(X) (X).norm()
#elif defined(BMO_USE_ARMA)
#define BMO_FRO_NORM(X) arma::norm(X)
#endif
#endif