#pragma once
#ifndef _OPTIMLIB_MACRO_MATRIX_OP_H_
#define _OPTIMLIB_MACRO_MATRIX_OP_H_

#include "macro.h"
/*------------------- N cols-------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_COLS(X) (X).cols()
#elif OPTIM_USE_ARMA
#define BMO_COLS(X) (X).n_cols
#endif
/*------------------- N rows-------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_ROWS(X) (X).rows()
#elif OPTIM_USE_ARMA
#define BMO_ROWS(X) (X).n_rows
#endif
/*------------------- Size  -------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_SIZE(X) (X).size()
#elif OPTIM_USE_ARMA
#define BMO_SIZE(X) (X).n_elem
#endif
/*------------------- Resize -------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_RESIZE(X, rows, cols) (X).resize(rows, cols)
#elif OPTIM_USE_ARMA
#define BMO_RESIZE(X, rows, cols) (X).resize(rows, cols)
#endif
/*------------------- Zero init -------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_INIT_ZERO(Mat, rows, cols) Mat::Zero(rows, cols)
#elif OPTIM_USE_ARMA
#define BMO_INIT_ZERO(Mat, rows, cols) Mat(rows, cols, arma::fill::zeros)
#endif
/*------------------- Set Zero -------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_SET_ZERO(X) (X).setZero()
#elif OPTIM_USE_ARMA
#define BMO_SET_ZERO(X) (X).zeros()
#endif
/*----------------- Rand init ------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_INIT_RAND(Mat, rows, cols) Mat::Random(rows, cols)
#elif OPTIM_USE_ARMA
#define BMO_INIT_RAND(Mat, rows, cols) Mat(rows, cols, arma::fill::randu)
#endif
/*------------------- Identity -------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_IDENTITY(Mat, rows, cols) Mat::Identity(rows, cols)
#elif OPTIM_USE_ARMA
#define BMO_IDENTITY(Mat, rows, cols) Mat(rows, cols, arma::fill::eye)
#endif
/*--------------------Array  div------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_ARRAY_DIV(X, Y) (X).array() / (Y).array()
#elif OPTIM_USE_ARMA
#define BMO_ARRAY_DIV(X, Y) (X) / (Y)
#endif
/*--------------------Array  inv------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_ARRAY_INV(X) (X).array().inverse().matrix()
#elif OPTIM_USE_ARMA
#define BMO_ARRAY_INV(X) arma::inv(X)
#endif
/*--------------------Array  mul------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_ARRAY_MUL(X, Y) ((X).array() * (Y).array()).matrix()
#elif OPTIM_USE_ARMA
#define BMO_ARRAY_MUL(X, Y) (X) % (Y)
#endif
/*--------------------Array  sqrt------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_ARRAY_SQRT(X) (X).array().sqrt().matrix()
#elif OPTIM_USE_ARMA
#define BMO_ARRAY_SQRT(X) arma::sqrt(X)
#endif
/*--------------------Array  exp------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_ARRAY_EXP(X) (X).array().exp().matrix()
#elif OPTIM_USE_ARMA
#define BMO_ARRAY_EXP(X) arma::exp(X)
#endif
/*--------------------Array + Scalar------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_ARRAY_ADD_SCALAR(X, Y) ((X).array() + (Y)).matrix()
#elif OPTIM_USE_ARMA
#define BMO_ARRAY_ADD_SCALAR(X, Y) (X) + (Y)
#endif
/*-----------------Mat Map init------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_INIT_MAP_MAT(M, X, n_rows, n_cols) M(X.data(), n_rows, n_cols)
#elif OPTIM_USE_ARMA
#define BMO_INIT_MAP_MAT(M, X, n_rows, n_cols) M(X.memptr(), n_rows, n_cols, false, true)
#endif
// /*------------------Col Map init-----------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_INIT_MAP_COL(M, X, n) M(X.data(), n)
#elif OPTIM_USE_ARMA
#define BMO_INIT_MAP_COL(M, X, n) M(X.memptr(), n, false, true)
#endif
/*-------------------  Swap  -------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_SWAP(X, Y) (X).swap(Y)
#elif OPTIM_USE_ARMA
#define BMO_SWAP(X, Y) (X).swap(Y)
#endif
/*------------------ Transpose ------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_TRANSPOSE(X) (X).transpose()
#elif OPTIM_USE_ARMA
#define BMO_TRANSPOSE(X) (X).t()
#endif
/*------------------ Inverse ------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_INVERSE(X) (X).inverse()
#elif OPTIM_USE_ARMA
#define BMO_INVERSE(X) arma::inv(X)
#endif
/*-------------------- Sum --------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_SUM(X) (X).array().sum()
#elif OPTIM_USE_ARMA
#define BMO_SUM(X) arma::accu(X)
#endif
/*------------------ Determinant ------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_DET(X) (X).determinant()
#elif OPTIM_USE_ARMA
#define BMO_DET(X) arma::det(X)
#endif
/*------------------ Inner Product ------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_DOT_PROD(X, Y) (X).dot(Y)
#elif OPTIM_USE_ARMA
#define BMO_DOT_PROD(X, Y) arma::dot(X, Y)
#endif
#ifdef OPTIM_USE_EIGEN
#define BMO_MAT_DOT_PROD(X, Y) ((X).array() * (Y).array()).sum()
#elif OPTIM_USE_ARMA
#define BMO_MAT_DOT_PROD(X, Y) arma::dot(X, Y)
#endif
/*------------------ Square (Frob)Norm ------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_SQUARE_NORM(X) (X).squaredNorm()
#elif OPTIM_USE_ARMA
#define BMO_SQUARE_NORM(X) arma::dot(X, X)
#endif
/*------------------(Frob) Norm ------------------*/
#ifdef OPTIM_USE_EIGEN
#define BMO_NORM(X) (X).norm()
#elif OPTIM_USE_ARMA
#define BMO_NORM(X) arma::norm(X)
#endif

#endif