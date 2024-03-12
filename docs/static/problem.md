# Problem

[TOC]

## Unconstrained Optimization

### Gradient Descent

`using Problem = ProxWrapper<fp_t, GradProblem, use_prox>;`

You have to implement the following member functions of the problem class:

1. Smoothing Case:
    - `fp_t loss(const Mat<fp_t> &x)`compute the total loss at point `x`
    - `void grad(const Mat<fp_t> &x, Mat<fp_t> &g)`compute the gradient at point `x` and store it in `g`.
2. Non-smoothing Case:
    - `fp_t sm_loss(const Mat<fp_t> &x)` compute the smooth part of the loss at point `x`
    - `fp_t nsm_loss(const Mat<fp_t> &x)` compute the non-smooth part of the loss at point `x`
    - `fp_t loss(const Mat<fp_t> &x)` compute the total loss at point `x`, which is the sum of `sm_loss` and `nsm_loss` does not need to be implemented.
    - `void grad(const Mat<fp_t> &x, Mat<fp_t> &g)` compute the gradient of **smoothing part** of loss function at point `x` and store it in `g`.
    - `void prox(fp_t step, const Mat<fp_t> &in_x, Mat<fp_t> &out_x)` compute the proximal operator at point `in_x` with step size `step`

### (Quasi)Newton's Method

`using Problem = GradProblem<fp_t>;`

Like problem of `Gradient Descent`, you have to implement the following member functions of the problem class:

- `fp_t loss(const Mat<fp_t> &x)` compute the total loss at point `x`
- `void grad(const Mat<fp_t> &x, Mat<fp_t> &g)` compute the gradient at point `x` and store it in `g`.

### Newton's Method

The problem class of `NewtonLDLT` and `NewtonCG`.

`using Problem = HessProblem<fp_t>;`

You should not only implement member functions of `GradProblem<fp_t>` like （L）BFGS， but also the following member functions of the problem class:
 - `void hess(const Mat<fp_t> &x, Mat<fp_t> &H)` compute the Hessian at point `x` and store it in `H`.

## Constrained Optimization
### Augmented Lagrangian Method
You should implement the following member functions of the problem class:
- `void grad(const Mat<fp_t> &x, Mat<fp_t> &g)` compute the gradient of the **augmented Lagrangian function** at point `x` and store it in `g`.
- `void equality_constraint(const Mat<fp_t> &x, Mat<fp_t> &c)` compute the equality constraint at point `x` and store it in `c`.
- `void inequality_constraint(const Mat<fp_t> &x, Mat<fp_t> &c)` compute the inequality constraint at point `x` and store it in `c`.
