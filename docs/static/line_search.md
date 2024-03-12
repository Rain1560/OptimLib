# Line Search

[TOC]

## Algorithms Details

Below Line Search algorithms are trying to find a step \f$\alpha_k \f$ that satisfies different conditions. To make sure the optimization algorithms are converging, a proper Line Search algorithm is significant.

We define 2 line search functions: \f$\phi(\alpha) = f(x + \alpha d)\f$ and auxiliary function \f$\psi(\alpha) = \phi(\alpha) - \phi(0) - \alpha \phi'(0)\f$.

### Armijo Line Search

Armijo line search is a backtracking line search algorithm that satisfies the Armijo condition: \f$\phi(\alpha) \leq \phi(0) + c_1 \alpha \phi'(0) \f$. The algorithm step is computed by the following steps:

The back tracking algorithm is described as follows:
1. check if Armijo condition is satisfied, if not, reduce the step size by multiplying a factor \f$\text{decay_rate} \in (0, 1) \f$.
2. repeat step 1 until the condition is satisfied or the maximum number of iterations is reached.

### Zhang-Hager Line Search

Zhang-Hager line search is a non-monotone line search algorithm that satisfies the following condition: 

$$
\begin{align*}
\phi(\alpha_k) &\leq C_k + \rho \alpha_k \phi'(0) \newline
C_k &= \frac{\gamma Q_{k-1}C_{k-1} + f(x_k)}{Q_k} \newline
Q_k &= \gamma Q_{k-1} + 1
\end{align*}
$$

We using back tracking method like Armijo line search.

### More-Thuente Line Search

The More-Thuente line search algorithm is a line search algorithm that satisfies the strong Wolfe conditions:

$$
\left\{
\begin{align*}
\phi(\alpha) &\leq \phi(0) + c_1 \alpha \phi'(0) \newline
|\phi'(\alpha)| &\leq c_2 |\phi'(0)| \newline
c_1 \in (0, \frac{1}{2}) &\quad c_2 \in (0, 1)    
\end{align*}
\right.
$$

Note that $c_1$ should be close to 0 and $c_2$ should be close to 1. Otherwise it will be hard to find a step that satisfies the conditions.

The algorithm step is computed by the following steps:

Denote \f$\alpha_l, \alpha_u, \alpha_t\f$ and there corresponding function values and derivatives as \f$f_l, f_u, f_t, g_l, g_u, g_t\f$. After step 4 below, we ensure that \f$f_l \leq f_u\f$. When the bracket is true, the minimum of \f$\phi\f$ is guaranteed to be within the interval \f$[\ \alpha_l, \alpha_u\ ]\f$.
1. check if the current step satisfies the strong Wolfe conditions. If it does, update loss and gradient then return.
2. If not, let \f$ \alpha_l = \alpha_u = 0, \alpha_t = t_k \f$ and `auxiliary = true`. Initialize the interval of uncertainty.
3. select a trial value base on \f$ \alpha_l, \alpha_u, \alpha_t \f$ and their corresponding function values and derivatives. We force the step to be within the interval.
4. use \f$ \alpha_t \f$ to update \f$ \alpha_l, \alpha_u\f$.
5. check if \f$ \alpha_l \f$ satisfies the strong Wolfe conditions.
6. let \f$ \alpha_t \f$ be the new trial step and update the function value and derivative.
7. check if we should switch to "Modified Updating Method" and if bracket is true.
8. update the interval of uncertainty, go to step 3.

## Usage

All line search class offers `void line_search(LineSearchArgs<fp_t,use_prox> &arg)` member function. Before you call this function, you should set the following member variables of `LineSearchArgs`:

```cpp
step // The initial step size
prev_x // The previous point, where the line search starts
prev_loss // The loss at the previous point
prev_grad // The (smoothing part)gradient at the previous point
direction // The search direction, which is the negative gradient in most cases
```

If `use_prox` is `true`, the line search class will calculate the gradient map base on your step size and store it in `prev_grad_map`. After the line search, we will storage the new point in `cur_x` and the new loss in `cur_loss`. If you set `update_cur_grad = true`, we will also storage the new gradient in `cur_grad`.