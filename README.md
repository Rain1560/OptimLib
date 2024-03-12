# OptimLib &nbsp; 
OptimLib is a C++ library of numerical optimization algorithms. It is designed to be a lightweight, efficient, and easy-to-use library for solving unconstrained and constrained optimization problems. The library is header-only with template classes and functions.

## Overview

 - c++17 or later, c++20 is recommended.
 - For fast matrix operations, the library uses the [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) or [Armadillo](https://arma.sourceforge.net/) library. Or you can use your own matrix library by implementing the Macro. Using Eigen is recommended. Don't forget to define the `OPTIM_USE_EIGEN` or `OPTIM_USE_ARMA` macro to use the corresponding library.
 - Available as a single precision or double precision library by substituting the `fp_t` template parameter.
 - Available algorithm are listed below:
    - unconstrained optimization:
        - (Proximal) Gradient Descent(with acceleration)
        - Newton's Method(NewtonCG, NewtonLDLT)
        - Quasi-Newton(BFGS, L-BFGS)
    - constrained optimization:
        - Augmented Lagrangian Method(ALM)
    

# Installation
The library using these libraries:
 - Logging library: [spdlog](https://github.com/gabime/spdlog.git)
 - [Eigen](https://gitlab.com/libeigen/eigen.git) or [Armadillo](https://gitlab.com/conradsnicta/armadillo-code.git) for matrix operations
 - Awesome-Doxygen(optional) for generating documentation: [Awesome-Doxygen](https://github.com/jothepro/doxygen-awesome-css.git)

The library using git submodules. To clone the repository and its submodules, use the following command:

```bash
git clone --recurse-submodules
```

# Usage

All the algorithms are implemented as template classes and offer a `fp_t solve(Mat<fp_t> x)` method that returns the final loss and modifies the input matrix `x` to the optimal solution. Take `NewtonCG` as an example, you need to following the steps below:

### Implement Problem Class

Navigate to [problem](problem.html) page to see what method that you need to implement for each solver. Here is an example of a problem class(or `HessProblem`) for the `NewtonCG` solver: 

```cpp
template <typename fp_t>
struct QuadraticFunction
{
    using Mat = optim::Mat<fp_t>;
    Mat init_x = BMO_INIT_ZERO(Mat, 2, 1);
    Mat true_x = BMO_INIT_ZERO(Mat, 2, 1);
};

template <typename fp_t>
struct Quadratic1 : public optim::HessProblem<fp_t>,
                    public QuadraticFunction<fp_t>
{
    using Mat = optim::Mat<fp_t>;
    using QuadraticFunction<fp_t>::true_x;
    using QuadraticFunction<fp_t>::init_x;
    Quadratic1()
    {
        true_x(0, 0) = 2.25, true_x(1, 0) = -4.75;
    }

    fp_t loss(const Mat &x) override
    {
        const fp_t x_1 = x(0, 0);
        const fp_t x_2 = x(1, 0);
        return 3 * x_1 * x_1 + 2 * x_1 * x_2 + x_2 * x_2 - 4 * x_1 + 5 * x_2;
    }

    void grad(const Mat &x, Mat &g) override
    {
        const fp_t x_1 = x(0, 0);
        const fp_t x_2 = x(1, 0);
        g(0, 0) = 6 * x_1 + 2 * x_2 - 4;
        g(1, 0) = 2 * x_1 + 2 * x_2 + 5;
    }

    void hess(const Mat &x, Mat &H) override
    {
        H(0, 0) = 6.0;
        H(0, 1) = 2.0;
        H(1, 0) = 2.0;
        H(1, 1) = 2.0;
    }
};
```
### Create corresponding solver

In the main function first create a shared_ptr to the problem, then create the solver and call the `solve` method. 

Here is an example of the main function:

```cpp
#include <iostream>
#include <optim.hpp>
int main(int argc, char const *argv[])
{
    using namespace optim;
    auto prob1 = std::make_shared<Quadratic1<double>>();
    {
        NewtonCG<double> solver1(prob1);
        Mat<double> x1 = prob1->init_x;
        solver1.solve(x1);
        std::cout << "err to true result: "
                  << BMO_FRO_NORM(x1 - prob1->true_x)
                  << std::endl;
    }
    return 0;
}
```

### Compile and run

Compile command:
```bash
g++ main.cpp -std=c++20 -I */include -I */eigen3/Eigen -I */spdlog/include -DOPTIM_USE_EIGEN
```

Possible output:
```bash
err to true result: 0
```