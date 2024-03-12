#pragma once
#ifndef _OPTIM_AUGMENTED_LAGRANGE_MULTIPLIER_HPP_
#define _OPTIM_AUGMENTED_LAGRANGE_MULTIPLIER_HPP_

#include "base/BaseSolver.hpp"
#include "functions/functions.hpp"

namespace optim
{
    /// @brief Augmented Lagrange Method
    /// @details Augmented Lagrange Method is a first-order optimization algorithm that solving equality and inequality constrained optimization problem.
    /// @tparam fp_t floating-point type
    template <typename fp_t>
    struct AugmentedLagrangeMethod final
        : public BaseSolver<fp_t>
    {
        using Constant = OptimConst<fp_t>;
        using BaseSolver<fp_t>::iter;

        struct Problem : public GradProblem<fp_t>
        {
            Mat<fp_t> eq_multiplier, ///< Equality multiplier
                ineq_multiplier;     ///< Inequality multiplier

            fp_t qd_reg = 1.; ///< Quadratic regularization

            fp_t gtol = Constant::nan; ///< Gradient tolerance

            virtual void equality_constraint(const Mat<fp_t> &x, Mat<fp_t> &res){};

            virtual void inequality_constraint(const Mat<fp_t> &x, Mat<fp_t> &res){};

            virtual void solve_subproblem(Mat<fp_t> &x) = 0;
        };

        std::shared_ptr<Problem> prob;

        int max_iter = 20; ///< Max iteration

        fp_t alpha = 0.5,
             beta = 1,
             pho = 10;

        fp_t gtol = 1e-6;          ///< Gradient tolerance
        fp_t cons_viol_tol = 1e-6; ///< Constraint violation tolerance

        fp_t init_qd_reg = 1.;

    private:
        void update_constraint_violation(
            const Mat<fp_t> &x, Mat<fp_t> &eq_res, Mat<fp_t> &ineq_res)
        {
            prob->equality_constraint(x, eq_res);
            prob->inequality_constraint(x, ineq_res);
        }

    public:
        explicit AugmentedLagrangeMethod(
            std::shared_ptr<Problem> p)
            : prob(p) {}

        fp_t solve(Mat<fp_t> &x) override
        {
            fp_t cons_loss, cur_cons_tol;
            // resize matrix
            Mat<fp_t> eq_res = prob->eq_multiplier,
                      ineq_res = prob->ineq_multiplier,
                      cur_grad = x;
            // init subproblem tol
            prob->qd_reg = init_qd_reg;
            prob->gtol = 1 / init_qd_reg;
            cons_viol_tol = 1 / std::pow(init_qd_reg, alpha);
            for (iter = 1; iter < max_iter; iter++)
            {
                prob->solve_subproblem(x);
                update_constraint_violation(x, eq_res, ineq_res);
                fn::max(ineq_res, -prob->gtol / prob->qd_reg, ineq_res);
                cons_loss = BMO_SQUARE_NORM(ineq_res) + BMO_SQUARE_NORM(eq_res);
                cons_loss = std::sqrt(cons_loss);
                prob->grad(x, cur_grad);
                if (cons_loss < cur_cons_tol)
                {
                    if (BMO_FRO_NORM(cur_grad) < gtol)
                        return 0;
                    // update multiplier
                    prob->eq_multiplier += prob->qd_reg * eq_res;
                    prob->ineq_multiplier += prob->qd_reg * ineq_res;
                    fn::max(ineq_res, fp_t(0), ineq_res);
                    prob->gtol /= prob->qd_reg;
                    cur_cons_tol = 1 / std::pow(prob->qd_reg, beta);
                }
                else
                {
                    prob->qd_reg *= pho;
                    prob->gtol = 1. / prob->qd_reg;
                    cur_cons_tol = 1. / std::pow(prob->qd_reg, alpha);
                }
            }
            return 0;
        }
    };

    template <typename fp_t>
    using ALM = AugmentedLagrangeMethod<fp_t>;
}

#endif