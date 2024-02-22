#pragma once
#ifndef _OPTIM_NEWTON_LBFGS_HPP_
#define _OPTIM_NEWTON_LBFGS_HPP_

#include "line_search/More_Thuente.hpp"
#include "LBFGS.ipp"

namespace optim
{
    template <typename fp_t,
              template <typename> class LS =
                  MTLineSearch>
    class LBFGS final : public BaseSolver<fp_t>
    {
    public:
        using Problem = GradProblem<fp_t>;
        using Constant = OptimConst<fp_t>;
        using BaseSolver<fp_t>::iter;
        // static_assert(std::is_base_of_v<>)
        using LineSearchImp = MTLineSearch<fp_t>;

    private:
        Problem *prob;

    public:
        LineSearchImp ls;
        int max_iter = 100; ///< max number of iterations
        fp_t xtol = 1e-6;   ///< stop if |x_{k+1} - x_k| < xtol
        fp_t ftol = 1e-6;   ///< stop if |f_{k+1} - f_k| < ftol
        fp_t gtol = 1e-4;   ///< stop if |g_{k}| < gtol
        fp_t step = 1e-2;   ///< initial step size
        int m = 4;          ///< number of memory
        int status;         // wether solve successfully

    public:
        explicit LBFGS(
            Problem &prob)
        {
            this->prob = &prob;
        };

        fp_t solve(Mat<fp_t> &cur_x) override
        {
            using namespace internal::LBFGS;
            const size_t n = BMO_ROWS(cur_x),
                         k = BMO_COLS(cur_x),
                         nk = n * k;
            Col<fp_t> d(nk);
            Mat<fp_t> prev_x = cur_x,
                      cur_g(n, k), prev_g(n, k);
            CircularArray<Storage<fp_t>> memory(m);
            fp_t prev_loss = prob->loss(cur_x), cur_loss,
                 g_nrm, x_diff_nrm, f_diff,
                 bb_step = step;
            ls.init(this->prob, prev_loss);
            prob->grad(cur_x, prev_g);
            // make step 0
            ls.line_search(
                bb_step, prev_g,
                prev_x, cur_x,
                prev_loss, cur_loss,
                prev_g, cur_g);
            Storage<fp_t> sy;
            {
                // map mat to col, without memory allocation
                MapCol<fp_t> BMO_INIT_MAP_COL(map_cx, cur_x, nk),
                    BMO_INIT_MAP_COL(map_px, prev_x, nk),
                    BMO_INIT_MAP_COL(map_cg, cur_g, nk),
                    BMO_INIT_MAP_COL(map_pg, prev_g, nk);
                // compute bb_step and update pho
                sy.s = map_cx - map_px;
                sy.y = map_cg - map_pg;
                bb_step = BMO_DOT_PROD(sy.s, sy.y);
                sy.rho = 1. / bb_step;
                bb_step /= BMO_SQUARE_NORM(sy.y);
                d = -map_cg;
                memory.push_back(std::move(sy));
            }
            // update prev_x, prev_g, prev_loss and memory
            prev_x = std::move(cur_x);
            prev_g = std::move(cur_g);
            prev_loss = cur_loss;
            // begin main loop
            for (iter = 1; iter <= max_iter; iter++)
            {
                // compute H * d
                lbfgs_update_direction(memory, d, bb_step);
                // begin line search
                bb_step = 1.;
                ls.line_search(
                    bb_step, d,
                    prev_x, cur_x,
                    prev_loss, cur_loss,
                    prev_g, cur_g);
                { // mapping may be expired, so we need to re-map
                    MapCol<fp_t> BMO_INIT_MAP_COL(map_cx, cur_x, nk),
                        BMO_INIT_MAP_COL(map_px, prev_x, nk),
                        BMO_INIT_MAP_COL(map_cg, cur_g, nk),
                        BMO_INIT_MAP_COL(map_pg, prev_g, nk);
                    // compute bb_step1
                    sy.s = map_cx - map_px;
                    sy.y = map_cg - map_pg;
                    const double sTy = BMO_MAT_DOT_PROD(sy.s, sy.y);
                    const double y_nrm2 = BMO_SQUARE_NORM(sy.y);
                    if (y_nrm2 < Constant::eps)
                    { // too small to storage!
                        d = -map_cg;
                        bb_step = step;
                        goto continue_main_loop;
                    }
                    if (sTy > 0) [[likely]]
                    { // use bb step1 or step2
                        bb_step = sTy / y_nrm2;
                        // force bb_step to be in [min_step, max_step]
                        bb_step = std::min(
                            std::max(ls.min_step, bb_step), ls.max_step);
                    }
                    else [[unlikely]] // sTy == 0
                        bb_step = step;
                    sy.rho = 1. / sTy;
                    // update -gradient to d
                    d = -map_cg;
                }
                // check stop criteria
                g_nrm = BMO_NORM(cur_g);
                x_diff_nrm = BMO_NORM(sy.s); // = norm(d) * step;
                f_diff = std::abs(cur_loss - prev_loss) /
                         std::abs(cur_loss);
                if (g_nrm < gtol && x_diff_nrm < xtol && f_diff < ftol)
                { // check stop criteria
                    status = 0;
                    break;
                }
                // update prev_x, prev_g, prev_loss and memory
                memory.push_back(std::move(sy));
            continue_main_loop:
                prev_x = std::move(cur_x);
                prev_g = std::move(cur_g);
                prev_loss = cur_loss;
            }
            status = 1; // TODO: logger
            return cur_loss;
        };
    };
}

// #ifdef OPTIM_HEADER_ONLY
// #include "unconstrained/newton/LBFGS.cpp"
// #endif
#endif