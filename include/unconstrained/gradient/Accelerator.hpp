#pragma once
#ifndef _OPTIM_GRADIENT_ACCELERATOR_HPP_
#define _OPTIM_GRADIENT_ACCELERATOR_HPP_

#include "line_search/base.hpp"

namespace optim
{
    template <typename fp_t>
    struct GDAccelerator
    {
        using BaseArgs = BaseLineSearchArgs<fp_t>;
        using GradFunc = typename std::function<void(const Mat<fp_t> &in_x, Mat<fp_t> &out_grad)>;

        virtual void init(BaseArgs &){};

        /// @brief update cur_x, direction and cur_grad base on current prev_x, prev_grad and cur_x.
        /// @param iter current iteration
        /// @param grad_f for computing gradient
        /// @param arg storage prev_x, prev_grad, cur_x, cur_grad, direction
        virtual void update(
            int iter,
            GradFunc grad_f,
            BaseArgs &arg)
        {
            grad_f(arg.cur_x, arg.cur_grad);
            arg.direction = -arg.cur_grad;
        };

        virtual void release(){};
    };

    template <typename fp_t>
    struct Nesterov
        : public GDAccelerator<fp_t>
    {
        using BaseArgs = typename GDAccelerator<fp_t>::BaseArgs;
        using GradFunc = typename GDAccelerator<fp_t>::GradFunc;

        fp_t momentum = 0.9;

    private:
        Mat<fp_t> v; ///< velocity = x_{k+1} - x_k

    public:
        void update(
            int iter, GradFunc grad_f, BaseArgs &arg) override
        {
            momentum = fp_t(iter - 1) / fp_t(iter + 2);
            v = arg.cur_x - arg.prev_x;
            arg.cur_x = arg.prev_x + momentum * v;
            grad_f(arg.cur_x, arg.cur_grad);
            arg.direction = -arg.cur_grad;
        }

        void release() override
        {
            BMO_RESIZE(v, 0, 0);
        }
    };

    template <typename fp_t>
    struct AdaGrad
        : public GDAccelerator<fp_t>
    {
        using BaseArgs = typename GDAccelerator<fp_t>::BaseArgs;
        using GradFunc = typename GDAccelerator<fp_t>::GradFunc;
        using Constant = OptimConst<fp_t>;

        Mat<fp_t> G, ///< G = sum(g_i.^2)
            RG;      ///< RG = sqrt(G+eps)

    public:
        void init(BaseArgs &arg) override
        {
            const Index n = BMO_ROWS(arg.cur_x),
                        k = BMO_COLS(arg.cur_x);
            G = BMO_INIT_ZERO(Mat<fp_t>, n, k);
            RG = BMO_INIT_ZERO(Mat<fp_t>, n, k);
        }

        void update(
            int iter, GradFunc grad_f, BaseArgs &arg) override
        {
            grad_f(arg.cur_x, arg.cur_grad);
            G += BMO_ARRAY_MUL(arg.cur_grad, arg.cur_grad);
            RG = BMO_ARRAY_ADD_SCALAR(G, Constant::eps);
            RG = BMO_ARRAY_SQRT(RG);
            arg.direction = -BMO_ARRAY_DIV(arg.cur_grad, RG);
        }

        void release() override
        {
            BMO_RESIZE(G, 0, 0);
            BMO_RESIZE(RG, 0, 0);
        }
    };

    template <typename fp_t>
    struct RMSProp
        : public GDAccelerator<fp_t>
    {
        using BaseArgs = typename GDAccelerator<fp_t>::BaseArgs;
        using GradFunc = typename GDAccelerator<fp_t>::GradFunc;
        using Constant = OptimConst<fp_t>;

        fp_t pho = 0.9;
        Mat<fp_t> M, ///< \(M_k = \pho * M_{k-1} + (1-\pho)g_k.*g_k\)
            RM;      ///< \(RM = \sqrt(M+\eps)\)

    public:
        void init(BaseArgs &arg) override
        {
            const Index n = BMO_ROWS(arg.cur_x),
                        k = BMO_COLS(arg.cur_x);
            M = BMO_INIT_ZERO(Mat<fp_t>, n, k);
            RM = BMO_INIT_ZERO(Mat<fp_t>, n, k);
        }

        void update(
            int iter, GradFunc grad_f, BaseArgs &arg) override
        {
            grad_f(arg.cur_x, arg.cur_grad);
            M = pho * M + (1 - pho) * BMO_ARRAY_MUL(arg.cur_grad, arg.cur_grad);
            RM = BMO_ARRAY_ADD_SCALAR(M, Constant::eps);
            RM = BMO_ARRAY_SQRT(RM);
            arg.direction = -BMO_ARRAY_DIV(arg.cur_grad, RM);
        }

        void release() override
        {
            BMO_RESIZE(M, 0, 0);
            BMO_RESIZE(RM, 0, 0);
        }
    };

    template <typename fp_t>
    struct AdaDelta
        : public GDAccelerator<fp_t>
    {
        using BaseArgs = typename GDAccelerator<fp_t>::BaseArgs;
        using GradFunc = typename GDAccelerator<fp_t>::GradFunc;
        using Constant = OptimConst<fp_t>;

        fp_t pho = 0.9;
        Mat<fp_t> M, RM, D, RD, v;

    public:
        void init(BaseArgs &arg) override
        {
            const Index n = BMO_ROWS(arg.cur_x),
                        k = BMO_COLS(arg.cur_x);
            M = BMO_INIT_ZERO(Mat<fp_t>, n, k);
            RM = BMO_INIT_ZERO(Mat<fp_t>, n, k);
            D = BMO_INIT_ZERO(Mat<fp_t>, n, k);
            RD = BMO_INIT_ZERO(Mat<fp_t>, n, k);
            v = BMO_INIT_ZERO(Mat<fp_t>, n, k);
        }

        void update(
            int iter, GradFunc grad_f, BaseArgs &arg) override
        {
            grad_f(arg.cur_x, arg.cur_grad);
            M = pho * M + (1 - pho) * BMO_ARRAY_MUL(arg.cur_grad, arg.cur_grad);
            RM = BMO_ARRAY_ADD_SCALAR(M, Constant::eps);
            RM = BMO_ARRAY_SQRT(RM);
            v = arg.cur_x - arg.prev_x;
            D = pho * D + (1 - pho) * BMO_ARRAY_MUL(v, v);
            RD = BMO_ARRAY_ADD_SCALAR(D, Constant::eps);
            RD = BMO_ARRAY_SQRT(RD);
            arg.direction = -BMO_ARRAY_DIV(RD, RM);
            arg.direction = BMO_ARRAY_MUL(arg.direction, arg.cur_grad);
        }

        void release() override
        {
            BMO_RESIZE(M, 0, 0);
            BMO_RESIZE(RM, 0, 0);
            BMO_RESIZE(D, 0, 0);
            BMO_RESIZE(RD, 0, 0);
            BMO_RESIZE(v, 0, 0);
        }
    };

    template <typename fp_t>
    struct Adam
        : public GDAccelerator<fp_t>
    {
        using BaseArgs = typename GDAccelerator<fp_t>::BaseArgs;
        using GradFunc = typename GDAccelerator<fp_t>::GradFunc;
        using Constant = OptimConst<fp_t>;

        fp_t pho1 = 0.9, pho2 = 0.999;
        Mat<fp_t> S, M;

    public:
        void init(BaseArgs &arg) override
        {
            const Index n = BMO_ROWS(arg.cur_x),
                        k = BMO_COLS(arg.cur_x);
            M = BMO_INIT_ZERO(Mat<fp_t>, n, k);
            S = BMO_INIT_ZERO(Mat<fp_t>, n, k);
        }

        void update(
            int iter, GradFunc grad_f, BaseArgs &arg) override
        {
            grad_f(arg.cur_x, arg.cur_grad);
            S = pho1 * S + (1 - pho1) * arg.cur_grad;
            M = pho2 * M + (1 - pho2) * BMO_ARRAY_MUL(arg.cur_grad, arg.cur_grad);
            arg.direction =
                -BMO_ARRAY_DIV(
                    BMO_ARRAY_ADD_SCALAR(S, Constant::eps),
                    BMO_ARRAY_ADD_SCALAR(M, Constant::eps)) /
                (1 - std::pow(pho1, iter)) *
                (1 - std::pow(pho2, iter));
        }

        void release() override
        {
            BMO_RESIZE(M, 0, 0);
            BMO_RESIZE(S, 0, 0);
        }

        // template <typename fp_t>
    };
};

// template <typename fp_t>
// struct RMSProp : public GDAccelerator<fp_t>
// {
//     Mat<fp_t> M, R;
//     fp_t pho = 0.9;

//     void init(const Mat<fp_t> &x) override
//     {
//         const size_t n = BMO_ROWS(x),
//                      k = BMO_COLS(x);
//         M = BMO_INIT_ZERO(Mat<fp_t>, n, k);
//     }

//     void update(
//         int iter, const fp_t step,
//         const Mat<fp_t> &g, Mat<fp_t> &direct) override
//     {
//         M = pho * M + (1 - pho) * BMO_ARRAY_MUL(g, g);
//         // R = sqrt(M+eps)
//         R = BMO_ARRAY_ADD_SCALAR(M, 1e-5);
//         R = BMO_ARRAY_SQRT(R);
//         direct = -step * BMO_ARRAY_DIV(g, R);
//     }

//     void release() override
//     {
//         BMO_RESIZE(M, 0, 0);
//         BMO_RESIZE(R, 0, 0);
//     }
// };

// template <typename fp_t>
// struct AdaDelta : public GDAccelerator<fp_t>
// {
//     fp_t pho = 0.9;
//     Mat<fp_t> M, T, d, R;

//     void init(const Mat<fp_t> &x) override
//     {
//         const size_t n = BMO_ROWS(x),
//                      k = BMO_COLS(x);
//         M = BMO_INIT_ZERO(Mat<fp_t>, n, k);
//         R = BMO_INIT_ZERO(Mat<fp_t>, n, k);
//         T = BMO_INIT_ZERO(Mat<fp_t>, n, k);
//         d = BMO_INIT_ZERO(Mat<fp_t>, n, k);
//     }

//     void update(
//         int iter, const fp_t step,
//         const Mat<fp_t> &g, Mat<fp_t> &direct) override
//     {
//         // T = sqrt(d+eps)
//         T = BMO_ARRAY_ADD_SCALAR(d, 1e-5);
//         T = BMO_ARRAY_SQRT(T);
//         // M = pho * M + (1 - pho) * g^2
//         M = pho * M + (1 - pho) * BMO_ARRAY_MUL(g, g);
//         // R = sqrt(M+eps)
//         R = BMO_ARRAY_ADD_SCALAR(M, 1e-5);
//         R = BMO_ARRAY_SQRT(R);
//         // direct = -T/R * g
//         direct = -BMO_ARRAY_DIV(T, R);
//         direct = BMO_ARRAY_MUL(direct, g);
//         // d = pho * d + (1 - pho) * direct^2
//         d = pho * d + (1 - pho) * BMO_ARRAY_MUL(direct, direct);
//     }

//     void release() override
//     {
//         BMO_RESIZE(M, 0, 0);
//         BMO_RESIZE(T, 0, 0);
//         BMO_RESIZE(d, 0, 0);
//         BMO_RESIZE(R, 0, 0);
//     }
// };

// template <typename fp_t>
// struct Adam : public GDAccelerator<fp_t>
// {
//     fp_t pho1 = 0.9, pho2 = 0.999;
//     Mat<fp_t> S, M, R;

//     void init(const Mat<fp_t> &x) override
//     {
//         const size_t n = BMO_ROWS(x),
//                      k = BMO_COLS(x);
//         M = BMO_INIT_ZERO(Mat<fp_t>, n, k);
//         S = BMO_INIT_ZERO(Mat<fp_t>, n, k);
//         R = BMO_INIT_ZERO(Mat<fp_t>, n, k);
//     }

//     void update(
//         int iter, const fp_t step,
//         const Mat<fp_t> &g, Mat<fp_t> &direct) override
//     {
//         S = pho1 * S + (1 - pho1) * g;
//         M = pho2 * M + (1 - pho2) * BMO_ARRAY_MUL(g, g);
//         // S = S / (1 - pho1^iter)
//         S /= (1 - std::pow(pho1, iter));
//         // M = M / (1 - pho2^iter)
//         M /= (1 - std::pow(pho2, iter));
//         // direct = -step * S / (sqrt(M) + eps)
//         //   R = sqrt(M+eps)
//         R = BMO_ARRAY_ADD_SCALAR(M, 1e-5);
//         R = BMO_ARRAY_SQRT(R);
//         // direct = -step * S / (sqrt(M) + eps)
//         direct = -step * BMO_ARRAY_DIV(S, R);
//     }

//     void release() override {}
// };

#endif