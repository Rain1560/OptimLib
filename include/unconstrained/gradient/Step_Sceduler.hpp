#pragma once
#ifndef _OPTIM_GRADIENT_LR_SCHEDULER_HPP_
#define _OPTIM_GRADIENT_LR_SCHEDULER_HPP_

namespace optim
{
    template <typename fp_t>
    struct StepScheduler
    {
        using BaseArgs = BaseLineSearchArgs<fp_t>;

        virtual void update(int iter, BaseArgs &arg){};
        virtual void release(){};
    };

    template <typename fp_t>
    struct BBStepScheduler
        : public StepScheduler<fp_t>
    {
        using BaseArgs = BaseLineSearchArgs<fp_t>;

        Mat<fp_t> s, y;

        void update(int iter, BaseArgs &arg) override
        {
            // update s and y
            s = arg.cur_x - arg.prev_x;
            y = arg.cur_grad - arg.prev_grad;
            const fp_t sTy = std::abs(BMO_MAT_DOT_PROD(s, y));
            fp_t bb_step;
            if (sTy > 0)
            { // use bb step1 or step2
                if (iter % 2 == 1)
                    bb_step = sTy / BMO_SQUARE_NORM(y);
                else
                    bb_step = BMO_SQUARE_NORM(s) / sTy;
                arg.step = bb_step;
            }
        }

        void release() override
        {
            BMO_RESIZE(s, 0, 0);
            BMO_RESIZE(y, 0, 0);
        }
    };

    template <typename fp_t>
    struct ExpStepScheduler
        : public StepScheduler<fp_t>
    {
        using BaseArgs = BaseLineSearchArgs<fp_t>;

        fp_t decay = 0.9;

        void update(int iter, BaseArgs &arg) override
        {
            arg.step *= decay;
        }
    };
}

#endif