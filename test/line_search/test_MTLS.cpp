#include "base/BaseSolver.hpp"
#include "misc/logger.hpp"
#include "line_search/More_Thuente.hpp"

using namespace optim;

struct Problem1 : MTLineSearch<double>::Problem
{
    double loss(const Mat<double> &params) override
    {
        return -params(0) / (params(0) + 1) / (params(0) + 1);
    }

    void grad(const Mat<double> &in, Mat<double> &out) override
    {
        double tmp = in(0) + 1;
        out(0) = 1. / tmp / tmp - 2. / tmp / tmp / tmp;
    }
};

int main(int argc, char const *argv[])
{
    optim::logger.set_verbosity(4);
    {
        Problem1 p;
        optim::MTLineSearch<double> ls(p);
        ls.max_iter = 100;
        ls.wolfe_c2 = 1e-5;
        optim::Mat<double> x(1, 1);
        optim::LineSearchArgs<double, false> arg(x);
        // set step, cur_x, direction
        arg.step = 0.1;
        arg.cur_x(0) = 0;
        arg.direction(0) = 0.1;
        // update cur_loss, cur_grad
        arg.update_loss(&p);
        arg.update_grad(&p);
        // move cur_x, cur_grad, cur_loss to prev_x, prev_grad, prev_loss
        arg.flush(); 
        // before call line_search, arg.step, arg.prev_x, arg.prev_loss, arg.prev_grad should be set
        ls.line_search(arg);
        std::cout << "iter: " << ls.n_iter() << std::endl;
        std::cout << "step: " << arg.step << std::endl;
        std::cout << "in_x: " << arg.prev_x << std::endl;
        std::cout << "out_x: " << arg.cur_x << std::endl;
        std::cout << "in_loss: " << arg.prev_loss << std::endl;
        std::cout << "out_loss: " << arg.cur_loss << std::endl;
        std::cout << "in_grad: " << arg.prev_grad << std::endl;
        std::cout << "out_grad: " << arg.cur_grad << std::endl;
    }
    return 0;
}
