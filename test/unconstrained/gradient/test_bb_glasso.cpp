#include "unconstrained/gradient/Barzilar_Borwein.hpp"
#include "../../problems/MutiLASSO.hpp"

#define M 128
#define N 256
#define K 2
#define MU 0.1
#define SPARSITY 0.1

int main(int argc, char const *argv[])
{
    using namespace optim;
    logger.set_verbosity(5);
    MultiLASSO<double> prob(M, N, K, 10, SPARSITY);
    ZHLineSearch<double,true> ls;
    ProxBB<double> solver(prob, ls);
    solver.step = 1e-4;
    solver.xtol = 1e-6;
    Mat<double> x = BMO_INIT_RAND(Mat<double>, N, K),
                g(N, K), px(N, K);
    solver.solve(x);
    report(prob, solver, x);
}
