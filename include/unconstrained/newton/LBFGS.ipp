#include "base/BaseSolver.hpp"
namespace optim
{
    namespace internal::LBFGS
    {
        template <typename fp_t>
        struct Storage
        {
            Mat<fp_t> s, y;
            fp_t rho, alpha;
        };

        template <typename fp_t>
        void lbfgs_update_direction(
            const fp_t bb_step,
            CircularArray<Storage<fp_t>> &memory,
            Mat<fp_t> &d)
        // note: d = -grad at the beginning
        {
            const int m = memory.size();
            for (int i = m - 1; i >= 0; i--)
            {
                auto &mem = memory[i];
                mem.alpha = mem.rho * BMO_MAT_DOT_PROD(mem.s, d);
                d -= mem.alpha * mem.y;
            }
            d *= bb_step;
            for (int i = 0; i <= m - 1; i++)
            {
                const fp_t beta = memory[i].rho * BMO_MAT_DOT_PROD(memory[i].y, d);
                d += (memory[i].alpha - beta) * memory[i].s;
            }
        };

    }

}