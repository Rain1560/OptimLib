#include "misc/error_report.hpp"

namespace optim
{
    namespace internal
    {
        void error_report(
            int &status,
            fp_t x_diff_nrm, fp_t xtol,
            fp_t f_diff, fp_t ftol,
            fp_t g_nrm, fp_t gtol)
        {
            status = 3;
            if (f_diff < ftol)
            {
            }
            else
            {
            }
        }
    }
}