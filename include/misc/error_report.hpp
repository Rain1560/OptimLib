#pragma once
#ifndef _OPTIM_MISC_ERROR_REPORT_HPP_
#define _OPTIM_MISC_ERROR_REPORT_HPP_

#include "macro/macro.h"

namespace optim
{
    namespace internal
    {
        void error_report(
            int &status,
            fp_t x_diff_nrm, fp_t xtol,
            fp_t f_diff, fp_t ftol,
            fp_t g_nrm, fp_t gtol);
    }
}

#endif