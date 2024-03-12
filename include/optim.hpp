#ifndef _OPTIM_LIB_HPP_
#define _OPTIM_LIB_HPP_

#include "macro/macro.h"
#include "macro/BasicMatrixOp.h"

#include "base/BaseSolver.hpp"
#include "base/Recorder.hpp"

#include "line_search/Armijo.hpp"
#include "line_search/More_Thuente.hpp"
#include "line_search/Zhang_Hager.hpp"

#include "unconstrained/gradient/Gradient_Descent.hpp"

#include "unconstrained/newton/BFGS.hpp"
#include "unconstrained/newton/LBFGS.hpp"
#include "unconstrained/newton/NewtonLDLT.hpp"
#include "unconstrained/newton/NewtonCG.hpp"

#include "constrained/ALM.hpp"

#endif