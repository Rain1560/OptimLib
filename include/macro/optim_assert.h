#pragma once
#ifndef _OPTIM_ASSERT_HPP_
#define _OPTIM_ASSERT_HPP_

#ifndef OPTIM_NO_DEBUG
#define optim_assert(expr, msg)        \
    if (!(expr))                       \
    {                                  \
        std::cerr << msg << std::endl; \
        assert(expr);                  \
    }
#else
#define optim_assert(msg, expr) ((void)0)
#endif

#endif