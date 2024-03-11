template <int p, typename fp_t>
void prox(fp_t step, const Mat<fp_t> &in_x, Mat<fp_t> &out)
{
    optim_assert(step > 0, "step must be positive.");
    const Index n = BMO_ROWS(in_x),
                m = BMO_COLS(in_x);
    BMO_RESIZE(out, n, m);
    if constexpr (p == 1)
    {
        for (Index i = 0, size = n * m; i < size; i++)
        {
            if (in_x(i) > step)
                out(i) = in_x(i) - step;
            else if (in_x(i) < -step)
                out(i) = in_x(i) + step;
            else
                out(i) = 0;
        }
    }
    else if constexpr (p == 2)
    {
        const fp_t nrm = BMO_FRO_NORM(in_x);
        if (nrm > step)
            out = (1 - step / nrm) * in_x;
        else
            BMO_SET_ZERO(out);
    }
    else if constexpr (p == -1)
    {
        out = BMO_ABS(in_x) / step;
        fp_t *begin = BMO_GET_DATA(out),
             *end = begin + n * m,
             sum, lambda, slice;
        // projection into the L1 ball
        sum = BMO_SUM(out);
        if (sum <= fp_t(1))
        { // in the L1 ball
            BMO_SET_ZERO(out);
            return;
        }
        else
            sum = fp_t(0);
        std::sort(begin, end);
        while (sum < fp_t(1) && end >= begin)
            sum += *end--;
        slice = *(end + 1);
        lambda = slice - (fp_t(1) - (sum - slice));
        for (Index i = 0; i < n * m; i++)
        {
            if (in_x(i) > lambda)
                out(i) = lambda * step;
            else if (in_x(i) < -lambda)
                out(i) = -lambda * step;
            else
                out(i) = in_x(i);
        }
    }
}