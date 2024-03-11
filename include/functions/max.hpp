template <typename fp_t>
void max(
    Mat<fp_t> &in_x,
    Mat<fp_t> &in_y,
    Mat<fp_t> &out)
{
    const Index n = BMO_ROWS(in_x),
                m = BMO_COLS(in_x);
    optim_assert(BMO_ROWS(in_y) == n &&
                     BMO_COLS(in_y) == m,
                 "size mismatch in max func.");
    BMO_RESIZE(out, n, m);
    for (Index i = 0, size = n * m; i < size; i++)
        out(i) = std::max(in_x(i), in_y(i));
}

template <typename fp_t>
void max(
    Mat<fp_t> &in_x,
    fp_t in_y,
    Mat<fp_t> &out)
{
    const Index n = BMO_ROWS(in_x),
                m = BMO_COLS(in_x);
    BMO_RESIZE(out, n, m);
    for (Index i = 0, size = n * m; i < size; i++)
        out(i) = std::max(in_x(i), in_y);
}