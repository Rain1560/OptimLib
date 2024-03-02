#pragma once
#ifndef _OPTIM_BASE_RECORDER_HPP_
#define _OPTIM_BASE_RECORDER_HPP_

#include "macro/macro.h"
#include "misc/logger.hpp"

///< @cond
namespace optim
{
    template <typename T>
    class CircularArray
    {
        int _head = 0;   // head index
        int _size = 0;   // size of the data
        int _length = 0; // length of the array
    public:
        T *data = 0;

        OPTIM_INLINE const T &front() const
        {
            return data[_head];
        };

        OPTIM_INLINE const T &back() const
        {
            return data[(_head + _length - 1) % _size];
        };

        OPTIM_INLINE T &front()
        {
            return data[_head];
        };

        OPTIM_INLINE T &back()
        {
            return data[(_head + _length - 1) % _size];
        };

        OPTIM_INLINE int
        size() const
        {
            return _length;
        };

        OPTIM_INLINE T &
        operator[](int i)
        {
            return data[(_head + i) % _size];
        };

        template <typename... Arg>
        OPTIM_INLINE void
        emplace_back(Arg &&...args)
        {
            data[(_head + _length) % _size] = T(std::forward<Arg>(args)...);
            _length++;
        }

        OPTIM_INLINE void
        pop_back()
        {
            if (_length == 0) [[unlikely]]
                return;
            _length--;
        };

        OPTIM_INLINE void
        pop_front()
        {
            if (_length == 0) [[unlikely]]
                return;
            _head = (_head + 1) % _size;
            _length--;
        };

        OPTIM_INLINE void
        push_back(const T &x)
        {
            if (_length == _size)
            {
                data[_head] = x;
                _head = (_head + 1) % _size;
            }
            else
            {
                data[(_head + _length) % _size] = x;
                _length++;
            }
        };

        OPTIM_INLINE void
        push_back(T &&x)
        {
            if (_length == _size)
            {
                data[_head] = std::move(x);
                _head = (_head + 1) % _size;
            }
            else
            {
                data[(_head + _length) % _size] = std::move(x);
                _length++;
            }
        };

        bool empty() const
        {
            return _length == 0;
        };

        CircularArray() = default;

        CircularArray(int size)
        {
            _size = size;
            data = new T[size];
            _head = 0;
        };

        ~CircularArray()
        {
            if (data)
                delete[] data;
        };
    };

    template <typename fp_t>
    class Recorder
    {
    public:
        struct Data
        {
            int appear_iter;
            double loss;

            Data(){};
            Data(int iter, fp_t _loss)
            {
                appear_iter = iter;
                loss = _loss;
            };
        };

    private:
        bool record_x;
        int length; // length of the recorder
        CircularArray<Data> history_loss;

    public:
        int best_iter;
        fp_t best_loss = std::numeric_limits<fp_t>::max();
        Mat<fp_t> best_x;

    public:
        explicit Recorder(int len = 8, bool record_x = true)
            : history_loss(len)
        {
            length = len;
            this->record_x = record_x;
        }

        /// @brief record current iteration and loss
        /// @param iter  current iteration
        /// @param loss  current loss
        /// @param x    current point
        void record(int iter, double loss,
                    const Mat<fp_t> &x)
        {
            if (loss < best_loss)
            {
                best_iter = iter;
                best_loss = loss;
                if (record_x)
                    best_x = x;
            }
            if (!history_loss.empty() &&
                history_loss.front().appear_iter <= iter - length)
                history_loss.pop_front();
            while (!history_loss.empty() && history_loss.back().loss < loss)
                history_loss.pop_back();
            history_loss.emplace_back(iter, loss);
        }

        fp_t prev_k_max_loss() const { return history_loss.front().loss; }

        OPTIM_INLINE bool should_stop(const double, const double) const { return false; }
    };

} // namespace optim

/// @endcond

// #ifdef OPTIM_HEADER_ONLY
// #include "base/Recorder.cpp"
// #endif

#endif