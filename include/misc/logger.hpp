#pragma once
#ifndef _OPTIM_LOGGING_HPP_
#define _OPTIM_LOGGING_HPP_

#define OPTIM_DEFAULT_VERBOSE 2

#include "macro/macro.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/ostream_sink.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace optim
{
    class Logger
    {
        std::shared_ptr<spdlog::sinks::basic_file_sink_mt> file_sink;
        std::shared_ptr<spdlog::sinks::ostream_sink_mt> ostream_sink;

        enum class LogType
        {
            CONSOLE,
            FILE,
            OSTREAM
        };

        LogType cur_type;
        std::shared_ptr<spdlog::logger> cur_logger;

    public:
        Logger();

        void write_to_console();
        void write_to_file(const char *);
        void write_to_oss(std::ostream &);

        void set_pattern(const char *pattern);
        void set_verbosity(int verbosity);
        void set_verbosity(const char *level);

        template <typename... Args>
        void trace(spdlog::format_string_t<Args...> fmt, Args &&...args)
        {
            cur_logger->log(spdlog::level::trace, fmt, std::forward<Args>(args)...);
        }

        template <typename... Args>
        void debug(spdlog::format_string_t<Args...> fmt, Args &&...args)
        {
            cur_logger->log(spdlog::level::debug, fmt, std::forward<Args>(args)...);
        }

        template <typename... Args>
        void info(spdlog::format_string_t<Args...> fmt, Args &&...args)
        {
            cur_logger->log(spdlog::level::info, fmt, std::forward<Args>(args)...);
        }

        template <typename... Args>
        void warn(spdlog::format_string_t<Args...> fmt, Args &&...args)
        {
            cur_logger->log(spdlog::level::warn, fmt, std::forward<Args>(args)...);
        }

        template <typename... Args>
        void error(spdlog::format_string_t<Args...> fmt, Args &&...args)
        {
            cur_logger->log(spdlog::level::err, fmt, std::forward<Args>(args)...);
        }

        template <typename... Args>
        void critical(spdlog::format_string_t<Args...> fmt, Args &&...args)
        {
            cur_logger->log(spdlog::level::critical, fmt, std::forward<Args>(args)...);
        }
    };

    extern Logger logger;
}

#ifndef OPTIM_EXPORT_DLL
#include "logger.cpp"
#endif
#endif /* _OPTIM_LOGGING_HPP_ */