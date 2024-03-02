#include <misc/logger.hpp>

namespace optim
{
    static const spdlog::level::level_enum
        verbosity_to_log_level[] = {
            spdlog::level::off,
            spdlog::level::err,
            spdlog::level::warn,
            spdlog::level::info,
            spdlog::level::trace};

    // std::shared_ptr<spdlog::sinks::basic_file_sink_mt> Logger::file_sink{};
    // std::shared_ptr<spdlog::sinks::ostream_sink_mt> Logger::ostream_sink{};

    Logger logger{};

    Logger::Logger()
    {
        cur_type = LogType::CONSOLE;
        cur_logger = spdlog::stdout_color_mt("console");
        cur_logger->set_pattern("[%^%L%$] %v");
        cur_logger->set_level(spdlog::level::warn);
    }

    void Logger::write_to_console()
    {
        cur_type = LogType::CONSOLE;
        cur_logger = spdlog::stdout_color_mt("console");
    }

    void Logger::write_to_file(const char *filename)
    {
        cur_type = LogType::FILE;
        cur_logger = spdlog::basic_logger_mt("file_logger", filename);
    }

    void Logger::write_to_oss(std::ostream &os)
    {
        cur_type = LogType::OSTREAM;
        auto ostream_sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(os);
        cur_logger = std::make_shared<spdlog::logger>("custom_logger", ostream_sink);
    }

    void Logger::set_verbosity(int verbosity)
    {
        if (verbosity > 4)
            cur_logger->set_level(verbosity_to_log_level[4]);
        else if (verbosity < 0)
            cur_logger->set_level(spdlog::level::warn);
        else
            cur_logger->set_level(verbosity_to_log_level[verbosity]);
    }

    void Logger::set_verbosity(const char *level)
    {
        cur_logger->set_level(spdlog::level::from_str(level));
    }

    void Logger::set_pattern(const char *pattern)
    {
        cur_logger->set_pattern(pattern);
    }
}