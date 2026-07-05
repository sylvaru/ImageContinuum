#pragma once
#include <filesystem>
#include <string>
#include <vector>


namespace
{
    inline std::vector<std::byte> readBinaryFile(
        const std::filesystem::path& path)
    {
        std::filesystem::path candidate = path;
        if (!std::filesystem::exists(candidate))
        {
            std::filesystem::path cursor = std::filesystem::current_path();
            while (!cursor.empty())
            {
                candidate = (cursor / path).lexically_normal();
                if (std::filesystem::exists(candidate))
                {
                    break;
                }

                const std::filesystem::path parent = cursor.parent_path();
                if (parent == cursor)
                {
                    break;
                }
                cursor = parent;
            }
        }

        std::ifstream file(candidate, std::ios::binary | std::ios::ate);
        if (!file)
        {
            throw std::runtime_error(
                "Could not open shader file: " + path.string());
        }

        const std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<std::byte> bytes(static_cast<size_t>(size));
        if (size > 0)
        {
            file.read(
                reinterpret_cast<char*>(bytes.data()),
                size);
        }

        return bytes;
    }

    inline std::string narrow(const wchar_t* text)
    {
        if (!text || text[0] == L'\0')
        {
            return {};
        }

        const int required = WideCharToMultiByte(
            CP_UTF8,
            0,
            text,
            -1,
            nullptr,
            0,
            nullptr,
            nullptr);

        if (required <= 1)
        {
            return {};
        }

        std::string result(static_cast<size_t>(required), '\0');

        WideCharToMultiByte(
            CP_UTF8,
            0,
            text,
            -1,
            result.data(),
            required,
            nullptr,
            nullptr);

        result.pop_back();
        return result;
    }
}