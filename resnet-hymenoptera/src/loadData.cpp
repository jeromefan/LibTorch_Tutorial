#include <filesystem>
#include "loadData.h"

namespace fs = std::filesystem;

namespace dataset
{
    namespace
    {
        std::vector<std::string> parse_classes(const std::string &directory)
        {
            std::vector<std::string> classes;

            for (auto &p : fs::directory_iterator(directory))
            {
                if (p.is_directory())
                {
                    classes.push_back(p.path().filename().string());
                }
            }

            std::sort(classes.begin(), classes.end());

            return classes;
        }
    }
}