#include <filesystem>
#include <opencv2/opencv.hpp>
#include "loadData.h"

namespace fs = std::filesystem;

namespace dataset
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

    std::unordered_map<std::string, int> create_class_to_index_map(const std::vector<std::string> &classes)
    {
        std::unordered_map<std::string, int> class_to_index;

        int index = 0;

        for (const auto &class_name : classes)
        {
            class_to_index[class_name] = index++;
        }

        return class_to_index;
    }

    std::vector<std::pair<std::string, int>> create_samples(
        const std::string &directory,
        const std::unordered_map<std::string, int> &class_to_index)
    {
        std::vector<std::pair<std::string, int>> samples;

        for (const auto &[class_name, class_index] : class_to_index)
        {
            for (const auto &p : fs::directory_iterator(directory + "/" + class_name))
            {
                if (p.is_regular_file())
                {
                    samples.emplace_back(p.path().string(), class_index);
                }
            }
        }

        return samples;
    }

    ImageFolderDataset::ImageFolderDataset(const std::string &root, int64_t image_size, Mode mode)
        : mode_(mode),
          image_size_(image_size),
          mode_dir_(root + "/" + (mode == Mode::TRAIN ? "train" : "val")),
          classes_(parse_classes(mode_dir_)),
          class_to_index_(create_class_to_index_map(classes_)),
          samples_(create_samples(mode_dir_, class_to_index_)) {}

    torch::optional<size_t> ImageFolderDataset::size() const
    {
        return samples_.size();
    }

    torch::data::Example<> ImageFolderDataset::get(size_t index)
    {
        const auto &[image_path, class_index] = samples_[index];
        cv::Mat img = cv::imread(image_path, 1);
        cv::resize(img, img, cv::Size(image_size_, image_size_));
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_32FC3, 1 / 255.0);
        torch::Tensor tensor_img = torch::from_blob(img.data, {img.rows, img.cols, 3}).toType(torch::kFloat32).permute({2, 0, 1});
        return {tensor_img, torch::tensor(class_index)};
    }
}