#ifndef LOADDATA_H
#define LOADDATA_H
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
namespace dataset
{
    class ImageFolderDataset : public torch::data::datasets::Dataset<ImageFolderDataset>
    {
    public:
        enum class Mode
        {
            TRAIN,
            VAL
        };
        explicit ImageFolderDataset(const std::string &root, int64_t image_size, Mode mode = Mode::TRAIN);
        torch::data::Example<> get(size_t index) override;
        torch::optional<size_t> size() const override;

    private:
        Mode mode_;
        int64_t image_size_;
        std::string mode_dir_;
        std::vector<std::string> classes_;
        std::unordered_map<std::string, int> class_to_index_;
        std::vector<std::pair<std::string, int>> samples_;
    };
}
#endif
