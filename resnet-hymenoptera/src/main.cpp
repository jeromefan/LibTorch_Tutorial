#include <iostream>
#include "resnet.h"
#include "loadData.h"

int main()
{
    std::cout << "Deep Residual Network for Imagefolder Dataset" << std::endl;

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << std::endl;

    // Hyper Parameters
    const int64_t num_classes = 2;
    const int64_t batch_size = 64;
    const int64_t image_size = 32;
    const size_t num_epochs = 20;
    const double learning_rate = 1e-3;
    const double weight_decay = 1e-3;
    const std::string data_path = "/home/jerome/data/hymenoptera_data";

    // Imagefolder Dataset
    auto train_dataset = dataset::ImageFolderDataset(data_path, image_size, dataset::ImageFolderDataset::Mode::TRAIN)
                             .map(torch::data::transforms::Stack<>());
    auto val_dataset = dataset::ImageFolderDataset(data_path, image_size, dataset::ImageFolderDataset::Mode::VAL)
                           .map(torch::data::transforms::Stack<>());

    // Number of samples in the dataset
    auto num_train_samples = train_dataset.size().value();
    auto num_val_samples = val_dataset.size().value();

    // Data Loaders
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);
    auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(val_dataset), batch_size);

    std::cout << "Data Loaded!" << std::endl;

    // Model
    std::array<int64_t, 4> layers{2, 2, 2, 2};
    resnet::ResNet<resnet::ResidualBlock> model(layers, num_classes);
    model->to(device);

    // Optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Start Training..." << std::endl;
    for (size_t epoch = 0; epoch != num_epochs; ++epoch)
    {
        double running_loss = 0.0;
        for (auto &batch : *train_loader)
        {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
            auto output = model->forward(data);
            auto loss = torch::nn::functional::cross_entropy(output, target);
            running_loss += loss.item<double>() * data.size(0);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
        auto sample_mean_loss = running_loss / num_train_samples;
        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                  << "]  Training Loss: " << sample_mean_loss << std::endl;
    }

    std::cout << "Start Validation..." << std::endl;
    model->eval();
    torch::NoGradGuard no_grad;
    size_t num_correct = 0;
    for (const auto &batch : *val_loader)
    {
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);
        auto output = model->forward(data);
        auto prediction = output.argmax(1);
        num_correct += prediction.eq(target).sum().item<int64_t>();
    }
    auto val_accuracy = static_cast<double>(num_correct) / num_val_samples;
    std::cout << "Validation Accuracy: " << val_accuracy << std::endl;

    return 0;
}