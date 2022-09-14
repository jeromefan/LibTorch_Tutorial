#include <iostream>
#include "FCnet.h"

int main()
{
    std::cout << "2 FC Layers Neural Network" << std::endl;

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << std::endl;

    // Hyper Parameters
    const int64_t input_size = 784;
    const int64_t hidden_size = 500;
    const int64_t num_classes = 10;
    const int64_t batch_size = 100;
    const size_t num_epochs = 5;
    const double learning_rate = 0.001;
    const std::string MNIST_data_path = "/home/jerome/data/MNIST/raw/";

    // MNIST Dataset
    auto train_dataset = torch::data::datasets::MNIST(MNIST_data_path)
                             .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                             .map(torch::data::transforms::Stack<>());
    auto test_dataset = torch::data::datasets::MNIST(MNIST_data_path, torch::data::datasets::MNIST::Mode::kTest)
                            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                            .map(torch::data::transforms::Stack<>());

    // Number of samples in the dataset
    auto num_train_samples = train_dataset.size().value();
    auto num_test_samples = test_dataset.size().value();

    // Data Loaders
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);

    std::cout << "Data Loaded!" << std::endl;

    // Neural Network Model
    FCnet model(input_size, hidden_size, num_classes);
    model->to(device);

    // Optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Start Training..." << std::endl;
    for (size_t epoch = 0; epoch != num_epochs; ++epoch)
    {
        // Initialize
        double running_loss = 0.0;

        for (auto &batch : *train_loader)
        {
            auto data = batch.data.view({batch_size, -1}).to(device);
            auto target = batch.target.to(device);

            // Forward Pass
            auto output = model->forward(data);
            auto loss = torch::nn::functional::cross_entropy(output, target);

            // Update running loss
            running_loss += loss.item<double>() * data.size(0);

            // Backward and Optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
        auto sample_mean_loss = running_loss / num_train_samples;
        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "]" << std::endl
                  << "Training Loss: " << sample_mean_loss << std::endl;
    }

    std::cout << "Start Testing..." << std::endl;
    model->eval();
    torch::NoGradGuard no_grad;
    size_t num_correct = 0;
    for (const auto &batch : *test_loader)
    {
        auto data = batch.data.view({batch_size, -1}).to(device);
        auto target = batch.target.to(device);
        auto output = model->forward(data);
        auto prediction = output.argmax(1);
        num_correct += prediction.eq(target).sum().item<int64_t>();
    }
    auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
    std::cout << "Testing Accuracy: " << test_accuracy << std::endl;

    return 0;
}