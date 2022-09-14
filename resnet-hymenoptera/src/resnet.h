#ifndef RESNET_H
#define RESNET_H

#include <torch/torch.h>

class ResnetImpl : public torch::nn::Module
{
public:
    ResnetImpl(int64_t input_size, int64_t hidden_size, int64_t num_classes);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
};
TORCH_MODULE(Resnet);

#endif
