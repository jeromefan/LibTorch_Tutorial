#ifndef FCNET_H
#define FCNET_H

#include <torch/torch.h>

class FCnetImpl : public torch::nn::Module
{
public:
    FCnetImpl(int64_t input_size, int64_t hidden_size, int64_t num_classes);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
};
TORCH_MODULE(FCnet);

#endif
