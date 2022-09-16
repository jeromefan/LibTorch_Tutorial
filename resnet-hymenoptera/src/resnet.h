#ifndef RESNET_H
#define RESNET_H

#include "residual_block.h"
namespace resnet
{
    template <typename Block>
    class ResNetImpl : public torch::nn::Module
    {
    public:
        explicit ResNetImpl(const std::array<int64_t, 4> &layers, int64_t num_classes);
        torch::Tensor forward(torch::Tensor x);

    private:
        int64_t in_channels = 64;
        torch::nn::Conv2d conv{conv7x7(3, 64)};
        torch::nn::BatchNorm2d bn{64};
        torch::nn::ReLU relu{true};
        torch::nn::MaxPool2d maxpool{torch::nn::MaxPool2dOptions(3).stride(2).padding(1)};
        torch::nn::Sequential layer1;
        torch::nn::Sequential layer2;
        torch::nn::Sequential layer3;
        torch::nn::Sequential layer4;
        torch::nn::AdaptiveAvgPool2d avg_pool{1};
        torch::nn::Linear fc;

        torch::nn::Sequential make_layer(int64_t out_channels, int64_t blocks, int64_t stride = 1);
    };

    template <typename Block>
    ResNetImpl<Block>::ResNetImpl(const std::array<int64_t, 4> &layers,
                                  int64_t num_classes)
        : layer1(make_layer(64, layers[0])),
          layer2(make_layer(128, layers[1], 2)),
          layer3(make_layer(256, layers[2], 2)),
          layer4(make_layer(512, layers[3], 2)),
          fc(512, num_classes)
    {
        register_module("conv1", conv);
        register_module("bn1", bn);
        register_module("relu", relu);
        register_module("maxpool", maxpool);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("avg_pool", avg_pool);
        register_module("fc", fc);
    }

    template <typename Block>
    torch::Tensor ResNetImpl<Block>::forward(torch::Tensor x)
    {
        x = conv->forward(x);
        x = bn->forward(x);
        x = relu->forward(x);
        x = maxpool->forward(x);
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        x = avg_pool->forward(x);
        x = x.view({x.size(0), -1});
        x = fc->forward(x);
        return x;
    }

    template <typename Block>
    torch::nn::Sequential ResNetImpl<Block>::make_layer(int64_t out_channels, int64_t blocks, int64_t stride)
    {
        torch::nn::Sequential layers;
        torch::nn::Sequential downsample{nullptr};

        if (stride != 1 || in_channels != out_channels)
        {
            downsample = torch::nn::Sequential{conv1x1(in_channels, out_channels), torch::nn::BatchNorm2d(out_channels)};
        }

        layers->push_back(Block(in_channels, out_channels, stride, downsample));

        in_channels = out_channels;

        for (int64_t i = 1; i != blocks; ++i)
        {
            layers->push_back(Block(out_channels, out_channels));
        }

        return layers;
    }

    template <typename Block = ResidualBlock>
    class ResNet : public torch::nn::ModuleHolder<ResNetImpl<Block>>
    {
    public:
        using torch::nn::ModuleHolder<ResNetImpl<Block>>::ModuleHolder;
    };
}
#endif
