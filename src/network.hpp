#ifndef NETWORK_H
#define NETWORK_H

#include <cstddef>
#include <vector>
#include <memory>
#include <span>

#include "initialization.hpp"
#include "activation.hpp"
#include "layer.hpp"

using Dataset = std::vector<std::pair<std::vector<double>, std::vector<double>>>;

class Optimizer;

class NeuralNetwork
{
public:
    NeuralNetwork(std::size_t input_size) : input_size(input_size), largest_layer(0) {}

    void add_layer(std::size_t neuron_count, initialization_function init_function, activation_pair act_pair);

    void initialize_parameters();
    std::span<double> infer(std::span<double> input);
    void train(Optimizer *optimizer, Dataset *dataset);

    inline std::size_t get_layer_count() const { return layers.size(); }
    constexpr std::span<std::unique_ptr<Layer>> get_layers() { return layers; }

private:
    std::size_t input_size;
    std::size_t largest_layer;
    std::vector<std::unique_ptr<Layer>> layers;
};

#endif // NETWORK_H