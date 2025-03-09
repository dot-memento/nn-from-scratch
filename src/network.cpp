#include "network.hpp"

#include <cstdlib>
#include <algorithm>

#include "layer.hpp"
#include "optimizer.hpp"


void NeuralNetwork::add_layer(std::size_t neuron_count, initialization_function init_function, activation_pair act_pair)
{
    layers.push_back(
        std::make_unique<Layer>(
            (layers.empty()) ? input_size : layers.back()->get_output_size(),
            neuron_count,
            init_function,
            act_pair
        )
    );

    largest_layer = std::max(largest_layer, neuron_count);
}

void NeuralNetwork::initialize_parameters()
{
    for (auto &layer : layers)
        layer->initialize_parameters();
}

std::span<double> NeuralNetwork::infer(std::span<double> input)
{
    for (auto &layer : layers)
    {
        layer->forward(input);
        input = layer->get_activations();
    }
    return input;
}

void NeuralNetwork::train(Optimizer *optimizer, Dataset *dataset)
{
    if (layers.empty())
        return;

    optimizer->initialize(this);

    for (auto [input, expected] : *dataset)
    {
        // Forward propagation.
        std::span<double> forward_input = input;
        for (auto &layer : layers)
        {
            layer->forward(forward_input);
            forward_input = layer->get_activations();
        }
    
        // Compute output gradient.
        optimizer->calculate_output_gradient(layers.back().get(), expected);
    
        // Backward propagation.
        std::vector<double> error(largest_layer);
        for (size_t i = layers.size() - 1; i > 0; --i)
        {
            layers[i]->calculate_local_error(error);
            layers[i - 1]->backpropagate(error);
        }
    
        // Network parameter update.
        optimizer->update_params(this, input);
    }
}
