#include "layer.hpp"

#include <cstdlib>

#include "hyperparameters.hpp"

Layer::Layer(std::size_t input_size, std::size_t output_size, initialization_function initialization, activation_pair activation)
{
    params.resize((input_size + 1) * output_size);
    preactivation_sums.resize(output_size);
    activations.resize(output_size);
    local_gradients.resize(output_size);

    this->input_size = input_size;
    this->output_size = output_size;
    this->init_function = initialization;
    this->act_pair = activation;
}

void Layer::forward(const std::span<double> input)
{
    for (std::size_t i = 0; i < output_size; ++i)
    {
        std::size_t offset = (input_size + 1) * i;
        double sum = params[offset++];
        for (std::size_t i_in = 0; i_in < input_size; ++i_in)
            sum = fma(params[offset + i_in], input[i_in], sum);
        preactivation_sums[i] = sum;
        activations[i] = act_pair.base(sum);
    }
}

void Layer::backpropagate(const std::span<double> error)
{
    for (std::size_t i = 0; i < output_size; ++i)
        local_gradients[i] = act_pair.derivative(preactivation_sums[i]) * error[i];
}

void Layer::calculate_local_error(std::span<double> error)
{
    for (std::size_t i = 0; i < input_size; ++i)
    {
        double error_sum = 0;
        for (std::size_t j = 0; j < output_size; ++j)
        {
            double w = params[1 + (input_size + 1) * j + i];
            double d = local_gradients[j];
            error_sum = fma(d, w, error_sum);
        }
        error[i] = error_sum;
    }
}
