#ifndef LAYER_H
#define LAYER_H

#include <cstddef>
#include <vector>
#include <span>

#include "initialization.hpp"
#include "activation.hpp"

class Layer
{
public:
    Layer(std::size_t input_size, std::size_t output_size, initialization_function initialization, activation_pair activation);
    virtual ~Layer() = default;

    void initialize_parameters() { init_function(this); }

    void forward(const std::span<double> input);

    void backpropagate(const std::span<double> error);
    void calculate_local_error(std::span<double> error);

    constexpr std::size_t get_input_size() const { return input_size; }
    constexpr std::size_t get_output_size() const { return output_size; }
    inline std::size_t get_param_count() const { return params.size(); }

    constexpr std::span<double> get_params() { return params; }
    constexpr std::span<double> get_preactivation_sums() { return preactivation_sums; }
    constexpr std::span<double> get_activations() { return activations; }
    constexpr std::span<double> get_local_gradients() { return local_gradients; }

    constexpr activation_pair get_activation_pair() { return act_pair; }

private:
    std::size_t input_size, output_size;
    
    initialization_function init_function;
    activation_pair act_pair;

    std::vector<double> params;
    std::vector<double> preactivation_sums;
    std::vector<double> activations;
    std::vector<double> local_gradients;
};


#endif // LAYER_H