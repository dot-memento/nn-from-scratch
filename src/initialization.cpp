#include "initialization.hpp"

#include <cmath>
#include <random>

#include "layer.hpp"

static std::default_random_engine rng(std::random_device{}());

void initialization_xavier(Layer *layer)
{
    double delta = sqrt(6. / (layer->get_input_size() + layer->get_output_size()));
    for (std::size_t i = 0; i < (layer->get_input_size() + 1) * layer->get_output_size(); ++i)
    {
        std::uniform_real_distribution<double> distribution(-delta, delta);
        layer->get_params()[i] = distribution(rng);
    }
}

void initialization_he(Layer *layer)
{
    double sigma = 2. / layer->get_input_size();
    for (std::size_t i = 0; i < (layer->get_input_size() + 1) * layer->get_output_size(); ++i)
    {
        std::normal_distribution<double> distribution(0, sigma);
        layer->get_params()[i] = distribution(rng);
    }
}
