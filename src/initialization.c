#include "initialization.h"

#include <math.h>

#include "layer.h"
#include "math_utils.h"

void initialization_xavier(layer *layer)
{
    double delta = sqrt(6. / (layer->input_size + layer->output_size));
    for (size_t i = 0; i < (layer->input_size) * layer->output_size; ++i)
        layer->weights[i] = rand_double_in_range(-delta, delta);
    for (size_t i = 0; i < layer->output_size; ++i)
        layer->biases[i] = rand_double_in_range(-delta, delta);
}

void initialization_he(layer *layer)
{
    double sigma = 2. / layer->input_size;
    for (size_t i = 0; i < (layer->input_size) * layer->output_size; ++i)
        layer->weights[i] = sample_gaussian_distribution(0, sigma);
    for (size_t i = 0; i < layer->output_size; ++i)
        layer->biases[i] = sample_gaussian_distribution(0, sigma);
}
