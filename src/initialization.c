#include "initialization.h"

#include <math.h>

#include "layer.h"
#include "math_utils.h"

void initialization_xavier(layer *layer)
{
    double delta = sqrt(6. / (layer->input_size + layer->output_size));
    for (size_t i = 0; i < (layer->input_size + 1) * layer->output_size; ++i)
        layer->params[i] = rand_double_in_range(-delta, delta);
}

void initialization_he(layer *layer)
{
    double sigma = 2. / layer->input_size;
    for (size_t i = 0; i < (layer->input_size + 1) * layer->output_size; ++i)
        layer->params[i] = sample_gaussian_distribution(0, sigma);
}
