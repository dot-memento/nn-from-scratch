#include "layer.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "hyperparameters.h"

layer* layer_create(size_t input_size, size_t output_size, initialization_function initialization, activation_pair activation)
{
    size_t data_block_size = output_size * input_size // Weights
        + output_size // Biases
        + output_size * (1 + input_size) // Momentum (for AdamW)
        + output_size * (1 + input_size) // Variance (for AdamW)
        + output_size * (1 + input_size); // Highest variance (for AMSGrad)

    size_t total_size = sizeof(layer) + data_block_size * sizeof(double);
    layer *new_layer = malloc(total_size);

    double *weights          = new_layer->data;
    double *biases           = weights  + output_size * input_size;
    double *momentum         = biases   + output_size;
    double *variance         = momentum + output_size * (1 + input_size);
    double *highest_variance = variance + output_size * (1 + input_size);

    *new_layer = (layer) {
        .input_size = input_size,
        .output_size = output_size,
        .initialization_function = initialization,
        .activation_pair = activation,
        .weights = weights,
        .biases = biases,
        .momentum = momentum,
        .variance = variance,
        .highest_variance = highest_variance,
    };
    memset(new_layer->data, 0, data_block_size * sizeof(double));

    return new_layer;
}

void layer_free(layer *layer)
{
    free(layer);
}
