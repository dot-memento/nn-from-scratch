#include "layer.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "hyperparameters.h"

layer* layer_create(size_t input_size, size_t output_size, initialization_function initialization, activation_pair activation)
{
    size_t data_block_size = input_size * output_size // Weights
        + output_size // Biases
        + (1 + input_size) * output_size // Momentum (for AdamW)
        + (1 + input_size) * output_size // Variance (for AdamW)
        + (1 + input_size) * output_size; // Highest variance (for AMSGrad)
    size_t total_size = sizeof(layer) + data_block_size * sizeof(double);
    layer *new_layer = malloc(total_size);
    memset(new_layer, 0, total_size);

    new_layer->input_size = input_size;
    new_layer->output_size = output_size;
    new_layer->initialization_function = initialization;
    new_layer->activation_pair = activation;

    new_layer->weights = new_layer->data;
    new_layer->biases = new_layer->weights + input_size * output_size;
    new_layer->momentum = new_layer->biases + output_size;
    new_layer->variance = new_layer->momentum + (1 + input_size) * output_size;
    new_layer->highest_variance = new_layer->variance + (1 + input_size) * output_size;

    return new_layer;
}

void layer_free(layer *layer)
{
    free(layer);
}
