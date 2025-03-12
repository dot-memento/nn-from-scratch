#include "layer.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "hyperparameters.h"

layer* layer_create(size_t input_size, size_t output_size, initialization_function initialization, activation_pair activation)
{
    size_t data_block_size = output_size * input_size + output_size;
    layer *new_layer = malloc(sizeof(layer) + data_block_size * sizeof(double));

    *new_layer = (layer) {
        .input_size = input_size,
        .output_size = output_size,
        .initialization_function = initialization,
        .activation_pair = activation,
        .parameter_count = output_size * input_size + output_size,
        .weights = new_layer->data,
        .biases = new_layer->data + output_size * input_size
    };

    return new_layer;
}

void layer_free(layer *layer)
{
    free(layer);
}
