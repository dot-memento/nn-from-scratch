#ifndef NETWORK_H
#define NETWORK_H

#include <stddef.h>
#include <stdio.h>

#include "initialization.h"
#include "activation.h"


typedef struct layer layer;

typedef struct network_layout {
    size_t input_size;
    size_t layer_count;
    struct layer_layout {
        size_t neuron_count;
        initialization_function initialization_function;
        activation_pair activation_pair;
    } *layers;
} network_layout;

typedef struct neural_network {
    size_t input_size;
    size_t largest_layer;
    size_t batch_count;
    size_t layer_count;
    layer *layers[];
} neural_network;


neural_network* network_create(network_layout *layout);
void network_free(neural_network *nn);

neural_network* network_initialize(neural_network *nn);

double* network_infer(neural_network *nn, double *input);
void network_train(neural_network *nn, double *inputs, double *expected);

#endif // NETWORK_H