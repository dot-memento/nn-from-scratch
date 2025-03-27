#ifndef NETWORK_H
#define NETWORK_H

#include <stddef.h>
#include <stdio.h>
#include <stdbool.h>

#include "initialization.h"
#include "activation.h"
#include "dataset.h"

typedef struct layer layer;
typedef struct loss_function loss_function;
typedef struct adamw adamw;

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
    size_t parameter_count;
    const loss_function *loss;
    size_t layer_count;
    layer *layers[];
} neural_network;

neural_network* network_create(network_layout *layout);

void network_free(neural_network *network);

neural_network* network_initialize(neural_network *network);

void network_infer(neural_network *network, double *input, double *output);

typedef struct training_parameters {
    dataset train_dataset;
    dataset test_dataset;
    size_t epoch_count;
    size_t batch_size;
    FILE *loss_output;
    FILE *final_output;
} training_parameters;

void network_train(neural_network *network, adamw *optimizer, training_parameters *options);

#endif // NETWORK_H