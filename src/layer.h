#ifndef LAYER_H
#define LAYER_H

#include <stddef.h>
#include <stdio.h>

#include "initialization.h"
#include "activation.h"


typedef struct layer layer;

typedef void (*forward_function)(layer *layer, const double *input);
typedef void (*backward_function)(layer *layer, const double *errors);

typedef struct layer {
    size_t input_size, output_size;
    
    initialization_function initialization_function;
    forward_function forward;
    backward_function backward;
    activation_pair activation_pair;

    double *params;
    double *preactivation_sums;
    double *activations;
    double *local_gradients;
    double *momentum;
    double *variance;
    double *highest_variance;
    double data[];
} layer;

layer* layer_create(size_t input_size, size_t output_size, initialization_function initialization, activation_pair activation);
void layer_free(layer *layer);

void layer_forward(layer *layer, const double *input);
void layer_backpropagate(layer *layer, const double error[]);
void layer_calculate_local_error(layer *next_layer, double error[]);

void layer_adjust_weights(layer *layer, const double *previous, size_t batch_index);

#endif // LAYER_H