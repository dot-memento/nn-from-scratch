#ifndef LAYER_H
#define LAYER_H

#include <stddef.h>
#include <stdio.h>

#include "initialization.h"
#include "activation.h"

typedef struct layer {
    size_t input_size, output_size;
    
    initialization_function initialization_function;
    activation_pair activation_pair;

    size_t parameter_count;
    double *weights;
    double *biases;

    double *momentum;
    double *variance;
    double *highest_variance;
    
    double data[];
} layer;

layer* layer_create(size_t input_size, size_t output_size, initialization_function initialization, activation_pair activation);
void layer_free(layer *layer);

/*double* layer_forward(layer *layer, const double *input);
void layer_backpropagate(layer *this_layer, layer *next_layer);

void batch_adjust_weights(layer *layer, const double *previous, size_t batch_index);*/

#endif // LAYER_H