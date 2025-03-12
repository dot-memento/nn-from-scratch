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

    double data[];
} layer;

layer* layer_create(size_t input_size, size_t output_size, initialization_function initialization, activation_pair activation);
void layer_free(layer *layer);

#endif // LAYER_H