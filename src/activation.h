#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <math.h>

typedef struct batch_buffer_layer_data batch_buffer_layer_data;

typedef struct activation_pair {
    void (*base)(batch_buffer_layer_data *layer);
    void (*derivative)(batch_buffer_layer_data *layer);
} activation_pair;

extern const activation_pair activation_linear;
extern const activation_pair activation_sigmoid;
extern const activation_pair activation_tanh;
extern const activation_pair activation_relu;
extern const activation_pair activation_leaky_relu;
extern const activation_pair activation_swish;
extern const activation_pair activation_softmax;

#endif // ACTIVATION_H