#ifndef BATCH_FRAGMENT_H
#define BATCH_FRAGMENT_H

#include <stddef.h>

typedef struct neural_network neural_network;

struct layer_data {
    const double *params;
    double *preactivation_sums;
    double *activations;
    double *local_gradients;
    double data[];
};

typedef struct batch_fragment {
    neural_network *network;
    struct layer_data *layers[];
} batch_fragment;

batch_fragment* batch_fragment_create(neural_network *network);
void batch_fragment_free(batch_fragment *fragment);

void batch_fragment_forward(batch_fragment *fragment, double *input);
//void batch_fragment_backward(batch_fragment *fragment, double *input);

#endif // BATCH_FRAGMENT_H