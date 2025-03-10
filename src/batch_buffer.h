#ifndef BATCH_BUFFER_H
#define BATCH_BUFFER_H

#include <stddef.h>

typedef struct neural_network neural_network;
typedef struct layer layer;

struct batch_buffer_layer_data {
    const layer *parent;
    double *input;
    double *preactivation_sums;
    double *activations;
    union {
        double *local_gradients;
        double *grad_b;
    };
    double *grad_W;
    double data[];
};

typedef struct batch_buffer {
    neural_network *network;
    struct batch_buffer_layer_data *layers[];
} batch_buffer;

batch_buffer* batch_buffer_create(neural_network *network);
void batch_buffer_free(batch_buffer *fragment);

void batch_buffer_forward(batch_buffer *fragment, double *input);
void batch_buffer_backpropagate(batch_buffer *fragment);
void batch_buffer_merge(batch_buffer *buffers[], size_t buffer_count);
void batch_buffer_update_params(batch_buffer *fragment, size_t batch_index);

#endif // BATCH_BUFFER_H