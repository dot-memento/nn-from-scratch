#ifndef BATCH_BUFFER_H
#define BATCH_BUFFER_H

#include <stddef.h>

typedef struct neural_network neural_network;
typedef struct layer layer;

struct batch_buffer_layer_data {
    size_t input_size, output_size;
    const double *input;
    double *preactivation_sums;
    double *activations;
    double *local_gradients;
    double data[];
};

typedef struct batch_buffer {
    size_t layer_count;
    struct batch_buffer_layer_data *layers[];
} batch_buffer;

batch_buffer* batch_buffer_create(neural_network *network);
void batch_buffer_free(batch_buffer *buffer);

void batch_buffer_forward(const neural_network *network, batch_buffer *buffer, const double *input);
void batch_buffer_backpropagate(const neural_network *network, batch_buffer *buffer);

#endif // BATCH_BUFFER_H