#ifndef LOSS_H
#define LOSS_H

#include <stddef.h>

typedef struct batch_buffer_layer_data batch_buffer_layer_data;
typedef struct layer layer;

typedef struct loss_function {
    double (*compute_loss)(const double predicted[], const double expected[], size_t size);
    void (*compute_output_gradient)(const layer *output_layer, batch_buffer_layer_data *output_layer_data, const double expected[]);
} loss_function;

extern const loss_function loss_bce;
extern const loss_function loss_bce_sigmoid;
extern const loss_function loss_cce_softmax;
extern const loss_function loss_mse;

#endif // LOSS_H