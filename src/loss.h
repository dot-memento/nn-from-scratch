#ifndef LOSS_H
#define LOSS_H

#include <stddef.h>

typedef struct batch_buffer_layer_data batch_buffer_layer_data;

typedef struct loss_function {
    double (*compute_loss)(const double predicted[], const double expected[], size_t size);
    void (*compute_output_gradient)(batch_buffer_layer_data *output_layer, const double expected[]);
} loss_function;

extern const loss_function loss_bce;
extern const loss_function loss_bce_sigmoid;
extern const loss_function loss_mse;

#endif // LOSS_H