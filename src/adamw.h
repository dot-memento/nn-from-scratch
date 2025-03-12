#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <stddef.h>
#include <stdbool.h>

typedef struct neural_network neural_network;
typedef struct batch_buffer batch_buffer;

typedef struct adamw {
    double alpha;        // Learning rate
    double beta1;        // Exponential decay rate for the first moment
    double beta2;        // Exponential decay rate for the second moment
    double epsilon;      // Small constant for numerical stability
    double weight_decay; // Weight decay parameter
    bool amsgrad;        // Flag to enable AMSGrad

    unsigned long t;     // Time step counter
    double m_correction_bias;
    double v_correction_bias;

    size_t size;         // Number of parameters
    double *m;           // First moment vector
    double *v;           // Second moment vector
    double *v_hat;       // Maximum of v values for AMSGrad (if enabled)

    double data[];
} adamw;

adamw* adamw_create(size_t size, double alpha, double beta1, double beta2, double epsilon, double weight_decay, bool amsgrad);
void adamw_free(adamw *optimizer);

void adamw_update_params(adamw *optimizer, neural_network *network, batch_buffer *buffer);

#endif /* OPTIMIZER_H */