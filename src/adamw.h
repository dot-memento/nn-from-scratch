#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <stddef.h>
#include <stdbool.h>

typedef struct neural_network neural_network;

typedef struct AdamW {
    double alpha;        // Learning rate
    double beta1;        // Exponential decay rate for the first moment
    double beta2;        // Exponential decay rate for the second moment
    double epsilon;      // Small constant for numerical stability
    double weight_decay; // Weight decay parameter
    bool amsgrad;        // Flag to enable AMSGrad

    size_t size;         // Number of parameters
    double *m;           // First moment vector
    double *v;           // Second moment vector
    double *v_hat;       // Maximum of v values for AMSGrad (if enabled)
    
    unsigned long t;     // Time step counter
} AdamW;

AdamW* adamw_create(size_t size, double alpha, double beta1, double beta2, double epsilon, double weight_decay, bool amsgrad);
void adamw_free(AdamW *optimizer);

void adamw_update_params(neural_network *network, AdamW *optimizer);

#endif /* OPTIMIZER_H */