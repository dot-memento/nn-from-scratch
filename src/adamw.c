#include "adamw.h"

#include <math.h>
#include <stdlib.h>

#include "layer.h"
#include "network.h"
#include "batch_buffer.h"

adamw* adamw_create(size_t size, double alpha, double beta1, double beta2, double epsilon, double weight_decay, bool amsgrad)
{
    adamw *optimizer = malloc(sizeof(adamw) + 3 * size * sizeof(double));

    *optimizer = (adamw) {
        .alpha = alpha,
        .beta1 = beta1,
        .beta2 = beta2,
        .epsilon = epsilon,
        .weight_decay = weight_decay,
        .amsgrad = amsgrad,
        .t = 0,
        .size = size,
        .m = optimizer->data,
        .v = optimizer->data + size,
        .v_hat = optimizer->data + 2 * size
    };

    return optimizer;
}

void adamw_free(adamw *optimizer)
{
    free(optimizer);
}

static void adjust_parameter(adamw *optimizer, double *param, double *m, double *v, double *max_v_hat, double g, double weight_decay)
{
    *m = optimizer->beta1 * *m + (1 - optimizer->beta1) * g;
    *v = optimizer->beta2 * *v + (1 - optimizer->beta2) * g * g;

    double m_hat = *m * optimizer->m_correction_bias;
    double v_hat = *v * optimizer->v_correction_bias;

    *max_v_hat = optimizer->amsgrad ? fmax(*max_v_hat, v_hat) : v_hat;

    *param -= optimizer->alpha * (m_hat / (sqrt(*max_v_hat) + optimizer->epsilon) + weight_decay * (*param));
}

void adamw_update_params(adamw *optimizer, neural_network *network, batch_buffer *buffer)
{
    optimizer->t++;
    optimizer->m_correction_bias = 1 / (1 - pow(optimizer->beta1, optimizer->t));
    optimizer->v_correction_bias = 1 / (1 - pow(optimizer->beta2, optimizer->t));

    size_t parameter_idx = 0;
    for (size_t layer_idx = 0; layer_idx < network->layer_count; ++layer_idx)
    {
        layer *this_layer = network->layers[layer_idx];
        struct batch_buffer_layer_data *this_layer_data = buffer->layers[layer_idx];

        for (size_t bias_idx = 0; bias_idx < this_layer->output_size; ++bias_idx, ++parameter_idx)
        {
            adjust_parameter(
                optimizer,
                &this_layer->biases[bias_idx],
                &optimizer->m[parameter_idx],
                &optimizer->v[parameter_idx],
                &optimizer->v_hat[parameter_idx],
                this_layer_data->grad_b[bias_idx],
                0.0
            );
        }
        
        size_t weight_count_in_layer = this_layer->input_size * this_layer->output_size;
        for (size_t weight_idx = 0; weight_idx < weight_count_in_layer; ++weight_idx, ++parameter_idx)
        {
            adjust_parameter(
                optimizer,
                &this_layer->weights[weight_idx],
                &optimizer->m[parameter_idx],
                &optimizer->v[parameter_idx],
                &optimizer->v_hat[parameter_idx],
                this_layer_data->grad_W[weight_idx],
                optimizer->weight_decay
            );
        }
    }
}
