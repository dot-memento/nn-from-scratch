#include "adamw.h"

#include <math.h>

#include "hyperparameters.h"
#include "layer.h"
#include "network.h"

static void adamw_update_param(double *param, double *m, double *v, double *v_hat, double grad, size_t batch_index, double weight_decay)
{
    *m = (ADAMW_BETA_MOMENTUM * (*m)) + (1 - ADAMW_BETA_MOMENTUM) * grad;
    *v = (ADAMW_BETA_VARIANCE * (*v)) + (1 - ADAMW_BETA_VARIANCE) * grad * grad;

    double m_hat = *m / (1 - pow(ADAMW_BETA_MOMENTUM, batch_index));
    double v_hat_local = *v / (1 - pow(ADAMW_BETA_VARIANCE, batch_index));

    *v_hat = fmax(*v_hat, v_hat_local);

    *param -= LEARNING_RATE * (m_hat / (sqrt(*v_hat) + ADAMW_EPSILON) + weight_decay * (*param));
}

void adamw_update_params(neural_network *network, AdamW *optimizer)
{
    for (size_t i = 0; i < network->layer_count; ++i)
    {
        layer *current_layer = network->layers[i];
        for (size_t i_out = 0; i_out < current_layer->output_size; ++i_out)
        {
            size_t offset = (current_layer->input_size + 1) * i_out;

            // Adjust bias (no weight decay)
            adamw_update_param(
                &get_weights(current_layer)[offset],
                &get_momentum(current_layer)[offset],
                &get_variance(current_layer)[offset],
                &get_highest_variance(current_layer)[offset],
                get_local_gradient(current_layer)[i_out],
                network->batch_count,
                0.0
            );

            offset++;

            // Adjust weights for each input
            for (size_t i_in = 0; i_in < current_layer->input_size; ++i_in)
            {
                adamw_update_param(
                    &get_weights(current_layer)[offset + i_in],
                    &get_momentum(current_layer)[offset + i_in],
                    &get_variance(current_layer)[offset + i_in],
                    &get_highest_variance(current_layer)[offset + i_in],
                    get_local_gradient(current_layer)[i_out],
                    network->batch_count,
                    ADAMW_WEIGHT_DECAY
                );
            }
        }
    }
}
