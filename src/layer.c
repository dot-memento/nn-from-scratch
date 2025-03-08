#include "layer.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "hyperparameters.h"

layer* layer_create(size_t input_size, size_t output_size, initialization_function initialization, activation_pair activation)
{
    size_t data_block_size = (1 + input_size) * output_size // Weights and biases
        + output_size // Pre-activation sums
        + output_size // Activations
        + output_size // Local gradients
        + (1 + input_size) * output_size // Momentum (for AdamW)
        + (1 + input_size) * output_size // Variance (for AdamW)
        + (1 + input_size) * output_size; // Highest variance (for AMSGrad)
    size_t total_size = sizeof(layer) + data_block_size * sizeof(double);
    layer *new_layer = malloc(total_size);
    memset(new_layer, 0, total_size);

    new_layer->input_size = input_size;
    new_layer->output_size = output_size;
    new_layer->forward = layer_forward;
    new_layer->backward = layer_backpropagate;
    new_layer->initialization_function = initialization;
    new_layer->activation_pair = activation;

    new_layer->params = new_layer->data;
    new_layer->preactivation_sums = new_layer->params + (1 + input_size) * output_size;
    new_layer->activations = new_layer->preactivation_sums + output_size;
    new_layer->local_gradients = new_layer->activations + output_size;
    new_layer->momentum = new_layer->local_gradients + output_size;
    new_layer->variance = new_layer->momentum + (1 + input_size) * output_size;
    new_layer->highest_variance = new_layer->variance + (1 + input_size) * output_size;

    return new_layer;
}

void layer_free(layer *layer)
{
    free(layer);
}


void layer_forward(layer *layer, const double *input)
{
    for (size_t i_out = 0; i_out < layer->output_size; ++i_out)
    {
        size_t offset = (layer->input_size + 1) * i_out;
        double sum = layer->params[offset++];
        for (size_t i_in = 0; i_in < layer->input_size; ++i_in)
            sum = fmaf(layer->params[offset + i_in], input[i_in], sum);
        layer->preactivation_sums[i_out] = sum;
        layer->activations[i_out] = layer->activation_pair.base(sum);
    }
}

void layer_output_gradient_sigmoid_bce(layer *layer, const double error[])
{
    for (size_t i = 0; i < layer->output_size; ++i)
        layer->local_gradients[i] = error[i];
}

void layer_output_gradient_mse(layer *layer, const double error[])
{
    for (size_t i = 0; i < layer->output_size; ++i)
        layer->local_gradients[i] = layer->activation_pair.derivative(layer->preactivation_sums[i]) * error[i];
}

void layer_backpropagate(layer *layer, const double error[])
{
    for (size_t i = 0; i < layer->output_size; ++i)
        layer->local_gradients[i] = layer->activation_pair.derivative(layer->preactivation_sums[i]) * error[i];
}

void layer_calculate_local_error(layer *next_layer, double error[])
{
    for (size_t i = 0; i < next_layer->input_size; ++i)
    {
        double error_sum = 0;
        for (size_t j = 0; j < next_layer->output_size; ++j)
        {
            double w = next_layer->params[1 + (next_layer->input_size + 1) * j + i];
            double d = next_layer->local_gradients[j];
            error_sum = fma(d, w, error_sum);
        }
        error[i] = error_sum;
    }
}

static void adjust_parameter(double *param, double *m, double *v, double *max_v_hat, double g, size_t batch_index, double weight_decay)
{
    *m = ADAMW_BETA_MOMENTUM * *m + (1 - ADAMW_BETA_MOMENTUM) * g;
    *v = ADAMW_BETA_VARIANCE * *v + (1 - ADAMW_BETA_VARIANCE) * g * g;

    double m_hat = *m / (1 - pow(ADAMW_BETA_MOMENTUM, batch_index));
    double v_hat = *v / (1 - pow(ADAMW_BETA_VARIANCE, batch_index));

    *max_v_hat = fmax(*max_v_hat, v_hat);

    *param -= LEARNING_RATE * (m_hat / (sqrt(*max_v_hat) + ADAMW_EPSILON) + weight_decay * (*param));
}

void layer_adjust_weights(layer *layer, const double *previous, size_t batch_index)
{
    for (size_t i_out = 0; i_out < layer->output_size; ++i_out)
    {
        size_t offset = (layer->input_size + 1) * i_out;

        // Adjust bias (no weight decay)
        adjust_parameter(
            &layer->params[offset],
            &layer->momentum[offset],
            &layer->variance[offset],
            &layer->highest_variance[offset],
            layer->local_gradients[i_out],
            batch_index,
            0.0
        );

        offset++;

        // Adjust weights for each input
        for (size_t i_in = 0; i_in < layer->input_size; ++i_in)
        {
            adjust_parameter(
                &layer->params[offset + i_in],
                &layer->momentum[offset + i_in],
                &layer->variance[offset + i_in],
                &layer->highest_variance[offset + i_in],
                layer->local_gradients[i_out] * previous[i_in],
                batch_index,
                ADAMW_WEIGHT_DECAY
            );
        }
    }
}
