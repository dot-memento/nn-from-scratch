#include "layer.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "hyperparameters.h"


double* get_weights(layer *layer)
{
    return layer->data;
}

double* get_pre_activation_sums(layer *layer)
{
    return layer->data + layer->output_size * (layer->input_size + 1);
}

double* get_activations(layer *layer)
{
    return layer->data + layer->output_size * (layer->input_size + 2);
}

double* get_local_gradient(layer *layer)
{
    return layer->data + layer->output_size * (layer->input_size + 3);
}

double* get_momentum(layer *layer)
{
    return layer->data + layer->output_size * (layer->input_size + 4);
}

double* get_variance(layer *layer)
{
    return layer->data + layer->output_size * (2 * layer->input_size + 5);
}

double* get_highest_variance(layer *layer)
{
    return layer->data + layer->output_size * (3 * layer->input_size + 6);
}


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
        double sum = get_weights(layer)[offset++];
        for (size_t i_in = 0; i_in < layer->input_size; ++i_in)
            sum = fmaf(get_weights(layer)[offset + i_in], input[i_in], sum);
        get_pre_activation_sums(layer)[i_out] = sum;
        get_activations(layer)[i_out] = layer->activation_pair.base(sum);
    }
}

void layer_output_gradient_sigmoid_bce(layer *layer, const double error[])
{
    for (size_t i = 0; i < layer->output_size; ++i)
        get_local_gradient(layer)[i] = error[i];
}

void layer_output_gradient_mse(layer *layer, const double error[])
{
    for (size_t i = 0; i < layer->output_size; ++i)
        get_local_gradient(layer)[i] = layer->activation_pair.derivative(get_pre_activation_sums(layer)[i]) * error[i];
}

void layer_backpropagate(layer *layer, const double error[])
{
    for (size_t i = 0; i < layer->output_size; ++i)
        get_local_gradient(layer)[i] = layer->activation_pair.derivative(get_pre_activation_sums(layer)[i]) * error[i];
}

void layer_calculate_local_error(layer *next_layer, double error[])
{
    for (size_t i = 0; i < next_layer->input_size; ++i)
    {
        double error_sum = 0;
        for (size_t j = 0; j < next_layer->output_size; ++j)
        {
            double w = get_weights(next_layer)[1 + (next_layer->input_size + 1) * j + i];
            double d = get_local_gradient(next_layer)[j];
            error_sum = fmaf(d, w, error_sum);
        }
        error[i] = error_sum;
    }
}

static void adjust_parameter(double *param, double *m, double *v, double *max_v_hat, double g, size_t batch_index, double weight_decay)
{
    *m = ADAMW_BETA_MOMENTUM * *m + (1 - ADAMW_BETA_MOMENTUM) * g;
    *v = ADAMW_BETA_VARIANCE * *v + (1 - ADAMW_BETA_VARIANCE) * g * g;

    double m_hat = *m / (1 - powf(ADAMW_BETA_MOMENTUM, batch_index));
    double v_hat = *v / (1 - powf(ADAMW_BETA_VARIANCE, batch_index));

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
            &get_weights(layer)[offset],
            &get_momentum(layer)[offset],
            &get_variance(layer)[offset],
            &get_highest_variance(layer)[offset],
            get_local_gradient(layer)[i_out],
            batch_index,
            0.0
        );

        offset++;

        // Adjust weights for each input
        for (size_t i_in = 0; i_in < layer->input_size; ++i_in)
        {
            adjust_parameter(
                &get_weights(layer)[offset + i_in],
                &get_momentum(layer)[offset + i_in],
                &get_variance(layer)[offset + i_in],
                &get_highest_variance(layer)[offset + i_in],
                get_local_gradient(layer)[i_out] * previous[i_in],
                batch_index,
                ADAMW_WEIGHT_DECAY
            );
        }
    }
}
