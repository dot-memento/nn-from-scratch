#include "batch_buffer.h"

#include <stdlib.h>

#include "layer.h"
#include "network.h"
#include "hyperparameters.h"

batch_buffer* batch_buffer_create(neural_network *network)
{
    batch_buffer *fragment = malloc(sizeof(batch_buffer) + network->layer_count * sizeof(struct batch_buffer_layer_data*));
    fragment->network = network;
    for (size_t i = 0; i < network->layer_count; ++i)
    {
        layer *current_layer = network->layers[i];
        struct batch_buffer_layer_data *layer_data = malloc(sizeof(struct batch_buffer_layer_data)
            + 3 * current_layer->output_size * sizeof(double)
            + current_layer->input_size * current_layer->output_size * sizeof(double));
        layer_data->parent = current_layer;
        layer_data->preactivation_sums = layer_data->data;
        layer_data->activations = layer_data->preactivation_sums + current_layer->output_size;
        layer_data->local_gradients = layer_data->activations + current_layer->output_size;
        layer_data->grad_W = layer_data->local_gradients + current_layer->output_size;
        fragment->layers[i] = layer_data;
    }
    return fragment;
}

void batch_buffer_free(batch_buffer *buffer)
{
    for (size_t layer_idx = 0; layer_idx < buffer->network->layer_count; ++layer_idx)
        free(buffer->layers[layer_idx]);
    free(buffer);
}

void batch_buffer_forward(batch_buffer *buffer, double *input)
{
    neural_network *network = buffer->network;
    for (size_t layer_idx = 0; layer_idx < network->layer_count; ++layer_idx)
    {
        layer *layer = network->layers[layer_idx];
        struct batch_buffer_layer_data *layer_data = buffer->layers[layer_idx];
        layer_data->input = input;
        for (size_t neuron = 0; neuron < layer->output_size; ++neuron)
        {
            double sum = layer->biases[neuron];
            size_t offset = layer->input_size * neuron;
            for (size_t input_idx = 0; input_idx < layer->input_size; ++input_idx)
                sum = fma(layer->weights[offset + input_idx], input[input_idx], sum);
            layer_data->preactivation_sums[neuron] = sum;
            layer_data->activations[neuron] = layer->activation_pair.base(sum);
        }
        input = layer_data->activations;
    }
}

void batch_buffer_backpropagate(batch_buffer *buffer)
{
    neural_network *network = buffer->network;
    for (size_t layer_idx = network->layer_count - 1; layer_idx > 0; --layer_idx)
    {
        layer *next_layer = network->layers[layer_idx];
        layer *this_layer = network->layers[layer_idx - 1];
        struct batch_buffer_layer_data *next_layer_data = buffer->layers[layer_idx];
        struct batch_buffer_layer_data *this_layer_data = buffer->layers[layer_idx - 1];

        for (size_t neuron = 0; neuron < this_layer->output_size; ++neuron)
        {
            double error_sum = 0;
            for (size_t output_idx = 0; output_idx < next_layer->output_size; ++output_idx)
            {
                double w = next_layer->weights[next_layer->input_size * output_idx + neuron];
                double d = next_layer_data->local_gradients[output_idx];
                error_sum = fma(d, w, error_sum);
            }
            this_layer_data->local_gradients[neuron] = this_layer->activation_pair.derivative(this_layer_data->preactivation_sums[neuron]) * error_sum;
        }
    }
}

void batch_buffer_merge(batch_buffer *buffers[], size_t buffer_count)
{
    batch_buffer *main_buffer = buffers[0];
    for (size_t layer_idx = 0; layer_idx < buffers[0]->network->layer_count; ++layer_idx)
    {
        layer *output_layer = main_buffer->network->layers[layer_idx];
        for (size_t neuron = 0; neuron < output_layer->output_size; ++neuron)
            for (size_t input_idx = 0; input_idx < output_layer->input_size; ++input_idx)
            {
                double sum = 0;
                for (size_t buffer_idx = 0; buffer_idx < buffer_count; ++buffer_idx)
                {
                    struct batch_buffer_layer_data *layer_buffer = buffers[buffer_idx]->layers[layer_idx];
                    sum += layer_buffer->local_gradients[neuron] * layer_buffer->input[input_idx];
                }
                main_buffer->layers[layer_idx]->grad_W[neuron * output_layer->input_size + input_idx] = sum;
            }
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

static void layer_adjust_weights(struct batch_buffer_layer_data *this_layer_data, size_t batch_index)
{
    const layer *this_layer = this_layer_data->parent;
    for (size_t neuron = 0; neuron < this_layer->output_size; ++neuron)
    {
        adjust_parameter(
            &this_layer->biases[neuron],
            &this_layer->momentum[neuron],
            &this_layer->variance[neuron],
            &this_layer->highest_variance[neuron],
            this_layer_data->grad_b[neuron],
            batch_index,
            0.0
        );
        
        size_t offset = this_layer->input_size * neuron;
        for (size_t input_idx = 0; input_idx < this_layer->input_size; ++input_idx)
        {
            adjust_parameter(
                &this_layer->weights[offset + input_idx],
                &this_layer->momentum[this_layer->output_size + offset + input_idx],
                &this_layer->variance[this_layer->output_size + offset + input_idx],
                &this_layer->highest_variance[this_layer->output_size + offset + input_idx],
                this_layer_data->grad_W[offset + input_idx],
                batch_index,
                ADAMW_WEIGHT_DECAY
            );
        }
    }
}

void batch_buffer_update_params(batch_buffer *buffer, size_t batch_index)
{
    neural_network *network = buffer->network;
    for (size_t layer_idx = 0; layer_idx < network->layer_count; ++layer_idx)
        layer_adjust_weights(buffer->layers[layer_idx], batch_index);
}
