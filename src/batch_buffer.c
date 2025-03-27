#include "batch_buffer.h"

#include <stdlib.h>

#include "layer.h"
#include "network.h"
#include "hyperparameters.h"

batch_buffer* batch_buffer_create(neural_network *network)
{
    batch_buffer *buffer = malloc(sizeof(batch_buffer) + network->layer_count * sizeof(struct batch_buffer_layer_data*));
    if (!buffer) return NULL;

    buffer->layer_count = network->layer_count;
    for (size_t i = 0; i < network->layer_count; ++i)
    {
        layer *current_layer = network->layers[i];
        size_t output_size = current_layer->output_size;

        size_t data_block_size = output_size // Preactivation sums
            + output_size // Activations
            + output_size; // Local gradient
        struct batch_buffer_layer_data *layer_data = malloc(sizeof(struct batch_buffer_layer_data)
            + data_block_size * sizeof(double));
        
        double *preactivation_sums = layer_data->data;
        double *activations        = preactivation_sums + output_size;
        double *local_gradients    = activations        + output_size;

        *layer_data = (struct batch_buffer_layer_data) {
            .input_size = current_layer->input_size,
            .output_size = current_layer->output_size,
            .preactivation_sums = preactivation_sums,
            .activations = activations,
            .local_gradients = local_gradients,
        };
        
        buffer->layers[i] = layer_data;
    }
    return buffer;
}

void batch_buffer_free(batch_buffer *buffer)
{
    for (size_t layer_idx = 0; layer_idx < buffer->layer_count; ++layer_idx)
        free(buffer->layers[layer_idx]);
    free(buffer);
}

void batch_buffer_forward(const neural_network *network, batch_buffer *buffer, const double *input)
{
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
        }
        layer->activation_pair.base(layer_data);
        input = layer_data->activations;
    }
}

void batch_buffer_backpropagate(const neural_network *network, batch_buffer *buffer)
{
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
            this_layer_data->local_gradients[neuron] = error_sum;
        }
        this_layer->activation_pair.derivative(this_layer_data);
    }
}
