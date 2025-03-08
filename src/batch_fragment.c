#include "batch_fragment.h"

#include <stdlib.h>

#include "layer.h"
#include "network.h"

batch_fragment* batch_fragment_create(neural_network *network)
{
    batch_fragment *fragment = malloc(sizeof(batch_fragment) + network->layer_count * sizeof(struct layer_data));
    fragment->network = network;
    for (size_t i = 0; i < network->layer_count; ++i)
    {
        layer *current_layer = network->layers[i];
        struct layer_data *layer_data = malloc(sizeof(struct layer_data) + 3 * current_layer->output_size * sizeof(double));
        layer_data->params = current_layer->data;
        layer_data->preactivation_sums = layer_data->data;
        layer_data->activations = layer_data->preactivation_sums + current_layer->output_size;
        layer_data->local_gradients = layer_data->activations + current_layer->output_size;
        fragment->layers[i] = layer_data;
    }
    return fragment;
}

void batch_fragment_free(batch_fragment *fragment)
{
    for (size_t i = 0; i < fragment->network->layer_count; ++i)
        free(fragment->layers[i]);
    free(fragment);
}

void batch_fragment_forward(batch_fragment *fragment, double *input)
{
    for (size_t i = 0; i < fragment->network->layer_count; ++i)
    {
        layer *current_layer = fragment->network->layers[i];
        current_layer->forward(current_layer, input);
        input = current_layer->activations;
    }
}
