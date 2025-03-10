#include "network.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "layer.h"
#include "loss.h"
#include "batch_buffer.h"

neural_network* network_create(network_layout *layout)
{
    // Allocate memory for the network and initialize parameters.
    neural_network *network = malloc(sizeof(neural_network) + layout->layer_count * sizeof(layer*));
    network->input_size = layout->input_size;
    network->layer_count = layout->layer_count;
    network->batch_count = 0;

    // Create and add layers using the layout.
    for (size_t i = 0; i < layout->layer_count; ++i)
    {
        network->layers[i] = layer_create(
            (i > 0) ? network->layers[i-1]->output_size : network->input_size,
            layout->layers[i].neuron_count,
            layout->layers[i].initialization_function,
            layout->layers[i].activation_pair
        );
    }

    return network;
}

void network_free(neural_network *network)
{
    // Free all layers then the network itself.
    for (size_t i = 0; i < network->layer_count; ++i)
        layer_free(network->layers[i]);
    free(network);
}

neural_network* network_initialize(neural_network *network)
{
    // Initialize each layer via its initialization_function.
    for (size_t i = 0; i < network->layer_count; ++i)
    {
        layer *current_layer = network->layers[i];
        current_layer->initialization_function(current_layer);
    }
    return network;
}

void network_infer(neural_network *network, double *input, double *output)
{
    batch_buffer *buffer = batch_buffer_create(network);
    batch_buffer_forward(buffer, input);
    memcpy(output, buffer->layers[network->layer_count - 1]->activations, network->layers[network->layer_count - 1]->output_size * sizeof(double));
    batch_buffer_free(buffer);
}

void network_train(neural_network *network, double *input, double *expected)
{
    if (network->layer_count == 0)
        return;

    network->batch_count++;

    batch_buffer *buffer = batch_buffer_create(network);

    // Forward Propagation: propagate input and save activations.
    batch_buffer_forward(buffer, input);

    // Calculate output gradient.
    struct batch_buffer_layer_data *output_layer_data = buffer->layers[network->layer_count - 1];
    loss_bce.compute_output_gradient(output_layer_data, expected);
    
    // Backward Propagation.
    batch_buffer_backpropagate(buffer);

    batch_buffer_merge(&buffer, 1);

    // Weight Update: adjust weights for each layer.
    batch_buffer_update_params(buffer, network->batch_count);

    batch_buffer_free(buffer);
}
