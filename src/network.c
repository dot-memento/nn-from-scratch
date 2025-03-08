#include "network.h"

#include <stdlib.h>
#include <stdio.h>

#include "layer.h"

neural_network* network_create(network_layout *layout)
{
    // Allocate memory for the network and initialize parameters.
    neural_network *network = malloc(sizeof(neural_network) + layout->layer_count * sizeof(layer*));
    network->input_size = layout->input_size;
    network->largest_layer = 0;
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

        // Update largest_layer if needed.
        if (network->largest_layer < layout->layers[i].neuron_count)
            network->largest_layer = layout->layers[i].neuron_count;
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

double* network_infer(neural_network *network, double *input)
{
    // Forward propagate through layers.
    for (size_t i = 0; i < network->layer_count; ++i)
    {
        layer *current_layer = network->layers[i];
        current_layer->forward(current_layer, input);
        input = get_activations(current_layer);
    }
    return input;
}

void network_train(neural_network *network, double *inputs, double *expected)
{
    if (network->layer_count == 0)
        return;

    network->batch_count++;

    // Forward Propagation: propagate input and save activations.
    double *layer_input = inputs;
    for (size_t i = 0; i < network->layer_count; ++i)
    {
        layer *current_layer = network->layers[i];
        current_layer->forward(current_layer, layer_input);
        layer_input = get_activations(current_layer);
    }

    // Compute Output Error.
    layer *output_layer = network->layers[network->layer_count - 1];
    double *output_activations = get_activations(output_layer);
    double *error = malloc(network->largest_layer * sizeof(double));
    for (size_t i = 0; i < output_layer->output_size; ++i)
        error[i] = output_activations[i] - expected[i];

    // Backward Propagation.
    // Set local gradient for the output layer.
    for (size_t i = 0; i < output_layer->output_size; ++i)
        get_local_gradient(output_layer)[i] = error[i];

    // Propagate error backwards.
    for (size_t i = network->layer_count - 1; i > 0; --i)
    {
        layer *current_layer = network->layers[i];
        layer *prev_layer = network->layers[i - 1];
        layer_calculate_local_error(current_layer, error);
        prev_layer->backward(prev_layer, error);
    }

    // Weight Update: adjust weights for each layer.
    layer_input = inputs;
    for (size_t i = 0; i < network->layer_count; ++i)
    {
        layer *current_layer = network->layers[i];
        layer_adjust_weights(current_layer, layer_input, network->batch_count);
        layer_input = get_activations(current_layer);
    }

    free(error);
}
