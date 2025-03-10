#include "network.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "layer.h"
#include "loss.h"
#include "batch_buffer.h"
#include "dataset.h"
#include "math_utils.h"

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

static void print_epoch_stats(neural_network *network, dataset *ds, size_t epoch_count)
{
    double total_loss = 0;
    double total_accuracy = 0;

    double *result = malloc(ds->output_size * sizeof(double));
    for (size_t entry_idx = 0; entry_idx < ds->entry_count; ++entry_idx)
    {
        double *entry_input = ds->data + ds->entry_size * entry_idx;
        double *entry_output = entry_input + ds->input_size;

        network_infer(network, entry_input, result);
        total_loss += loss_bce.compute_loss(result, entry_output, ds->output_size);

        int correct_bits = 0;
        for (size_t i = 0; i < ds->output_size; ++i)
        {
            if (round(result[i]) == entry_output[i])
                correct_bits++;
        }
        if (correct_bits == ds->output_size)
            total_accuracy++;
    }
    free(result);

    double avg_loss = total_loss / ds->entry_count;
    double accuracy = total_accuracy / ds->entry_count;
    printf("%zu,%f,%f\n", epoch_count, avg_loss, accuracy);
}

void network_train(neural_network *network, const loss_function *loss, dataset *ds, size_t epoch_count, size_t batch_size)
{
    if (network->layer_count == 0)
        return;
    
    batch_buffer **buffers = malloc(sizeof(batch_buffer*) * batch_size);
    for (size_t i = 0; i < batch_size; ++i)
        buffers[i] = batch_buffer_create(network);

    size_t update_counter = 1;

    printf("epoch,loss,accuracy\n");
    for (size_t epoch_idx = 0; epoch_idx < epoch_count; ++epoch_idx)
    {
        print_epoch_stats(network, ds, epoch_idx);
        
        shuffle(ds->data, ds->entry_count, ds->entry_size * sizeof(double));
        for (size_t entry_idx = 0; entry_idx + batch_size <= ds->entry_count;)
        {
            
            for (size_t i = 0; i < batch_size; ++i, ++entry_idx)
            {
                double *entry_input = ds->data + ds->entry_size * entry_idx;
                double *entry_output = entry_input + ds->input_size;

                batch_buffer *buffer = buffers[i];
                batch_buffer_forward(buffer, entry_input);

                struct batch_buffer_layer_data *output_layer_data = buffer->layers[network->layer_count - 1];
                loss->compute_output_gradient(output_layer_data, entry_output);
                
                batch_buffer_backpropagate(buffer);
            }

            batch_buffer_merge(buffers, batch_size);

            batch_buffer_update_params(buffers[0], update_counter++);
        }
    }
    print_epoch_stats(network, ds, epoch_count);

    for (size_t i = 0; i < batch_size; ++i)
        batch_buffer_free(buffers[i]);
    free(buffers);
}
