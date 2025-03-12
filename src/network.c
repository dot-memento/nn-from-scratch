#include "network.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "layer.h"
#include "loss.h"
#include "batch_buffer.h"
#include "dataset.h"
#include "math_utils.h"
#include "adamw.h"

neural_network* network_create(network_layout *layout)
{
    size_t layer_count = 0;
    while (layout->layers[layer_count].neuron_count)
        layer_count++;

    // Allocate memory for the network and initialize parameters.
    neural_network *network = malloc(sizeof(neural_network) + layer_count * sizeof(layer*));
    *network = (neural_network) {
        .input_size = layout->input_size,
        .layer_count = layer_count,
    };

    // Create and add layers using the layout.
    size_t parameter_count = 0;
    for (size_t i = 0; i < layer_count; ++i)
    {
        layer *new_layer = layer_create(
            (i > 0) ? network->layers[i-1]->output_size : network->input_size,
            layout->layers[i].neuron_count,
            layout->layers[i].initialization_function,
            layout->layers[i].activation_pair
        );
        network->layers[i] = new_layer;
        parameter_count += new_layer->parameter_count;
    }
    network->parameter_count = parameter_count;

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

static void fprint_epoch_stats(FILE *file, neural_network *network, dataset *ds, size_t epoch_count)
{
    if (file == NULL)
        return;

    double total_loss = 0;

    double *result = malloc(ds->output_size * sizeof(double));
    for (size_t entry_idx = 0; entry_idx < ds->entry_count; ++entry_idx)
    {
        double *entry_input = ds->data + ds->entry_size * entry_idx;
        double *entry_output = entry_input + ds->input_size;

        network_infer(network, entry_input, result);
        total_loss += network->loss->compute_loss(result, entry_output, ds->output_size);
    }
    free(result);

    double avg_loss = total_loss / ds->entry_count;
    fprintf(file, "%zu,%f\n", epoch_count, avg_loss);
}

static void fprint_network_output(FILE *file, neural_network *network, dataset *ds)
{
    fputs("input,expected,predicted\n", file);
    for (size_t entry_idx = 0; entry_idx < ds->entry_count; ++entry_idx)
    {
        double *entry_input = ds->data + ds->entry_size * entry_idx;
        double *entry_output = entry_input + ds->input_size;
        double output;
        network_infer(network, entry_input, &output);
        fprintf(file, "%f,%f,%f\n", *entry_input, *entry_output, output);
    }
}

static void split_dataset(const dataset *ds, dataset *training_ds, dataset *validation_ds, double split_ratio)
{
    *training_ds = (dataset) {
        .entry_count = ds->entry_count * split_ratio,
        .entry_size = ds->entry_size,
        .input_size = ds->input_size,
        .output_size = ds->output_size,
        .data = ds->data
    };

    *validation_ds = (dataset) {
        .entry_count = ds->entry_count - training_ds->entry_count,
        .entry_size = ds->entry_size,
        .input_size = ds->input_size,
        .output_size = ds->output_size,
        .data = ds->data + training_ds->entry_count * ds->entry_size
    };
}

void network_train(neural_network *network, adamw *optimizer, dataset *ds, const training_options *options)
{
    if (network->layer_count == 0)
        return;
    
    size_t batch_size = options->batch_size;
    batch_buffer **buffers = malloc(sizeof(batch_buffer*) * batch_size);
    for (size_t buffer_idx = 0; buffer_idx < batch_size; ++buffer_idx)
        buffers[buffer_idx] = batch_buffer_create(network);

    dataset training_ds;
    dataset validation_ds;
    split_dataset(ds, &training_ds, &validation_ds, 0.8);

    if (options->loss_output != NULL)
        fputs("epoch,loss,accuracy\n", options->loss_output);
    for (size_t epoch_idx = 0; epoch_idx < options->epoch_count; ++epoch_idx)
    {
        fprint_epoch_stats(options->loss_output, network, &validation_ds, epoch_idx);
        
        shuffle(training_ds.data, training_ds.entry_count, training_ds.entry_size * sizeof(double));
        for (size_t entry_idx = 0; entry_idx + batch_size <= training_ds.entry_count;)
        {
            
            for (size_t buffer_idx = 0; buffer_idx < batch_size; ++buffer_idx, ++entry_idx)
            {
                double *entry_input = training_ds.data + training_ds.entry_size * entry_idx;
                double *entry_output = entry_input + training_ds.input_size;

                batch_buffer *buffer = buffers[buffer_idx];
                batch_buffer_forward(buffer, entry_input);

                struct batch_buffer_layer_data *output_layer_data = buffer->layers[network->layer_count - 1];
                network->loss->compute_output_gradient(output_layer_data, entry_output);
                
                batch_buffer_backpropagate(buffer);
            }

            batch_buffer_merge(buffers, batch_size);

            adamw_update_params(optimizer, network, buffers[0]);
        }
    }
    fprint_epoch_stats(options->loss_output, network, &validation_ds, options->epoch_count);

    for (size_t buffer_idx = 0; buffer_idx < batch_size; ++buffer_idx)
        batch_buffer_free(buffers[buffer_idx]);
    free(buffers);

    if (options->final_output != NULL)
        fprint_network_output(options->final_output, network, &validation_ds);
}
