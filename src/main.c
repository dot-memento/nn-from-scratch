#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#include "network.h"
#include "dataset.h"
#include "loss.h"
#include "adamw.h"
#include "hyperparameters.h"
#include "math_utils.h"

dataset* dataset_create()
{
    dataset *ds = malloc(sizeof(dataset));
    ds->entry_count = 512;
    ds->input_size = 1;
    ds->output_size = 1;
    ds->data = malloc(ds->entry_count * (ds->input_size + ds->output_size) * sizeof(double));
    ds->entry_size = ds->input_size + ds->output_size;
    for (size_t i = 0; i < ds->entry_count; ++i)
    {
        double *entry_input = ds->data + ds->entry_size * i;
        double *entry_output = entry_input + ds->input_size;
        *entry_input = rand_double_in_range(-1, 1);
        *entry_output = 3 * (*entry_input) * exp(-9 * (*entry_input) * (*entry_input)) + sample_gaussian_distribution(0, 0.1);
    }
    return ds;
}

void dataset_free(dataset *ds)
{
    free(ds->data);
    free(ds);
}

int main(int argc, char *argv[])
{
    unsigned int seed = (unsigned int)time(NULL);
    srand(seed);
    fprintf(stderr, "Using seed: %u\n", seed);

    dataset *ds = dataset_create();
    
    network_layout layout = {
        .input_size = ds->input_size,
        .layers = (struct layer_layout[]) {
            {16,                initialization_he,  activation_swish},
            {16,                initialization_he,  activation_swish},
            {8,                 initialization_he,  activation_swish},
            {ds->output_size,   initialization_xavier,  activation_linear},
            {0}
        }
    };

    neural_network *network = network_create(&layout);
    network->loss = &loss_mse;
    network_initialize(network);

    adamw *optimizer = adamw_create(network->parameter_count, LEARNING_RATE, ADAMW_BETA_MOMENTUM, ADAMW_BETA_VARIANCE, ADAMW_EPSILON, ADAMW_WEIGHT_DECAY, true);

    FILE *loss = fopen("loss.csv", "w");
    FILE *final_output = fopen("scatter.csv", "w");

    training_options options = {
        .epoch_count = 100,
        .batch_size = 8,
        .loss_output = loss,
        .final_output = final_output
    };

    network_train(network, optimizer, ds, options);

    fclose(loss);
    fclose(final_output);

    adamw_free(optimizer);

    network_free(network);
    dataset_free(ds);
    
    return 0;
}