#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#include "network.h"
#include "dataset.h"
#include "loss.h"

#define MAX_EPOCH 2500
#define DATASET_ENTRIES 256
#define INPUT_SIZE 8

dataset* dataset_create()
{
    dataset *new_dataset = malloc(sizeof(dataset) + DATASET_ENTRIES * (2 * INPUT_SIZE) * sizeof(double));
    new_dataset->entry_count = DATASET_ENTRIES;
    new_dataset->input_size = INPUT_SIZE;
    new_dataset->output_size = INPUT_SIZE;
    new_dataset->entry_size = new_dataset->input_size + new_dataset->output_size;
    for (size_t i = 0; i < DATASET_ENTRIES; ++i)
    {
        double *entry_input = new_dataset->data + new_dataset->entry_size * i;
        double *entry_output = entry_input + new_dataset->input_size;
        for (size_t j = 0; j < INPUT_SIZE; ++j)
        {
            double bit = (i >> j) & 1;
            entry_input[j] = bit ? 1.0 : -1.0;
            entry_output[j] = bit;
        }
    }
    return new_dataset;
}

void dataset_free(dataset *data)
{
    free(data);
}

int main(int argc, char *argv[])
{
    dataset *data = dataset_create();
    
    unsigned int seed = (unsigned int)time(NULL);
    srand(seed);
    fprintf(stderr, "Using seed: %u\n", seed);
    
    network_layout layout = {
        .input_size = data->input_size,
        .layer_count = 4,
        .layers = (struct layer_layout[]) {
            {8,             initialization_xavier,  activation_tanh},
            {5,             initialization_xavier,  activation_tanh},
            {8,             initialization_xavier,  activation_tanh},
            {data->output_size,    initialization_xavier,  activation_sigmoid}
        }
    };

    neural_network *network = network_create(&layout);
    network_initialize(network);
    
    network_train(network, &loss_bce, data, MAX_EPOCH, 8);
    
    network_free(network);
    dataset_free(data);
    
    return 0;
}