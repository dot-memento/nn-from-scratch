#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#include "math_utils.h"
#include "network.h"

#define MAX_EPOCH 5000
#define DATASET_ENTRIES 256
#define INPUT_SIZE 8

typedef struct dataset_entry {
    double input[INPUT_SIZE];
    double output[INPUT_SIZE];
} dataset_entry;

static dataset_entry dataset[DATASET_ENTRIES];

void generate_dataset()
{
    for (size_t i = 0; i < DATASET_ENTRIES; ++i)
    {
        for (size_t j = 0; j < INPUT_SIZE; ++j)
        {
            double bit = (i >> j) & 1;
            dataset[i].input[j] = bit;
            dataset[i].output[j] = bit;
        }
    }
}

void train_network(neural_network *network)
{
    size_t dataset_size = DATASET_ENTRIES;
    for (int epoch_count = 0; epoch_count < MAX_EPOCH; ++epoch_count)
    {
        // Shuffle dataset at the beginning of each epoch.
        shuffle(dataset, dataset_size, sizeof(dataset_entry));
    
        double total_loss = 0;
        double total_accuracy = 0;
    
        // === Inference and loss/accuracy calculation ===
        for (size_t j = 0; j < dataset_size; ++j)
        {
            double *result = network_infer(network, dataset[j].input);
            total_loss += binary_cross_entropy(result, dataset[j].output, INPUT_SIZE);
    
            int correct_bits = 0;
            for (size_t i = 0; i < INPUT_SIZE; ++i)
            {
                if (round(result[i]) == dataset[j].output[i])
                    correct_bits++;
            }
            if (correct_bits == INPUT_SIZE)
                total_accuracy++;
        }
    
        double avg_loss = total_loss / dataset_size;
        double accuracy = total_accuracy / dataset_size;
        printf("%d,%f,%f\n", epoch_count, avg_loss, accuracy);
    
        // === Update network with each dataset entry ===
        for (size_t j = 0; j < dataset_size; ++j)
            network_train(network, dataset[j].input, dataset[j].output);
    }
}

int main(int argc, char *argv[])
{
    generate_dataset();
    
    unsigned int seed = (unsigned int)time(NULL);
    srand(seed);
    fprintf(stderr, "Using seed: %u\n", seed);
    
    network_layout layout = {
        .input_size = INPUT_SIZE,
        .layer_count = 4,
        .layers = (struct layer_layout[]) {
            {8,             initialization_xavier,  activation_tanh},
            {5,             initialization_xavier,  activation_tanh},
            {8,             initialization_xavier,  activation_tanh},
            {INPUT_SIZE,    initialization_xavier,  activation_sigmoid}
        }
    };

    neural_network *network = network_create(&layout);

    network_initialize(network);
    
    printf("epoch,loss,accuracy\n");
    train_network(network);
    
    network_free(network);
    
    return 0;
}