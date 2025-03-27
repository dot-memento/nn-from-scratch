#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "json.h"
#include "network.h"
#include "layer.h"
#include "dataset.h"
#include "loss.h"
#include "adamw.h"
#include "hyperparameters.h"
#include "math_utils.h"
#include "constants.h"
#include "errno.h"

network_layout parse_json_for_layout(const json_value *json_root)
{
    network_layout layout = {0};
    
    json_value *buffer_value;
    double double_value = 1.0;
    json_object_get(json_root, "input_size", &buffer_value);
    json_number_get(buffer_value, &double_value);
    layout.input_size = double_value;

    json_value *layers_entry;
    json_object_get(json_root, "layers", &layers_entry);
    json_array_length(layers_entry, &layout.layer_count);

    layout.layers = malloc((layout.layer_count) * sizeof(layer));
    if (!layout.layers)
    {
        fprintf(stderr, PROGRAM_NAME": error: failed to allocate memory for layers\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < layout.layer_count; ++i)
    {
        json_value *layer_entry;
        if (json_array_get(layers_entry, i, &layer_entry))
        {
            fprintf(stderr, PROGRAM_NAME": error: failed to get layer entry\n");
            exit(EXIT_FAILURE);
        }

        json_object_get(layer_entry, "units", &buffer_value);
        json_number_get(buffer_value, &double_value);
        layout.layers[i].neuron_count = double_value;

        const char *activation_name = "Linear";
        json_object_get(layer_entry, "activation", &buffer_value);
        json_string_get(buffer_value, &activation_name);
        if (!strcmp(activation_name, "Swish"))
            layout.layers[i].activation_pair = activation_swish;
        else if (!strcmp(activation_name, "ReLU"))
            layout.layers[i].activation_pair = activation_relu;
        else if (!strcmp(activation_name, "Tanh"))
            layout.layers[i].activation_pair = activation_tanh;
        else if (!strcmp(activation_name, "Sigmoid"))
            layout.layers[i].activation_pair = activation_sigmoid;
        else if (!strcmp(activation_name, "Softmax"))
            layout.layers[i].activation_pair = activation_softmax;
        else
            layout.layers[i].activation_pair = activation_linear;

        const char *initialization_name = "He";
        json_object_get(layer_entry, "init", &buffer_value);
        json_string_get(buffer_value, &initialization_name);
        if (!strcmp(initialization_name, "He"))
            layout.layers[i].initialization_function = initialization_he;
        else //if (!strcmp(activation_name, "Xavier"))
            layout.layers[i].initialization_function = initialization_xavier;
    }

    return layout;
}

adamw* parse_json_for_optimizer(const neural_network *network, const json_value *json_root)
{
    json_value *optimizer_entry, *buffer_value;
    json_object_get(json_root, "optimizer", &optimizer_entry);

    double learning_rate = LEARNING_RATE;
    if (!json_object_get(optimizer_entry, "learning_rate", &buffer_value))
        json_number_get(buffer_value, &learning_rate);

    double beta1 = ADAMW_BETA_MOMENTUM;
    if (!json_object_get(optimizer_entry, "beta1", &buffer_value))
        json_number_get(buffer_value, &beta1);

    double beta2 = ADAMW_BETA_VARIANCE;
    if (!json_object_get(optimizer_entry, "beta2", &buffer_value))
        json_number_get(buffer_value, &beta2);

    double epsilon = ADAMW_EPSILON;
    if (!json_object_get(optimizer_entry, "epsilon", &buffer_value))
        json_number_get(buffer_value, &epsilon);

    double weight_decay = ADAMW_WEIGHT_DECAY;
    if (!json_object_get(optimizer_entry, "weight_decay", &buffer_value))
        json_number_get(buffer_value, &weight_decay);

    return adamw_create(
        network->parameter_count,
        learning_rate,
        beta1,
        beta2,
        epsilon,
        weight_decay,
        true
    );
}

training_parameters parse_json_for_training_options(const json_value *json_root, const network_layout *layout)
{
    json_value *training_entry, *buffer_value;
    json_object_get(json_root, "training", &training_entry);

    double batch_size = 1.0;
    if (!json_object_get(training_entry, "batch_size", &buffer_value))
        json_number_get(buffer_value, &batch_size);
        
    double epoch_count = 100.0;
    if (!json_object_get(training_entry, "epoch_count", &buffer_value))
        json_number_get(buffer_value, &epoch_count);  
        
    const char *train_dataset_path = "train_dataset.csv";
    if (!json_object_get(training_entry, "train_dataset", &buffer_value))
        json_string_get(buffer_value, &train_dataset_path);

    const char *test_dataset_path = "test_dataset.csv";
    if (!json_object_get(training_entry, "test_dataset", &buffer_value))
        json_string_get(buffer_value, &test_dataset_path);

    dataset train_ds = (dataset) {
        .input_size = layout->input_size,
        .output_size = layout->layers[layout->layer_count-1].neuron_count
    };
    dataset test_ds = train_ds;

    if (dataset_load_csv(train_dataset_path, &train_ds) ||
        dataset_load_csv(test_dataset_path, &test_ds))
    {
        fprintf(stderr, PROGRAM_NAME": error: failed to load training dataset\n");
        exit(EXIT_FAILURE);
    }

    return (training_parameters) {
        .train_dataset = train_ds,
        .test_dataset = test_ds,
        .batch_size = batch_size,
        .epoch_count = epoch_count,
        .loss_output = NULL,
        .final_output = NULL
    };
}

const loss_function* parse_json_for_loss_function(const json_value *json_root)
{
    json_value *loss_function_entry;
    json_object_get(json_root, "loss_function", &loss_function_entry);

    const char *loss_function_name = "MSE";
    json_string_get(loss_function_entry, &loss_function_name);
    if (!strcmp(loss_function_name, "BinaryCrossEntropy"))
        return &loss_bce;
    if (!strcmp(loss_function_name, "CategoricalCrossEntropy"))
        return &loss_cce_softmax;
    else
        return &loss_mse;
}

int main(int argc, char *argv[])
{
    unsigned int seed = (unsigned int)time(NULL);
    srand(seed);
    printf("Using seed: %u\n", seed);
    
    FILE *json_file = fopen("config.json", "r");
    if (!json_file)
    {
        fprintf(stderr, PROGRAM_NAME": error: can't open 'config.json': %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    
    json_value *json_data = NULL;
    json_error error = json_parse_file(json_file, &json_data, NULL);
    if (error)
    {
        fprintf(stderr, PROGRAM_NAME": error: failed to parse JSON file: %s\n", json_error_to_string(error));
        exit(EXIT_FAILURE);
    }

    network_layout layout = parse_json_for_layout(json_data);

    neural_network *network = network_create(&layout);
    network->loss = parse_json_for_loss_function(json_data);

    network_initialize(network);

    adamw *optimizer = parse_json_for_optimizer(network, json_data);

    FILE *loss = fopen("loss.csv", "w");
    FILE *final_output = fopen("scatter.csv", "w");

    training_parameters train_param = parse_json_for_training_options(json_data, &layout);
    train_param.loss_output = loss;
    train_param.final_output = final_output;

    json_free(json_data);

    printf("Starting training...\n");
    network_train(network, optimizer, &train_param);
    printf("Training finished successfully\n");

    fclose(loss);
    fclose(final_output);

    adamw_free(optimizer);

    network_free(network);

    free(train_param.train_dataset.data);
    free(train_param.test_dataset.data);
    
    return 0;
}