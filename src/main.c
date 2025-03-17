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

network_layout parse_json_for_layout(json_entry *json_root)
{
    network_layout layout;
    
    json_entry *input_size_entry = json_object_get(json_root, "input_size");
    layout.input_size = (size_t)json_as_number(input_size_entry);

    json_entry *layers_entry = json_object_get(json_root, "layers");
    layout.layer_count = json_array_count(layers_entry);

    layout.layers = malloc((layout.layer_count) * sizeof(layer));
    for (size_t i = 0; i < layout.layer_count; ++i)
    {
        json_entry *layer_entry = json_array_get(layers_entry, i);
        //json_entry *type_entry = json_object_get(layer_entry, "type");
        json_entry *units_entry = json_object_get(layer_entry, "units");
        json_entry *activation_entry = json_object_get(layer_entry, "activation");
        json_entry *init_entry = json_object_get(layer_entry, "init");

        layout.layers[i].neuron_count = (size_t)json_as_number(units_entry);

        const char *activation_name = json_as_string(activation_entry);
        if (!strcmp(activation_name, "Swish"))
            layout.layers[i].activation_pair = activation_swish;
        else if (!strcmp(activation_name, "ReLU"))
            layout.layers[i].activation_pair = activation_relu;
        else if (!strcmp(activation_name, "Tanh"))
            layout.layers[i].activation_pair = activation_tanh;
        else if (!strcmp(activation_name, "Sigmoid"))
            layout.layers[i].activation_pair = activation_sigmoid;
        else
            layout.layers[i].activation_pair = activation_linear;

        const char *initialization_name = json_as_string(init_entry);
        if (!strcmp(initialization_name, "He"))
            layout.layers[i].initialization_function = initialization_he;
        else //if (!strcmp(activation_name, "Xavier"))
            layout.layers[i].initialization_function = initialization_xavier;
    }

    return layout;
}

adamw* parse_json_for_optimizer(neural_network *network, json_entry *json_root)
{
    json_entry *optimizer_entry = json_object_get(json_root, "optimizer");

    json_entry *learning_rate_entry = json_object_get(optimizer_entry, "learning_rate");
    double learning_rate = learning_rate_entry ? json_as_number(learning_rate_entry) : LEARNING_RATE;

    json_entry *beta1_entry = json_object_get(optimizer_entry, "beta1");
    double beta1 = beta1_entry ? json_as_number(beta1_entry) : ADAMW_BETA_MOMENTUM;

    json_entry *beta2_entry = json_object_get(optimizer_entry, "beta2");
    double beta2 = beta2_entry ? json_as_number(beta2_entry) : ADAMW_BETA_VARIANCE;

    json_entry *epsilon_entry = json_object_get(optimizer_entry, "epsilon");
    double epsilon = epsilon_entry ? json_as_number(epsilon_entry) : ADAMW_EPSILON;

    json_entry *weight_decay_entry = json_object_get(optimizer_entry, "weight_decay");
    double weight_decay = weight_decay_entry ? json_as_number(weight_decay_entry) : ADAMW_WEIGHT_DECAY;

    adamw *optimizer = adamw_create(network->parameter_count,
        learning_rate,
        beta1,
        beta2,
        epsilon,
        weight_decay,
        true);

    return optimizer;
}

training_options parse_json_for_training_options(json_entry *json_root)
{
    json_entry *training_entry = json_object_get(json_root, "training");

    json_entry *batch_size_entry = json_object_get(training_entry, "batch_size");
    size_t batch_size = batch_size_entry ? json_as_number(batch_size_entry) : 1;

    json_entry *epoch_count_entry = json_object_get(training_entry, "epoch_count");
    size_t epoch_count = epoch_count_entry ? json_as_number(epoch_count_entry) : 100;

    return (training_options) {
        .batch_size = batch_size,
        .epoch_count = epoch_count
    };
}

int main(int argc, char *argv[])
{
    unsigned int seed = (unsigned int)time(NULL);
    srand(seed);
    fprintf(stderr, "Using seed: %u\n", seed);

    json_entry *json_data = json_parse_file("config.json");

    network_layout layout = parse_json_for_layout(json_data);

    dataset ds = (dataset) {
        .input_size = layout.input_size,
        .output_size = layout.layers[layout.layer_count-1].neuron_count
    };
    if (dataset_load_csv("func_dataset.csv", &ds))
        exit(EXIT_FAILURE);

    neural_network *network = network_create(&layout);

    json_entry *loss_function_entry = json_object_get(json_data, "loss_function");
    const char *loss_function_name = json_as_string(loss_function_entry);
    if (!strcmp(loss_function_name, "CrossEntropy"))
        network->loss = &loss_bce;
    else
        network->loss = &loss_mse;
    
    network_initialize(network);

    adamw *optimizer = parse_json_for_optimizer(network, json_data);

    FILE *loss = fopen("loss.csv", "w");
    FILE *final_output = fopen("scatter.csv", "w");

    training_options options = parse_json_for_training_options(json_data);
    options.loss_output = loss;
    options.final_output = final_output;

    json_free(json_data);

    network_train(network, optimizer, &ds, &options);

    fclose(loss);
    fclose(final_output);

    adamw_free(optimizer);

    network_free(network);

    free(ds.data);
    
    return 0;
}