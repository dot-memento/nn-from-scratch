#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>

#include "network.hpp"
#include "optimizer.hpp"
#include "hyperparameters.hpp"
#include "loss.hpp"

constexpr std::size_t MAX_EPOCH = 5000;
constexpr std::size_t DATASET_ENTRIES = 256;
constexpr std::size_t INPUT_SIZE = 8;

static Dataset dataset(DATASET_ENTRIES);

void generate_dataset()
{
    for (std::size_t i = 0; i < DATASET_ENTRIES; ++i)
    {
        dataset[i].first.resize(INPUT_SIZE);
        dataset[i].second.resize(INPUT_SIZE);
        for (std::size_t j = 0; j < INPUT_SIZE; ++j)
        {
            double bit = (i >> j) & 1;
            dataset[i].first[j] = bit;
            dataset[i].second[j] = bit;
        }
    }
}

static std::unique_ptr<LossFunction> loss = std::make_unique<LossBinaryCrossEntropy>();
static Optimizer optimizer(LEARNING_RATE, ADAMW_BETA_MOMENTUM, ADAMW_BETA_VARIANCE, ADAMW_EPSILON, ADAMW_WEIGHT_DECAY, loss.get());

void train_network(NeuralNetwork *network)
{
    std::size_t dataset_size = DATASET_ENTRIES;
    for (std::size_t epoch_count = 0; epoch_count < MAX_EPOCH; ++epoch_count)
    {
        double total_loss = 0;
        double total_accuracy = 0;

        for (std::size_t j = 0; j < dataset_size; ++j)
        {
            std::span<double> result = network->infer(dataset[j].first);
            total_loss += loss->calculate(result, dataset[j].second);

            std::size_t correct_bits = 0;
            for (std::size_t i = 0; i < INPUT_SIZE; ++i)
                correct_bits += (std::round(result[i]) == dataset[j].second[i]);
            if (correct_bits == INPUT_SIZE)
                total_accuracy++;
        }

        double avg_loss = total_loss / dataset_size;
        double accuracy = total_accuracy / dataset_size;
        std::cout << epoch_count << "," << avg_loss << "," << accuracy << "\n";

        std::random_shuffle(dataset.begin(), dataset.end());
        network->train(&optimizer, &dataset);
    }
}

int main(int argc, char *argv[])
{
    generate_dataset();

    unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
    std::srand(seed);
    std::cerr << "Using seed: " << seed << "\n";

    NeuralNetwork network(INPUT_SIZE);
    network.add_layer(8, initialization_xavier, activation_tanh);
    network.add_layer(5, initialization_xavier, activation_tanh);
    network.add_layer(8, initialization_xavier, activation_tanh);
    network.add_layer(INPUT_SIZE, initialization_xavier, activation_sigmoid);

    network.initialize_parameters();

    std::cout << "epoch,loss,accuracy\n";
    train_network(&network);

    return 0;
}
