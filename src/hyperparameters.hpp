#ifndef HYPERPARAMETERS_H
#define HYPERPARAMETERS_H

constexpr double LEAKY_RELU_LEAK = 0.01;
constexpr double SWISH_BETA = 1.0;

constexpr double LEARNING_RATE = 0.002;
constexpr double ADAMW_WEIGHT_DECAY = 0.001;
constexpr double ADAMW_BETA_MOMENTUM = 0.8;
constexpr double ADAMW_BETA_VARIANCE = 0.99;
constexpr double ADAMW_EPSILON = 1e-8;

#endif // HYPERPARAMETERS_H