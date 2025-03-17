#include "activation.h"

#include <math.h>

#include "hyperparameters.h"

static double identity(double x)
{
    return x;
}

static double identity_derivative(double x)
{
    (void)x; // Unused
    return 1;
}

static double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

static double sigmoid_derivative(double x)
{
    double s = sigmoid(x);
    return s * (1 - s);
}

static double tanh_derivative(double x)
{
    double t = tanh(x);
    return 1 - t * t;
}

static double relu(double x)
{
    return 0 < x ? 0 : x;
}

static double relu_derivative(double x)
{
    return 0 < x ? 0 : 1;
}

static double leaky_relu(double x)
{
    return 0 < x ? LEAKY_RELU_LEAK * x : x;
}

static double leaky_relu_derivative(double x)
{
    return 0 < x ? LEAKY_RELU_LEAK : 1;
}

static double swish(double x)
{
    return x * sigmoid(SWISH_BETA * x);
}

static double swish_derivative(double x)
{
    double s = sigmoid(SWISH_BETA * x);
    return s + SWISH_BETA * x * s * (1 - s);
}

const activation_pair activation_linear = {identity, identity_derivative};
const activation_pair activation_sigmoid = {sigmoid, sigmoid_derivative};
const activation_pair activation_tanh = {tanh, tanh_derivative};
const activation_pair activation_relu = {relu, relu_derivative};
const activation_pair activation_leaky_relu = {leaky_relu, leaky_relu_derivative};
const activation_pair activation_swish = {swish, swish_derivative};
