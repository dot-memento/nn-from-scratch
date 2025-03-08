#include "activation.h"

#include <math.h>

#include "hyperparameters.h"

double identity(double x)
{
    return x;
}

double identity_derivative(double x)
{
    return 1;
}

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double sigmoid_derivative(double x)
{
    double s = sigmoid(x);
    return s * (1 - s);
}

double tanh_derivative(double x)
{
    double t = tanh(x);
    return 1 - t * t;
}

double relu(double x)
{
    return 0 < x ? 0 : x;
}

double relu_derivative(double x)
{
    return 0 < x ? 0 : 1;
}

double leaky_relu(double x)
{
    return 0 < x ? LEAKY_RELU_LEAK * x : x;
}

double leaky_relu_derivative(double x)
{
    return 0 < x ? LEAKY_RELU_LEAK : 1;
}

double swish(double x)
{
    return x * sigmoid(SWISH_BETA * x);
}

double swish_derivative(double x)
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
