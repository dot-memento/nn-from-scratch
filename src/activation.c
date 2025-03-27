#include "activation.h"

#include <math.h>

#include "batch_buffer.h"

#define LEAKY_RELU_LEAK 0.01

static void identity(batch_buffer_layer_data *layer)
{
    layer->activations = layer->preactivation_sums;
}

static void identity_derivative(batch_buffer_layer_data *layer)
{
    (void)layer;
}

static void sigmoid(batch_buffer_layer_data *layer)
{
    for (size_t neuron = 0; neuron < layer->output_size; ++neuron)
        layer->activations[neuron] = 1 / (1 + exp(-layer->preactivation_sums[neuron]));
}

static void sigmoid_derivative(batch_buffer_layer_data *layer)
{
    for (size_t neuron = 0; neuron < layer->output_size; ++neuron)
    {
        double s = layer->activations[neuron];
        layer->local_gradients[neuron] *= s * (1 - s);
    }
}

static void tanh_layer(batch_buffer_layer_data *layer)
{
    for (size_t neuron = 0; neuron < layer->output_size; ++neuron)
    {
        double x = layer->preactivation_sums[neuron];
        layer->activations[neuron] = tanh(x);
    }
}

static void tanh_derivative(batch_buffer_layer_data *layer)
{
    for (size_t neuron = 0; neuron < layer->output_size; ++neuron)
    {
        double t = layer->activations[neuron];
        layer->local_gradients[neuron] *= 1 - t * t;
    }
}

static void relu(batch_buffer_layer_data *layer)
{
    for (size_t neuron = 0; neuron < layer->output_size; ++neuron)
    {
        double x = layer->preactivation_sums[neuron];
        layer->activations[neuron] = 0 < x ? x : 0;
    }
}

static void relu_derivative(batch_buffer_layer_data *layer)
{
    for (size_t neuron = 0; neuron < layer->output_size; ++neuron)
        layer->local_gradients[neuron] *= 0 < layer->preactivation_sums[neuron] ? 1 : 0;
}

static void leaky_relu(batch_buffer_layer_data *layer)
{
    for (size_t neuron = 0; neuron < layer->output_size; ++neuron)
    {
        double slope = 0 < layer->preactivation_sums[neuron] ? LEAKY_RELU_LEAK : 0;
        layer->activations[neuron] = slope * layer->preactivation_sums[neuron];
    }
}

static void leaky_relu_derivative(batch_buffer_layer_data *layer)
{
    for (size_t neuron = 0; neuron < layer->output_size; ++neuron)
        layer->local_gradients[neuron] *= 0 < layer->preactivation_sums[neuron] ? LEAKY_RELU_LEAK : 1;
}

static void swish(batch_buffer_layer_data *layer)
{
    for (size_t neuron = 0; neuron < layer->output_size; ++neuron)
    {
        double x = layer->preactivation_sums[neuron];
        layer->activations[neuron] = x / (1 + exp(-x));   
    }
}

static void swish_derivative(batch_buffer_layer_data *layer)
{
    for (size_t neuron = 0; neuron < layer->output_size; ++neuron)
    {
        double y = layer->activations[neuron];
        double x = layer->preactivation_sums[neuron];
        double s = 1 / (1 + exp(-x));
        layer->local_gradients[neuron] *= y + s * (1 - y);
    }
}

static void softmax(batch_buffer_layer_data *layer)
{
    // To avoid numerical instability, we subtract the maximum value from the preactivation sums.
    double max = layer->preactivation_sums[0];
    for (size_t neuron = 1; neuron < layer->output_size; ++neuron)
        if (layer->preactivation_sums[neuron] > max)
            max = layer->preactivation_sums[neuron];
    
    double sum = 0;
    for (size_t neuron = 0; neuron < layer->output_size; ++neuron)
    {
        double x = exp(layer->preactivation_sums[neuron] - max);
        layer->activations[neuron] = x;
        sum += x;
    }
    
    for (size_t neuron = 0; neuron < layer->output_size; ++neuron)
        layer->activations[neuron] /= sum;
}

static void softmax_derivative(batch_buffer_layer_data *layer)
{
    // Actual implementation is in the loss function.
    (void)layer;
}

const activation_pair activation_linear = {identity, identity_derivative};
const activation_pair activation_sigmoid = {sigmoid, sigmoid_derivative};
const activation_pair activation_tanh = {tanh_layer, tanh_derivative};
const activation_pair activation_relu = {relu, relu_derivative};
const activation_pair activation_leaky_relu = {leaky_relu, leaky_relu_derivative};
const activation_pair activation_swish = {swish, swish_derivative};
const activation_pair activation_softmax = {softmax, softmax_derivative};
