#include "loss.h"

#include <math.h>
#include <float.h>

#include "layer.h"
#include "batch_buffer.h"

static double binary_cross_entropy(const double predicted[], const double expected[], size_t size)
{
    double sum = 0;
    for (size_t i = 0; i < size; ++i)
    {
        double p = fmin(fmax(DBL_MIN, predicted[i]), 1 - DBL_EPSILON);
        sum -= expected[i] * log(p) + (1 - expected[i]) * log(1 - p);
    }
    return sum;
}

static void output_gradient_bce(batch_buffer_layer_data *output_layer_data, const double y_true[])
{
    const layer *output_layer = output_layer_data->parent;
    for (size_t i = 0; i < output_layer->output_size; ++i)
    {
        double y_pred = output_layer_data->activations[i];
        double dL_dy = (y_pred - y_true[i]) / (y_pred * (1 - y_pred));
        double dy_dz = output_layer->activation_pair.derivative(output_layer_data->preactivation_sums[i]);
        output_layer_data->local_gradients[i] = dL_dy * dy_dz;
    }
}

const loss_function loss_bce = {
    .compute_loss = binary_cross_entropy,
    .compute_output_gradient = output_gradient_bce
};

static void output_gradient_bce_sigmoid(batch_buffer_layer_data *output_layer_data, const double y_true[])
{
    const layer *output_layer = output_layer_data->parent;
    for (size_t i = 0; i < output_layer->output_size; ++i)
    {
        double y_pred = output_layer_data->activations[i];
        output_layer_data->local_gradients[i] = y_pred - y_true[i];
    }
}

const loss_function loss_bce_sigmoid = {
    .compute_loss = binary_cross_entropy,
    .compute_output_gradient = output_gradient_bce_sigmoid
};

static double mean_squared_error(const double predicted[], const double expected[], size_t size)
{
    double sum = 0;
    for (size_t i = 0; i < size; ++i)
    {
        double diff = predicted[i] - expected[i];
        sum += diff * diff;
    }
    return sum / size;
}

static void output_gradient_mse(batch_buffer_layer_data *output_layer_data, const double y_true[])
{
    const layer *output_layer = output_layer_data->parent;
    for (size_t i = 0; i < output_layer->output_size; ++i)
    {
        double y_pred = output_layer_data->activations[i];
        double dL_dy = y_pred - y_true[i];
        double dy_dz = output_layer->activation_pair.derivative(output_layer_data->preactivation_sums[i]);
        output_layer_data->local_gradients[i] = dL_dy * dy_dz;
    }
}

const loss_function loss_mse = {
    .compute_loss = mean_squared_error,
    .compute_output_gradient = output_gradient_mse
};
