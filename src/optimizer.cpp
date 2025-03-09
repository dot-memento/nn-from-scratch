#include "optimizer.hpp"

#include <cmath>

#include "hyperparameters.hpp"
#include "network.hpp"
#include "layer.hpp"
#include "loss.hpp"

Optimizer::Optimizer(double alpha, double beta1, double beta2, double epsilon, double weight_decay, LossFunction *loss_function)
    : alpha(alpha),
      beta1(beta1),
      beta2(beta2),
      epsilon(epsilon),
      weight_decay(weight_decay),
      step_count(0),
      loss_function(std::move(loss_function))
{
}

void Optimizer::initialize(NeuralNetwork *network)
{
    for (auto &layer : network->get_layers())
        layer_data.emplace_back(layer->get_param_count());
}

static void adjust_parameter(double *param, double *m, double *v, double *max_v_hat, double g, size_t batch_index, double weight_decay)
{
    *m = ADAMW_BETA_MOMENTUM * *m + (1 - ADAMW_BETA_MOMENTUM) * g;
    *v = ADAMW_BETA_VARIANCE * *v + (1 - ADAMW_BETA_VARIANCE) * g * g;

    double m_hat = *m / (1 - pow(ADAMW_BETA_MOMENTUM, batch_index));
    double v_hat = *v / (1 - pow(ADAMW_BETA_VARIANCE, batch_index));

    *max_v_hat = fmax(*max_v_hat, v_hat);

    *param -= LEARNING_RATE * (m_hat / (sqrt(*max_v_hat) + ADAMW_EPSILON) + weight_decay * (*param));
}

void Optimizer::adjust_layer(Layer *layer, LayerData *optimizer_data, const std::span<double> previous)
{
    for (size_t i = 0; i < layer->get_output_size(); ++i)
    {
        size_t offset = (layer->get_input_size() + 1) * i;

        // Adjust bias (no weight decay)
        adjust_parameter(
            &layer->get_params()[offset],
            &optimizer_data->m[offset],
            &optimizer_data->v[offset],
            &optimizer_data->v_hat[offset],
            layer->get_local_gradients()[i],
            step_count,
            0.0
        );

        offset++;

        // Adjust weights for each input
        for (size_t j = 0; j < layer->get_input_size(); ++j)
        {
            adjust_parameter(
                &layer->get_params()[offset + j],
                &optimizer_data->m[offset + j],
                &optimizer_data->v[offset + j],
                &optimizer_data->v_hat[offset + j],
                layer->get_local_gradients()[i] * previous[j],
                step_count,
                ADAMW_WEIGHT_DECAY
            );
        }
    }
}

void Optimizer::update_params(NeuralNetwork *network, std::span<double> input)
{
    step_count++;
    std::span<double> layer_input = input;
    for (std::size_t i = 0; i < network->get_layers().size(); ++i)
    {
        Layer *layer = network->get_layers()[i].get();
        LayerData &optimizer_data = layer_data[i];
        adjust_layer(layer, &optimizer_data, layer_input);
        layer_input = layer->get_activations();
    }
}

void Optimizer::calculate_output_gradient(Layer *output_layer, std::span<double> expected)
{
    std::span<double> output_activations = output_layer->get_activations();
    std::span<double> output_layer_gradient = output_layer->get_local_gradients();
    for (size_t i = 0; i < output_layer->get_output_size(); ++i)
        output_layer_gradient[i] = loss_function->output_gradient(output_activations[i], expected[i]);
}
