#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <cstddef>
#include <span>
#include <vector>
#include <memory>

class Layer;
class LossFunction;
class NeuralNetwork;

class Optimizer
{
public:
    Optimizer(double alpha, double beta1, double beta2, double epsilon, double weight_decay, LossFunction *loss_function);
    virtual ~Optimizer() = default;

    void initialize(NeuralNetwork *network);

    void update_params(NeuralNetwork *network, std::span<double> input);
    void calculate_output_gradient(Layer *layer, std::span<double> expected);

private:
    struct LayerData
    {
        LayerData(std::size_t size)
        {
            m.resize(size, 0);
            v.resize(size, 0);
            v_hat.resize(size, 0);
        }

        std::vector<double> m;
        std::vector<double> v;
        std::vector<double> v_hat;
    };

    void adjust_layer(Layer *layer, LayerData *optimizer_data, const std::span<double> previous);

    double alpha;        // Learning rate
    double beta1;        // Exponential decay rate for the first moment
    double beta2;        // Exponential decay rate for the second moment
    double epsilon;      // Small constant for numerical stability
    double weight_decay; // Weight decay parameter

    std::vector<LayerData> layer_data;
    unsigned long step_count;

    LossFunction *loss_function;
};

#endif /* OPTIMIZER_H */