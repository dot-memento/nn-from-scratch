#ifndef LOSS_H
#define LOSS_H

#include <span>

class LossFunction
{
public:
    virtual double calculate(std::span<double> predicted, std::span<double> expected) const = 0;
    virtual double output_gradient(double predicted, double expected) const = 0;
};

class LossBinaryCrossEntropy : public LossFunction
{
public:
    double calculate(std::span<double> predicted, std::span<double> expected) const override;
    double output_gradient(double predicted, double expected) const override;
};

#endif // LOSS_H