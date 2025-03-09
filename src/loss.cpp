#include "loss.hpp"

#include <cmath>
#include <limits>

double LossBinaryCrossEntropy::calculate(std::span<double> predicted, std::span<double> expected) const
{
    double bce = 0.0;
    double epsilon = std::numeric_limits<double>::epsilon();
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        double p = predicted[i];
        if (p < epsilon) p = epsilon;
        if (p > 1 - epsilon) p = 1 - epsilon;
        bce += - (expected[i] * std::log(p) + (1.0 - expected[i]) * std::log(1.0 - p));
    }
    return bce;
}

double LossBinaryCrossEntropy::output_gradient(double predicted, double expected) const
{
    return predicted - expected;
}
