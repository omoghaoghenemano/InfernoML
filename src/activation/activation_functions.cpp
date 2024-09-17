#include "activation/activation_functions.h"
#include <cmath>
#include <algorithm>

namespace activation {

    // Activation functions
    double ActivationFunctions::sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double ActivationFunctions::sigmoid_derivative(double x) {
        double sig = sigmoid(x);
        return sig * (1.0 - sig);
    }

    double ActivationFunctions::tanh(double x) {
        return std::tanh(x);
    }

    double ActivationFunctions::tanh_derivative(double x) {
        double tanh_x = tanh(x);
        return 1.0 - tanh_x * tanh_x;
    }

    double ActivationFunctions::relu(double x) {
        return std::max(0.0, x);
    }

    double ActivationFunctions::relu_derivative(double x) {
        return (x > 0) ? 1.0 : 0.0;
    }

    // Apply an activation function to a vector
    template <typename Func>
    std::vector<double> ActivationFunctions::apply(const std::vector<double>& inputs, Func func) {
        std::vector<double> result;
        result.reserve(inputs.size());
        for (double input : inputs) {
            result.push_back(func(input));
        }
        return result;
    }

    // Explicit template instantiation
    template std::vector<double> ActivationFunctions::apply(const std::vector<double>&, double (*)(double));

} // namespace activation
