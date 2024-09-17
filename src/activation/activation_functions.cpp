#include "activation/activation_functions.h"

namespace activation {

// Sigmoid activation function
inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Derivative of sigmoid function
inline double sigmoid_derivative(double x) {
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

// Tanh activation function
inline double tanh(double x) {
    return std::tanh(x);
}

// Derivative of tanh function
inline double tanh_derivative(double x) {
    double tanh_x = tanh(x);
    return 1.0 - tanh_x * tanh_x;
}

// ReLU activation function
inline double relu(double x) {
    return std::max(0.0, x);
}

// Derivative of ReLU function
inline double relu_derivative(double x) {
    return (x > 0) ? 1.0 : 0.0;
}

// Apply an activation function to a vector
template <typename Func>
std::vector<double> apply(const std::vector<double>& inputs, Func func) {
    std::vector<double> result;
    result.reserve(inputs.size());
    for (double input : inputs) {
        result.push_back(func(input));
    }
    return result;
}

// Explicit template instantiations
template std::vector<double> apply(const std::vector<double>& inputs, double (*func)(double));

} // namespace activation
