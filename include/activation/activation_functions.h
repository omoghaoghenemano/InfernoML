#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <vector>
#include <cmath>

namespace activation {

    // Sigmoid activation function
    inline double sigmoid(double x);

    // Derivative of sigmoid function
    inline double sigmoid_derivative(double x);

    // Tanh activation function
    inline double tanh(double x);

    // Derivative of tanh function
    inline double tanh_derivative(double x);

    // ReLU activation function
    inline double relu(double x);

    // Derivative of ReLU function
    inline double relu_derivative(double x);

    // Apply an activation function to a vector
    template <typename Func>
    std::vector<double> apply(const std::vector<double>& inputs, Func func);

} // namespace activation

#endif // ACTIVATION_FUNCTIONS_H
