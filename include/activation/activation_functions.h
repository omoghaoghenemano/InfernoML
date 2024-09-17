#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <vector>

namespace activation {

    class ActivationFunctions {
    public:
        // Activation functions
        static double sigmoid(double x);
        static double sigmoid_derivative(double x);
        static double tanh(double x);
        static double tanh_derivative(double x);
        static double relu(double x);
        static double relu_derivative(double x);

        // Apply an activation function to a vector
        template <typename Func>
        static std::vector<double> apply(const std::vector<double>& inputs, Func func);
    };

} // namespace activation

#endif // ACTIVATION_FUNCTIONS_H
