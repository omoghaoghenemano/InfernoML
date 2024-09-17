#include <iostream>
#include <vector>
#include "activation/activation_functions.h"  // Include your header file

int main() {
    // Define input vector
    std::vector<double> inputs = {0.5, -0.3, 0.8, -1.2, 0.0};

    // Apply sigmoid function to the inputs
    std::vector<double> sigmoid_results = activation::ActivationFunctions::apply(inputs, activation::ActivationFunctions::sigmoid);

    // Apply ReLU function to the inputs
    std::vector<double> relu_results = activation::ActivationFunctions::apply(inputs, activation::ActivationFunctions::relu);

    // Print results
    std::cout << "Sigmoid Results:" << std::endl;
    for (double result : sigmoid_results) {
        std::cout << result << " ";
    }
    std::cout << std::endl;

    std::cout << "ReLU Results:" << std::endl;
    for (double result : relu_results) {
        std::cout << result << " ";
    }
    std::cout << std::endl;

    return 0;
}
