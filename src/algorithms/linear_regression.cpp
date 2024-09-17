#include "algorithms/linear_regression.h"
#include <numeric>
#include <cmath>

namespace algorithms {

// Compute the mean of a vector
double LinearRegression::mean(const std::vector<double>& v) const {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

// Compute the cost (Mean Squared Error)
double LinearRegression::computeCost(const std::vector<double>& x, const std::vector<double>& y) const {
    double total_error = 0.0;
    size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        double prediction = m_slope * x[i] + m_intercept;
        double error = prediction - y[i];
        total_error += error * error;
    }
    return total_error / (2 * n); // Mean Squared Error
}

// Perform gradient descent to optimize slope and intercept
void LinearRegression::gradientDescent(const std::vector<double>& x, const std::vector<double>& y) {
    size_t n = x.size();
    for (int i = 0; i < m_iterations; ++i) {
        double slope_gradient = 0.0;
        double intercept_gradient = 0.0;
        for (size_t j = 0; j < n; ++j) {
            double prediction = m_slope * x[j] + m_intercept;
            double error = prediction - y[j];
            slope_gradient += error * x[j];
            intercept_gradient += error;
        }
        slope_gradient /= n;
        intercept_gradient /= n;

        m_slope -= m_learning_rate * slope_gradient;
        m_intercept -= m_learning_rate * intercept_gradient;

        // Optional: Print cost every 100 iterations
        if (i % 100 == 0) {
            double cost = computeCost(x, y);
             //uncomment to see cost progress
            // std::cout << "Iteration " << i << ": Cost " << cost << std::endl;
        }
    }
}

// Fit the model using gradient descent
void LinearRegression::fit(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Input vectors must have the same size.");
    }
    gradientDescent(x, y);
}

// Predict the output for a given input
double LinearRegression::predict(double x) const {
    return m_slope * x + m_intercept;
}

} // namespace algorithms
