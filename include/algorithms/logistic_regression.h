#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <vector>
#include <stdexcept>

namespace algorithms {

class LinearRegression {
public:
    // Constructor
    LinearRegression() : m_slope(0.0), m_intercept(0.0), m_learning_rate(0.01), m_iterations(1000) {}

    // Fit the model to the training data using gradient descent
    void fit(const std::vector<double>& x, const std::vector<double>& y);

    // Predict the output for a given input
    double predict(double x) const;

    // Getters for the parameters
    double getSlope() const { return m_slope; }
    double getIntercept() const { return m_intercept; }

    // Set learning rate and number of iterations
    void setLearningRate(double lr) { m_learning_rate = lr; }
    void setIterations(int it) { m_iterations = it; }

private:
    double m_slope;
    double m_intercept;
    double m_learning_rate;
    int m_iterations;

    // Helper function to compute the mean of a vector
    double mean(const std::vector<double>& v) const;

    // Helper functions for gradient descent
    double computeCost(const std::vector<double>& x, const std::vector<double>& y) const;
    void gradientDescent(const std::vector<double>& x, const std::vector<double>& y);
};

} // namespace algorithms

#endif // LINEAR_REGRESSION_H
