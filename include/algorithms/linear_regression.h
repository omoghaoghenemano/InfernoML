#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <vector>
#include <stdexcept>

namespace algorithms {

class LinearRegression {
public:
    // Constructor
    LinearRegression() : m_slope(0.0), m_intercept(0.0) {}

    // Fit the model to the training data
    void fit(const std::vector<double>& x, const std::vector<double>& y);

    // Predict the output for a given input
    double predict(double x) const;

    // Getters for the parameters
    double getSlope() const { return m_slope; }
    double getIntercept() const { return m_intercept; }

private:
    double m_slope;
    double m_intercept;

    // Helper function to compute the mean of a vector
    double mean(const std::vector<double>& v) const;
};

} // namespace algorithms

#endif // LINEAR_REGRESSION_H
