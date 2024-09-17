#include "algorithms/linear_regression.h"
#include <numeric> // for std::accumulate

namespace algorithms {

void LinearRegression::fit(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.empty()) {
        throw std::invalid_argument("Input vectors must be of the same size and non-empty.");
    }

    double x_mean = mean(x);
    double y_mean = mean(y);

    double numerator = 0.0;
    double denominator = 0.0;

    for (size_t i = 0; i < x.size(); ++i) {
        numerator += (x[i] - x_mean) * (y[i] - y_mean);
        denominator += (x[i] - x_mean) * (x[i] - x_mean);
    }

    if (denominator == 0.0) {
        throw std::runtime_error("Denominator in slope calculation is zero.");
    }

    m_slope = numerator / denominator;
    m_intercept = y_mean - m_slope * x_mean;
}

double LinearRegression::predict(double x) const {
    return m_slope * x + m_intercept;
}

double LinearRegression::mean(const std::vector<double>& v) const {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

} // namespace algorithms
