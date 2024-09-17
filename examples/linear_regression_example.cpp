#include <iostream>
#include <vector>
#include "algorithms/linear_regression.h"

int main() {
    // Sample data
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {2.0, 4.1, 6.0, 8.1, 10.0};

    // Create and train the LinearRegression model
    algorithms::LinearRegression lr;
    lr.fit(x, y);

    // Display the parameters
    std::cout << "Slope: " << lr.getSlope() << std::endl;
    std::cout << "Intercept: " << lr.getIntercept() << std::endl;

    // Make predictions
    double test_value = 6.0;
    std::cout << "Prediction for " << test_value << ": " << lr.predict(test_value) << std::endl;

    return 0;
}
