#include <iostream>
#include <vector>
#include "algorithms/kmeans.h"

int main() {
    // Define data points (each point is 1D in this example)
    std::vector<std::vector<double>> data = {
        {0.5}, {-0.3}, {0.8}, {-1.2}, {0.0}, {2.5}, {3.0}, {2.8}, {-2.0}, {1.5}
    };

    // Instantiate the KMeans class with 2 clusters
    algorithms::KMeans kmeans(2);

    // Fit the KMeans algorithm to the data
    kmeans.fit(data);

    // Retrieve cluster labels and centroids
    std::vector<int> labels = kmeans.getLabels();
    std::vector<std::vector<double>> centroids = kmeans.getCentroids();

    // Print cluster labels for each data point
    std::cout << "KMeans Labels:" << std::endl;
    for (size_t i = 0; i < labels.size(); i++) {
        std::cout << "Data Point " << i << " is in Cluster " << labels[i] << std::endl;
    }

    // Print the centroids of each cluster
    std::cout << "Cluster Centroids:" << std::endl;
    for (size_t i = 0; i < centroids.size(); i++) {
        std::cout << "Centroid " << i << ": " << centroids[i][0] << std::endl;
    }

    return 0;
}
