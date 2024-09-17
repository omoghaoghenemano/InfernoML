#ifndef KMEANS_H
#define KMEANS_H

#include <vector>

namespace algorithms{

class KMeans {
public:
    // Constructor
    KMeans(int k, int maxIterations = 100);

    // Fit the model to the data
    void fit(const std::vector<std::vector<double>>& data);

    // Get cluster labels
    const std::vector<int>& getLabels() const;

    // Get cluster centers
    const std::vector<std::vector<double>>& getCentroids() const;

private:
    int k;  // Number of clusters
    int maxIterations;  // Maximum number of iterations
    std::vector<std::vector<double>> centroids;  // Cluster centroids
    std::vector<int> labels;  // Cluster labels for each point

    // Randomly initialize centroids from the data points
    void initializeCentroids(const std::vector<std::vector<double>>& data, int dimensions);

    // Assign each point to the nearest centroid
    void assignClusters(const std::vector<std::vector<double>>& data);

    // Update centroids based on the mean of the assigned points
    void updateCentroids(const std::vector<std::vector<double>>& data, int dimensions);

    // Calculate Euclidean distance between two points
    double euclideanDistance(const std::vector<double>& p1, const std::vector<double>& p2) const;

    // Check if the centroids have converged (optional, currently fixed iterations)
    bool converged() const;
};

}

#endif 
