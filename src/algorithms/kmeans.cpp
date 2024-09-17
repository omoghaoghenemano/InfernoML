#include "algorithms/kmeans.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <random>


namespace algorithms{

// Constructor
KMeans::KMeans(int k, int maxIterations) : k(k), maxIterations(maxIterations) {}

// Fit the model to the data
void KMeans::fit(const std::vector<std::vector<double>>& data) {
    int numPoints = data.size();
    int dimensions = data[0].size();

    // Randomly initialize centroids
    initializeCentroids(data, dimensions);

    // Run the algorithm for a fixed number of iterations
    for (int iteration = 0; iteration < maxIterations; iteration++) {
        // Step 1: Assign points to the closest centroid
        assignClusters(data);

        // Step 2: Update centroids based on the points assigned to them
        updateCentroids(data, dimensions);

        // Check for convergence (not implemented for simplicity, fixed iterations)
        if (converged()) {
            break;
        }
    }
}

// Get cluster labels
const std::vector<int>& KMeans::getLabels() const {
    return labels;
}

// Get cluster centers
const std::vector<std::vector<double>>& KMeans::getCentroids() const {
    return centroids;
}

// Randomly initialize centroids from the data points
void KMeans::initializeCentroids(const std::vector<std::vector<double>>& data, int dimensions) {
    centroids.resize(k, std::vector<double>(dimensions));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data.size() - 1);

    for (int i = 0; i < k; i++) {
        centroids[i] = data[dis(gen)];
    }
}

// Assign each point to the nearest centroid
void KMeans::assignClusters(const std::vector<std::vector<double>>& data) {
    labels.resize(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        double minDist = std::numeric_limits<double>::max();
        int closestCentroid = 0;
        for (int j = 0; j < k; j++) {
            double dist = euclideanDistance(data[i], centroids[j]);
            if (dist < minDist) {
                minDist = dist;
                closestCentroid = j;
            }
        }
        labels[i] = closestCentroid;
    }
}

// Update centroids based on the mean of the assigned points
void KMeans::updateCentroids(const std::vector<std::vector<double>>& data, int dimensions) {
    std::vector<std::vector<double>> newCentroids(k, std::vector<double>(dimensions, 0.0));
    std::vector<int> pointsPerCentroid(k, 0);

    // Sum the points assigned to each centroid
    for (size_t i = 0; i < data.size(); i++) {
        int centroidIndex = labels[i];
        pointsPerCentroid[centroidIndex]++;
        for (int d = 0; d < dimensions; d++) {
            newCentroids[centroidIndex][d] += data[i][d];
        }
    }

    // Update the centroids by computing the average
    for (int j = 0; j < k; j++) {
        if (pointsPerCentroid[j] > 0) {
            for (int d = 0; d < dimensions; d++) {
                newCentroids[j][d] /= pointsPerCentroid[j];
            }
        }
    }

    centroids = newCentroids;
}

// Calculate Euclidean distance between two points
double KMeans::euclideanDistance(const std::vector<double>& p1, const std::vector<double>& p2) const {
    double sum = 0.0;
    for (size_t i = 0; i < p1.size(); i++) {
        sum += std::pow(p1[i] - p2[i], 2);
    }
    return std::sqrt(sum);
}

// Check if the centroids have converged (optional, currently fixed iterations)
bool KMeans::converged() const {
    // This can be implemented with a threshold to check if centroids have stopped moving
    return false;
}

}