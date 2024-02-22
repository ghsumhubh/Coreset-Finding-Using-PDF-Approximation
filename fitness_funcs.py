from scipy.stats import wasserstein_distance
import numpy as np
from sklearn.neighbors import KernelDensity


def fitness_wasserstein_distance(samples_picked, train_data):
    samples_picked = samples_picked.astype(bool)
    samples_picked = samples_picked.flatten()


    feature_count = train_data.shape[1]
    distance = 0
    for i in range(feature_count):
        feature = train_data[:, i]
        sample_feature = train_data[samples_picked, i]

        # In case the feature is constant in all train data
        if np.unique(feature).shape[0] == 1:
            pass  

        # Otherwise calculate the wasserstein distance
        else:
            # Step 1: Create KDR for each feature
            kde1 = KernelDensity(bandwidth='silverman', kernel='gaussian')
            kde1.fit(feature.reshape(-1, 1))  # Reshape to make it a column vector

            kde2 = KernelDensity(bandwidth='silverman', kernel='gaussian')
            kde2.fit(sample_feature.reshape(-1, 1))  # Reshape to make it a column vector

            # Step 2: Evaluate the KDRs at a set of points
            x = np.linspace(min(feature), max(feature), 1000).reshape(-1, 1)  # Reshape to make it a column vector
            pdf1 = np.exp(kde1.score_samples(x))  # Use score_samples to get log likelihood and convert to probabilities
            pdf2 = np.exp(kde2.score_samples(x))

            # Step 3: Calculate the Wasserstein distance
            distance += wasserstein_distance(pdf1, pdf2)




    return distance