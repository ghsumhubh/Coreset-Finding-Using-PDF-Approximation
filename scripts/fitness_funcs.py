from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import jensenshannon

import numpy as np
from sklearn.neighbors import KernelDensity
import time
NUMBER_OF_POINTS = 1000 



def reverse_wasserstein_distance(samples_picked, train_data, train_pdfs, is_constant, mins, maxes):
    return -fitness_wasserstein_distance(samples_picked, train_data, train_pdfs, is_constant, mins, maxes)


def fitness_pdf_based(samples_picked, train_data, train_pdfs, is_constant, mins, maxes, distance_function, kde_bandwidth):
    samples_picked = samples_picked.astype(bool)
    samples_picked = samples_picked.flatten()

    feature_count = train_data.shape[1]
    distance = 0
    for i in range(feature_count):
        sample_feature = train_data[samples_picked, i]

        # In case the feature is constant in all train data
        if is_constant[i]:
            continue

        # Otherwise calculate the wasserstein distance
        else:
            if kde_bandwidth == None:
                bandwidth = 'silverman'
            else:
                bandwidth = kde_bandwidth[i]
            kde2 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde2.fit(sample_feature.reshape(-1, 1))  # Reshape to make it a column vector

            # Step 2: Evaluate the KDRs at a set of points
            x = np.linspace(mins[i], maxes[i], NUMBER_OF_POINTS).reshape(-1, 1)  # Reshape to make it a column vector
            pdf1 = train_pdfs[i]
            pdf2 = np.exp(kde2.score_samples(x))

            # Step 3: Calculate the Wasserstein distance
            distance += distance_function(pdf1, pdf2)

    # the fitness is minus the distance
    return -distance




def fitness_wasserstein_distance(samples_picked, train_data, train_pdfs, is_constant, mins, maxes, kde_bandwidth):
    return fitness_pdf_based(samples_picked, train_data, train_pdfs, is_constant, mins, maxes, wasserstein_distance, kde_bandwidth)

def fitness_kl_divergence(samples_picked, train_data, train_pdfs, is_constant, mins, maxes, kde_bandwidth):
    return fitness_pdf_based(samples_picked, train_data, train_pdfs, is_constant, mins, maxes, entropy, kde_bandwidth)

def fitness_js_divergence(samples_picked, train_data, train_pdfs, is_constant, mins, maxes, kde_bandwidth):
    return fitness_pdf_based(samples_picked, train_data, train_pdfs, is_constant, mins, maxes, jensenshannon, kde_bandwidth)


# Calculate the pdfs for all features in the training data in order to not have to do it for sampling
def full_train_pdf(train_data, kde_bandwidth=None):

    feature_count = train_data.shape[1]
    pdfs = []
    mins = []
    maxes = []
    is_constant = []
    for i in range(feature_count):
        feature = train_data[:, i]
        mins.append(min(feature))
        maxes.append(max(feature))

        # In case the feature is constant in all train data
        if np.unique(feature).shape[0] == 1:
            pdfs.append(np.zeros(NUMBER_OF_POINTS))
            is_constant.append(True)

        # Otherwise calculate the wasserstein distance
        else:
            # Step 1: Create KDR for each feature
            if kde_bandwidth == None:
                bandwidth = 'silverman'
            else:
                bandwidth = kde_bandwidth[i]
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde.fit(feature.reshape(-1, 1))  # Reshape to make it a column vector

            # Step 2: Evaluate the KDRs at a set of points
            x = np.linspace(min(feature), max(feature), NUMBER_OF_POINTS).reshape(-1, 1)  # Reshape to make it a column vector
            pdf = np.exp(kde.score_samples(x))  # Use score_samples to get log likelihood and convert to probabilities
            pdfs.append(pdf)
            is_constant.append(False)

    return pdfs, is_constant, mins, maxes
