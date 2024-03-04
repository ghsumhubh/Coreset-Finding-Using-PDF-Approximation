from scipy.stats import wasserstein_distance
import numpy as np
from sklearn.neighbors import KernelDensity
import time

NUMBER_OF_POINTS = 1000 # was 1000


def fitness_wasserstein_distance(samples_picked, train_data, train_pdfs, is_constant, mins, maxes):
    samples_picked = samples_picked.astype(bool)
    samples_picked = samples_picked.flatten()


    
    feature_count = train_data.shape[1]
    distance = 0
    for i in range(feature_count):
        feature = train_data[:, i]
        sample_feature = train_data[samples_picked, i]

        # In case the feature is constant in all train data
        if is_constant[i]:
            continue

        # Otherwise calculate the wasserstein distance
        else:
            #starting_time = time.perf_counter()
            kde2 = KernelDensity(bandwidth='silverman', kernel='gaussian')
            kde2.fit(sample_feature.reshape(-1, 1))  # Reshape to make it a column vector

            #print("Time to fit kde for feature ", i, " is: ", time.perf_counter() - starting_time)
            #starting_time = time.perf_counter()

            # Step 2: Evaluate the KDRs at a set of points
            x = np.linspace(mins[i], maxes[i], NUMBER_OF_POINTS).reshape(-1, 1)  # Reshape to make it a column vector
            pdf1 = train_pdfs[i]
            pdf2 = np.exp(kde2.score_samples(x))

            #print("Time to fit and evaluate kde for feature ", i, " is: ", time.perf_counter() - starting_time)
            #starting_time = time.perf_counter()

            # Step 3: Calculate the Wasserstein distance
            distance += wasserstein_distance(pdf1, pdf2)
            #print("Time to calculate wasserstein distance for feature ", i, " is: ", time.perf_counter() - starting_time)




    return distance

def full_train_pdf(train_data):

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
            kde = KernelDensity(bandwidth='silverman', kernel='gaussian')
            kde.fit(feature.reshape(-1, 1))  # Reshape to make it a column vector

            # Step 2: Evaluate the KDRs at a set of points
            x = np.linspace(min(feature), max(feature), NUMBER_OF_POINTS).reshape(-1, 1)  # Reshape to make it a column vector
            pdf = np.exp(kde.score_samples(x))  # Use score_samples to get log likelihood and convert to probabilities
            pdfs.append(pdf)
            is_constant.append(False)

    return pdfs, is_constant, mins, maxes
