from scipy.stats import gaussian_kde
from scipy.stats import wasserstein_distance
import numpy as np

def fitness_wasserstein_distance(samples_picked, train_data):
    samples_picked = samples_picked.astype(bool)
    samples_picked = samples_picked.flatten()


    feature_count = train_data.shape[1]
    distance = 0
    for i in range(feature_count):
        feature = train_data[:, i]
        sample_feature = train_data[samples_picked, i]

        # In case one of them is constant
        if np.unique(feature).shape[0] == 1 or np.unique(sample_feature).shape[0] == 1: #TODO: what is both are constant? Maybe add based on rarity?
            distance += 1000 # TODO: Change this to np.inf if can work
        # Otherwise calculate the wasserstein distance
        else:
            # Step 1: Create KDR for each feature
            kde1 = gaussian_kde(feature, bw_method='silverman')
            kde2 = gaussian_kde(sample_feature, bw_method='silverman')


            # Step 2: Evaluate the KDRs at a set of points
            x = np.linspace(min(feature), max(feature), 1000)
            pdf1 = kde1(x)
            pdf2 = kde2(x)

            # Step 3: Calculate the Wasserstein distance
            distance += wasserstein_distance(pdf1, pdf2)




    return distance