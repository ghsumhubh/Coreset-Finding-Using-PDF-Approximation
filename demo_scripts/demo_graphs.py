import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns


def plot_objective_over_generation(objective_array):
    # negate it
    objective_array = [-x for x in objective_array]
    plt.plot(objective_array)
    plt.xlabel('Generation')
    plt.ylabel('Aggregated Distance')
    plt.title('Best Chromosome\'s Aggregated Distance Over Generation')
    plt.show()


def silverman_bandwidth(data):
    # Calculate standard deviation of the data
    std_dev = np.std(data)
    # Number of data points
    n = len(data)
    # Calculate bandwidth using Silverman's rule of thumb
    bandwidth = (4 * std_dev**5 / (3 * n))**(1/5)
    return bandwidth


def plot_head_to_head_kde(original_df, sampler_1_df, sampler_2_df, feature_name, sampler_1_name, sampler_2_name):
        values_original = original_df[feature_name].values
        values_sampler_1 = sampler_1_df[feature_name].values
        values_sampler_2 = sampler_2_df[feature_name].values

        bandwidth = silverman_bandwidth(values_sampler_1) 
        # the same but compare to kde of values_original
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        # plot it

        kde.fit(values_original[:, None])
        min = np.min(values_original)
        max = np.max(values_original)                                         
        x = np.linspace(min, max, 100000)
        logprob = kde.score_samples(x[:, None])
        plt.plot(x, np.exp(logprob), color='black', linestyle='-', linewidth=1, label='original')

        kde.fit(values_sampler_1[:, None])
        x = np.linspace(min, max, 100000)
        logprob = kde.score_samples(x[:, None])
        plt.fill_between(x, np.exp(logprob), alpha=0.5, label=sampler_1_name)

        kde.fit(values_sampler_2[:, None])
        x = np.linspace(min, max, 100000)
        logprob = kde.score_samples(x[:, None])
        plt.fill_between(x, np.exp(logprob), alpha=0.5, label=sampler_2_name)

        plt.title(f'{feature_name} KDE')
        plt.legend()
        plt.show()



def plot_correlation_heatmap(dataset_name,df):

    # Create subfolder for correlation heatmaps
    subfolder_path = f'output/plots/correlation_heatmaps/{dataset_name}'
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # Calculate correlation matrix
    corr_matrix = df.corr()
    should_annotate = len(corr_matrix.columns) <= 15

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=should_annotate, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title(f'Feature Correlation Heatmap: {dataset_name}')
    plt.tight_layout()

    # Save the heatmap
    plt.show()
    plt.close()
