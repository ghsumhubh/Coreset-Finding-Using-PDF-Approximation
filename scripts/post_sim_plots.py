from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os




def silverman_bandwidth(data, files_concacted = 1):
    # Calculate standard deviation of the data
    std_dev = np.std(data)
    # Number of data points
    n = len(data)/files_concacted
    # Calculate bandwidth using Silverman's rule of thumb
    bandwidth = (4 * std_dev**5 / (3 * n))**(1/5)
    return bandwidth


def get_silverman_bandwitdth_single_feature(df):
    # assumes the df has only one column
    return silverman_bandwidth(df.values)

def get_silverman_bandwitdth(df):
    return [get_silverman_bandwitdth_single_feature(df.iloc[:, i]) for i in range(df.shape[1])]



def plot_kde(values, feature_name, sampler, fixed= False):
        bandwidth = silverman_bandwidth(values)
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        # plot it
        kde.fit(values[:, None])
        min = np.min(values)
        max = np.max(values)
        x = np.linspace(min, max, 100000)
        logprob = kde.score_samples(x[:, None])
        plt.fill_between(x, np.exp(logprob), alpha=0.5)
        plt.title(f'{feature_name} KDE, sampler: {sampler}')
        plt.show()



def plot_comparative_kde(values_original, values_sampled, feature_name, sampler, fixed=False):
    if not fixed:
         kde = KernelDensity(bandwidth='silverman', kernel='gaussian')
    else:
    # the same but compare to kde of values_original
        bandwidth = silverman_bandwidth(values_sampled) 
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    # plot it
    kde.fit(values_original[:, None])
    min = np.min(values_original)
    max = np.max(values_original)
                                                
    x = np.linspace(min, max, 100000)
    logprob = kde.score_samples(x[:, None])
    plt.plot(x, np.exp(logprob), color='black', linestyle='-', linewidth=1, label='original')

    kde.fit(values_sampled[:, None])
    x = np.linspace(min, max, 100000)
    logprob = kde.score_samples(x[:, None])
    plt.fill_between(x, np.exp(logprob), alpha=0.5, label='sampled')
    plt.title(f'{feature_name} KDE, sampler: {sampler}')
    plt.legend()
    plt.show()

def plot_head_to_head_kde(values_original, values_sampler_1, values_sampler_2, feature_name, sampler_1_name, sampler_2_name, fixed=False, files_concacted = 1):
        if not fixed:
            kde = KernelDensity(bandwidth='silverman', kernel='gaussian')
        else:
            bandwidth = silverman_bandwidth(values_sampler_1, files_concacted=files_concacted) 
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


def plot_kde_per_feature_sampled(paths, feature_names, sampler_names):
    dfs = [pd.read_csv(path, header=None) for path in paths]

    for i in range(dfs[0].shape[1]):
        for j in range(len(dfs)):
            df = dfs[j]
            plot_kde(df.iloc[:, i].values, feature_names[i], sampler_names[j])


def plot_comparative_kde_per_feature_sampled(original_path, paths, feature_names, sampler_names, fixed=False):
    original = pd.read_csv(original_path)
    dfs = [pd.read_csv(path, header=None) for path in paths]

    for i in range(original.shape[1]):
        for j in range(len(dfs)):
            df = dfs[j]
            plot_comparative_kde(original.iloc[:, i].values, df.iloc[:, i].values, feature_names[i], sampler_names[j], fixed=fixed)


def plot_head_to_head_kde_per_feature_sampled(original_path, path_1, path_2, feature_names, sampler_1_name, sampler_2_name, fixed=False):
    original = pd.read_csv(original_path)
    df_1 = pd.read_csv(path_1, header=None)
    df_2 = pd.read_csv(path_2, header=None)

    for i in range(original.shape[1]):
        plot_head_to_head_kde(original.iloc[:, i].values, df_1.iloc[:, i].values, df_2.iloc[:, i].values, feature_names[i], sampler_1_name, sampler_2_name, fixed=fixed)




def plot_head_to_head_kde_per_feature_sampled_concacted_substrat(original_path, path_1, path_2, feature_names, sampler_1_name, sampler_2_name, n_files = None, fixed=False):
    original = pd.read_csv(original_path)
    # get all csv files in path_1
    files_concacted = 0
    dfs_1 = []
    for f in os.listdir(path_1):
        if f.endswith('.csv') and (n_files is None or len(dfs_1) < n_files):
            dfs_1.append(pd.read_csv(path_1 + '/' + f, header=None))
            files_concacted += 1

    df_1 = pd.concat(dfs_1, axis=0)
        



    for i in range(original.shape[1]):
        dfs_2 = []
        for f in os.listdir(path_2):
            if f.endswith('.csv') and (n_files is None or len(dfs_2) < n_files):
                new_df = pd.read_csv(path_2 + '/' + f)
                # if has diameter column add
                if feature_names[i] in new_df.columns:
                    dfs_2.append(new_df)
        df_2 = pd.concat(dfs_2, axis=0)
        #print(df_2.head())
        sub_feature_value = df_2[feature_names[i]].values
        plot_head_to_head_kde(original.iloc[:, i].values, df_1.iloc[:, i].values, sub_feature_value, feature_names[i], sampler_1_name, sampler_2_name, fixed=fixed, files_concacted = files_concacted)




def plot_head_to_head_kde_per_feature_sampled_concacted(original_path, path_1, path_2, feature_names, sampler_1_name, sampler_2_name, n_files = None, fixed=False):
    original = pd.read_csv(original_path)
    # get all csv files in path_1
    files_concacted = 0
    dfs_1 = []
    for f in os.listdir(path_1):
        if f.endswith('.csv') and (n_files is None or len(dfs_1) < n_files):
            dfs_1.append(pd.read_csv(path_1 + '/' + f, header=None))
            files_concacted += 1

    df_1 = pd.concat(dfs_1, axis=0)
    
    dfs_2 = []
    for f in os.listdir(path_2):
        if f.endswith('.csv') and (n_files is None or len(dfs_2) < n_files):
            dfs_2.append(pd.read_csv(path_2 + '/' + f, header=None))
    df_2 = pd.concat(dfs_2, axis=0)




    print("Concacted {} files".format(files_concacted))
    for i in range(original.shape[1]):
        plot_head_to_head_kde(original.iloc[:, i].values, df_1.iloc[:, i].values, df_2.iloc[:, i].values, feature_names[i], sampler_1_name, sampler_2_name, fixed=fixed, files_concacted = files_concacted)

def plot_shap(df):

    # Plotting
    plt.figure(figsize=(10, 8))  # Adjust the size as per your preference
    plt.barh(df['Feature'], df['SHAP Importance'], color='skyblue')  # You can choose any color
    plt.xlabel('SHAP Importance')
    plt.title('Feature Importance Ranking')
    plt.grid(True, linestyle='--', alpha=0.6)  # Adding a grid for easier reading; customize as needed

    # Show the plot
    plt.show()