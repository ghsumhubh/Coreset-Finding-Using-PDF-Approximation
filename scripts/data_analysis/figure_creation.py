from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os


def ensure_output_folder():
    if not os.path.exists('output'):
        os.makedirs('output')

    if not os.path.exists('output/plots'):
        os.makedirs('output/plots')

    if not os.path.exists('output/raw_numbers'):
        os.makedirs('output/raw_numbers')

    if not os.path.exists('output/distributions'):
        os.makedirs('output/distributions')

def create_feature_distributions(dataset_name, df):
    ensure_output_folder()

    # create subfolder for distributions
    if not os.path.exists(f'output/plots/{dataset_name}/distributions'):
        os.makedirs(f'output/plots/{dataset_name}/distributions')

    #get kde for each feature
    for i in df.columns:
        # check if feature is only 0 or 1
        if len(df[i].unique()) <3:
            # if so do a normalized bar plot
            df[i].value_counts(normalize=True).plot(kind='bar')
            plt.title('Feature: ' + i)
            plt.savefig(f'output/plots/{dataset_name}/distributions/{i}.png')
            plt.close()
            
        else:
            df[i].plot.kde()
            plt.title('Feature: ' + i)
            plt.savefig(f'output/plots/{dataset_name}/distributions/{i}.png')
            plt.close()


def create_label_distribution(dataset_name, df):
    ensure_output_folder()

    # create subfolder for distributions
    if not os.path.exists(f'output/plots/{dataset_name}/distributions'):
        os.makedirs(f'output/plots/{dataset_name}/distributions')

    for i in df.columns:
        if len(df[i].unique()) <3:
            df[i].value_counts(normalize=True).plot(kind='bar')
            plt.title('Label')
            plt.savefig(f'output/plots/{dataset_name}/distributions/label.png')
            plt.close()
            
        else:
            df[i].plot.kde()
            plt.title('Label')
            plt.savefig(f'output/plots/{dataset_name}/distributions/label.png')
            plt.close()         

    