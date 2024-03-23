from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import seaborn as sns

def ensure_output_folder():
    if not os.path.exists('output'):
        os.makedirs('output')

    if not os.path.exists('output/plots'):
        os.makedirs('output/plots')

    if not os.path.exists('output/raw_numbers'):
        os.makedirs('output/raw_numbers')

    if not os.path.exists('output/plots/distributions'):
        os.makedirs('output/plots/distributions')

    if not os.path.exists('output/plots/correlation_heatmaps'):
        os.makedirs('output/plots/correlation_heatmaps')

def create_distributions(dataset_name, df, processed):
    ensure_output_folder()

    # Create subfolder for distributions
    subfolder_path = f'output/plots/distributions/{dataset_name}'
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    if processed:
        subfolder_path += '/processed'
    else:
        subfolder_path += '/preprocessed'
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # Get distribution for each feature
    for column in df.columns:
        # For binary features (0 or 1)
        if len(df[column].unique()) <= 2:
            df[column].value_counts(normalize=True).plot(kind='bar')
        # For categorical features
        elif not pd.api.types.is_numeric_dtype(df[column]):
            value_counts = df[column].value_counts(normalize=True)
            if len(value_counts) > 20:
                top_19 = value_counts.iloc[:19]
                other = value_counts.iloc[19:].sum()
                top_19['Other'] = other
                top_19.plot(kind='bar')
            else:
                value_counts.plot(kind='bar')
            plt.xticks(rotation=45)  # Rotate labels to prevent overlap
        # For numerical features
        else:
            df[column].plot.kde()
        
        # remove any char not abc or space in the column name
        plt.title(f'{column}')
        column_name = ''.join(e if e.isalnum() or e.isspace() else '_' for e in column)
        plt.tight_layout()  # Adjust layout to make room for the rotated x-labels
        plt.savefig(f'{subfolder_path}/{column_name}.png')
        plt.close()

def create_correlation_heatmap(dataset_name,df):
    ensure_output_folder()

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
    plt.savefig(f'{subfolder_path}/{dataset_name}_correlation_heatmap.png')
    plt.close()

    