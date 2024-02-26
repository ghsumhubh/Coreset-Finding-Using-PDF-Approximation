import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



    
def plot_comparison_line(metric_name, dictionaries, labels, baseline_results=None):
    # Plotting data for each dictionary
    for i, data_dict in enumerate(dictionaries):
        sample_sizes = list(data_dict.keys())
        plt.plot(sample_sizes, [np.mean(list(values)) for values in data_dict.values()], label=labels[i])

    # Plotting baseline results
    if baseline_results:
        for baseline, value in baseline_results.items():
            plt.axhline(value[metric_name], color='r', linestyle='--')
            plt.text(sample_sizes[-1], value[metric_name], baseline)

    # Adding labels, title, and legend
    plt.xlabel('Sample Size')
    # change the x axis to sample sizes
    plt.xticks(sample_sizes)

    plt.ylabel(metric_name)
    plt.legend(labels + list(baseline_results.keys()) + ['all data'])

    # Display the plot
    plt.title(f'{metric_name} vs Sample Size')
    plt.show()



def plot_comparison_bar(sample_sizes, avg_dict, std_dict, methods):
    """
    Creates a bar plot showing average and standard deviation per method for each sample rate.

    Args:
        sample_sizes (list): List of sample rates.
        avg_dict (dict): Dictionary mapping sample rates to lists of averages (one per method).
        std_dict (dict): Dictionary mapping sample rates to lists of standard deviations (one per method).
        methods (list): List of methods.
    """

    # Create a Pandas DataFrame to organize the data
    data = {'Sample Size': sample_sizes}
    for method in methods:
        data[f'{method} Avg'] = avg_dict[method]
        data[f'{method} Std'] = std_dict[method]
    df = pd.DataFrame(data)

    # Set the sample rate as the index for easier plotting
    df.set_index('Sample Size', inplace=True)

    # Get bar positions based on the number of methods
    num_methods = len(methods)
    bar_width = 50 / num_methods
    positions = [(i-num_methods / 2) * bar_width for i in range(num_methods) ]


    # Create the plot
    fig, ax = plt.subplots()

    # Plot average bars for each method
    for i, method in enumerate(methods):
        ax.bar(df.index + positions[i], df[f'{method} Avg'], width=bar_width, label=f'{method} Avg')

        # Add error bars for standard deviations
        ax.errorbar(df.index + positions[i], df[f'{method} Avg'], 
                    yerr=df[f'{method} Std'], fmt='none', ecolor='black', capsize=5)

    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Value')
    # only show sample rates that are in the data
    ax.set_xticks(df.index)
    ax.set_title('Average and Standard Deviation per Method')
    ax.legend()

    plt.show()

