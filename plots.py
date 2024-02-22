import numpy as np
import matplotlib.pyplot as plt


def plot_comparison_bar(metric_name, dictionaries, labels):
    # Extracting data from dictionaries
    averages = [np.mean(list(d.values())) for d in dictionaries]  # List of averages
    stds = [np.std(list(d.values())) for d in dictionaries]  # List of standard deviations
    sample_sizes = list(dictionaries[0].keys())  # List of sample sizes

    # Plotting
    width = 0.35  # Width of the bars
    x = np.arange(len(dictionaries[0]))  # X-axis positions for bars

    fig, ax = plt.subplots()

    # Plotting data for each dictionary
    for i, avg in enumerate(averages):
        rects = ax.bar(x + (i - len(dictionaries)//2) * width, avg, width, label=labels[i], yerr=stds[i], capsize=5)
    
    # Adding sample size labels below the bars
    ax.set_xticks(x - width/2)
    ax.set_xticklabels(sample_sizes)

    # Adding labels, title, and legend
    ax.set_xlabel('Sample Size')
    ax.set_ylabel(f'Average {metric_name}')
    ax.set_title(f'{metric_name}')
    ax.legend()

    # Display the plot
    plt.show()