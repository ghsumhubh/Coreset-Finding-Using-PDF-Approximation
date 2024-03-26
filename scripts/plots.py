import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os



    
def plot_comparison_line(metric_name, dictionaries, labels, baseline_results=None, save = False, dataset_name = None):
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
    if save:
        plt.savefig(f'output/plots/xgb_results/{dataset_name}/{metric_name}_line.png')
    else:
        plt.show()

    plt.close()




def plot_comparison_bar(metric_name, sample_sizes, avg_dict, std_dict, methods, save = False, dataset_name = None):

     # Prepare data in a structured format
    data = []
    for sample_size in sample_sizes:
        for method in methods:
            index = sample_sizes.index(sample_size)
            avg = avg_dict[method][index]
            std = std_dict[method][index]
            data.append((sample_size, method, avg, std))
    df = pd.DataFrame(data, columns=['Sample Size', 'Method', 'Average', 'StdDev'])

    # Adjust bar width and spacing based on the number of methods
    num_methods = len(methods)
    total_width = 0.8  # Total width for all bars in a group
    bar_width = total_width / num_methods
    spacing = 0.1  # Space between groups
    group_width = total_width + spacing

    # Initialize plot
    fig, ax = plt.subplots()

    # Plot bars for each method
    for i, method in enumerate(methods):
        method_positions = [p * group_width + i * bar_width for p in range(len(sample_sizes))]
        method_data = df[df['Method'] == method]
        ax.bar(method_positions, method_data['Average'], width=bar_width, label=method, align='center')

        # Add error bars
        ax.errorbar(method_positions, method_data['Average'], yerr=method_data['StdDev'], fmt='none', ecolor='black', capsize=5)

    # Adjusting plot aesthetics and labels
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Value')
    ax.set_xticks([p * group_width + total_width / 2 - bar_width / 2 for p in range(len(sample_sizes))])
    ax.set_xticklabels(sample_sizes)
    ax.set_title(f'{metric_name} per method vs Sample Size')
    ax.legend()

    if save:
        filename = f'output/plots/xgb_results/{dataset_name}/comparison_bar.png'
        plt.savefig(filename)
    else:
        plt.show()

    plt.close()



def do_sim_plots(   dataset_name,
                avg_dict,
                std_dict,
                mse_dicts,
                labels,
                all_data_results,
                baseline_results,
                sample_sizes):
    
    plot_comparison_bar(metric_name='MSE',
                        sample_sizes=sample_sizes,
                        avg_dict=avg_dict,
                        std_dict=std_dict,
                        methods=labels, 
                        save = True,
                        dataset_name=dataset_name)
    



    baseline_results_for_print = baseline_results.copy()
    baseline_results_for_print['All Data']= all_data_results['Testing Metrics']


    plot_comparison_line(metric_name = 'MSE',
                            dictionaries = mse_dicts,
                            labels = labels,
                            baseline_results = baseline_results_for_print,
                            save = True,
                            dataset_name=dataset_name)