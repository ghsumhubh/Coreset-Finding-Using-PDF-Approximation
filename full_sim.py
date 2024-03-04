from scripts.xgb_results import xgb_results_regression, get_all_data_and_baseline_results
from pprint import pprint
from scripts.utils import *
from scripts.get_data import get_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scripts.genetic import GeneticAlgorithmSampler
from scripts.fitness_funcs import *
from IPython.display import clear_output
from scripts.plots import *
import time
import sys

REDUNDANCY = 2 
SAMPLE_SIZES = [100, 200, 300 ,400]


def sample_and_get_results(dataset_id):
    x_train, x_test, y_train, y_test, description = get_dataset(dataset_id) #Dataset 1 has 2 colums we need to predict...

    pprint(description)
    dataset_name = description['dataset name']

    all_data_results, baseline_results = get_all_data_and_baseline_results(x_train, x_test, y_train, y_test)


    mse_dict_random, mse_dict_ga = {}, {}



    for _, sample_size in enumerate(SAMPLE_SIZES):
        clear_output()
        print('Sample Size:', sample_size, '\n')
        
        mse_dict_ga[sample_size] = []
        mse_dict_random[sample_size] = []
        
        for i in range(REDUNDANCY):
            print('Iteration {}/{}'.format(i+1, REDUNDANCY))
            np.random.seed(i)
            
            x_train_sample, y_train_sample = sample_data(x_train, y_train, sample_size)
            
            genetic_sampler = GeneticAlgorithmSampler(
                fitness_function=fitness_wasserstein_distance,
                sample_size=sample_size,
                x_train=x_train,
                y_train=y_train,
                population_size=2, # was 20
                max_generations=1, # was 10
                mutation_rate=0.6,
                mutation_cap=2,
                elite_size=1, # was 2
                verbose=False
            )
            
            x_train_new_sample, y_train_new_sample, history = genetic_sampler.sample()
            
            results_random = xgb_results_regression(x_train_sample, x_test, y_train_sample, y_test)
            results_ga = xgb_results_regression(x_train_new_sample, x_test, y_train_new_sample, y_test)
            
            mse_dict_random[sample_size].append(results_random['Testing Metrics']['MSE'])
            mse_dict_ga[sample_size].append(results_ga['Testing Metrics']['MSE'])


    avg_dict ={}
    std_dict = {}

    avg_dict['Random'] = []
    std_dict['Random'] = []
    avg_dict['GA'] = []
    std_dict['GA'] = []
    for sample_size in SAMPLE_SIZES:
        avg_dict['Random'].append(np.mean(mse_dict_random[sample_size]))
        std_dict['Random'].append(np.std(mse_dict_random[sample_size]))
        avg_dict['GA'].append(np.mean(mse_dict_ga[sample_size]))
        std_dict['GA'].append(np.std(mse_dict_ga[sample_size]))

    return dataset_name, avg_dict, std_dict, mse_dict_random, mse_dict_ga, all_data_results, baseline_results

def do_plots(dataset_name, avg_dict, std_dict, mse_dict_random, mse_dict_ga, all_data_results, baseline_results):
    plot_comparison_bar(metric_name='MSE',
                        sample_sizes=SAMPLE_SIZES,
                        avg_dict=avg_dict,
                        std_dict=std_dict,
                        methods=['Random', 'GA'], 
                        save = True,
                        dataset_name=dataset_name)
    



    baseline_results_for_print = baseline_results.copy()
    baseline_results_for_print['All Data']= all_data_results['Testing Metrics']


    plot_comparison_line(metric_name = 'MSE',
                            dictionaries = [mse_dict_random, mse_dict_ga],
                            labels = ['Random', 'Genetic Algorithm'],
                            baseline_results = baseline_results_for_print,
                            save = True,
                            dataset_name=dataset_name)


def main():
    dataset_id = sys.argv[1]
    if dataset_id == 'ALL':
        dataset_ids = [0, 1, 2, 3, 4]
    else:
        dataset_ids = [int(dataset_id)]

    for dataset_id in dataset_ids:
        dataset_name, avg_dict, std_dict, mse_dict_random, mse_dict_ga, all_data_results, baseline_results = sample_and_get_results(dataset_id)
        create_plot_output_folder(dataset_name)
        do_plots(dataset_name, avg_dict, std_dict, mse_dict_random, mse_dict_ga, all_data_results, baseline_results)




                     















if __name__ == '__main__':
    main()
