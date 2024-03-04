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

REDUNDANCY = 10 
SAMPLE_SIZES = [50, 100, 150, 200,250, 300, 350 ,400, 450, 500]

def create_output_folder(dataset_name):
    if not os.path.exists('output'):
        os.makedirs('output')

    if not os.path.exists('output/plots'):
        os.makedirs('output/plots')
    if not os.path.exists(f'output/plots/{dataset_name}'):
        os.makedirs(f'output/plots/{dataset_name}')

    if not os.path.exists('output/raw_numbers'):
        os.makedirs('output/raw_numbers')
    if not os.path.exists(f'output/raw_numbers/{dataset_name}'):
        os.makedirs(f'output/raw_numbers/{dataset_name}')


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
                population_size=10, 
                max_generations=20, 
                mutation_rate=0.6,
                mutation_cap=0.2,
                elite_size=1,
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


def save_dicts_to_csv(dataset_name, avg_dict, std_dict, mse_dict_random, mse_dict_ga, all_data_results, baseline_results):
    baseline_results_for_print = baseline_results.copy()
    baseline_results_for_print['All Data']= all_data_results['Testing Metrics']

    df = pd.DataFrame(avg_dict)
    df.to_csv(f'output/raw_numbers/{dataset_name}/avg_dict.csv')

    df = pd.DataFrame(std_dict)
    df.to_csv(f'output/raw_numbers/{dataset_name}/std_dict.csv')

    df = pd.DataFrame(mse_dict_random)
    df.to_csv(f'output/raw_numbers/{dataset_name}/mse_dict_random.csv')

    df = pd.DataFrame(mse_dict_ga)
    df.to_csv(f'output/raw_numbers/{dataset_name}/mse_dict_ga.csv')

    df = pd.DataFrame(baseline_results_for_print)
    df.to_csv(f'output/raw_numbers/{dataset_name}/baseline_results.csv')

    df = pd.DataFrame(all_data_results)
    df.to_csv(f'output/raw_numbers/{dataset_name}/all_data_results.csv')

    df = pd.DataFrame(SAMPLE_SIZES)
    df.to_csv(f'output/raw_numbers/{dataset_name}/sample_sizes.csv')


def main():
    dataset_id = sys.argv[1]
    if dataset_id == 'ALL':
        dataset_ids = [0, 1, 2, 3, 4]
    else:
        dataset_ids = [int(dataset_id)]

    for dataset_id in dataset_ids:
        dataset_name, avg_dict, std_dict, mse_dict_random, mse_dict_ga, all_data_results, baseline_results = sample_and_get_results(dataset_id)
        create_output_folder(dataset_name)
        save_dicts_to_csv(dataset_name, avg_dict, std_dict, mse_dict_random, mse_dict_ga, all_data_results, baseline_results)
        do_plots(dataset_name, avg_dict, std_dict, mse_dict_random, mse_dict_ga, all_data_results, baseline_results)




                     















if __name__ == '__main__':
    main()
