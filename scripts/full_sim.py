from scripts.xgb_results import xgb_results_regression, get_all_data_and_baseline_results
from pprint import pprint
from scripts.utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scripts.genetic import GeneticAlgorithmSampler
from scripts.fitness_funcs import *
from IPython.display import clear_output
from scripts.plots import *
import time
import sys
from scripts.outlier_removal import remove_outliers


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


def sample_and_get_results(train, test,sample_sizes , redundancy):
    # y is the last column
    x_train = train[:, :-1]
    y_train = train[:, -1]

    x_test = test[:, :-1]
    y_test = test[:, -1]

    x_train_no_outliers_2, y_train_no_outliers_2 = remove_outliers(x_train, y_train, outlier_threshold=2)
    x_train_no_outliers_25, y_train_no_outliers_25 = remove_outliers(x_train, y_train, outlier_threshold=2.5)
    x_train_no_outliers_3, y_train_no_outliers_3 = remove_outliers(x_train, y_train, outlier_threshold=3)
    x_train_no_outliers_35, y_train_no_outliers_35 = remove_outliers(x_train, y_train, outlier_threshold=3)
    x_train_no_outliers_4, y_train_no_outliers_4 = remove_outliers(x_train, y_train, outlier_threshold=4)


    all_data_results, baseline_results = get_all_data_and_baseline_results(x_train, x_test, y_train, y_test)


    mse_dict_random = {}
    mse_dict_random_no_outliers_2 = {}
    mse_dict_random_no_outliers_25 = {}
    mse_dict_random_no_outliers_3 = {}
    mse_dict_random_no_outliers_35 = {}
    mse_dict_random_no_outliers_4 = {}
    mse_dict_ga = {}
    mse_dict_reverse_ga = {}



    for _, sample_size in enumerate(sample_sizes):
        clear_output()
        print('Sample Size:', sample_size, '\n')

        mse_dict_random[sample_size] = []
        mse_dict_random_no_outliers_2[sample_size] = []
        mse_dict_random_no_outliers_25[sample_size] = []
        mse_dict_random_no_outliers_3[sample_size] = []
        mse_dict_random_no_outliers_35[sample_size] = []
        mse_dict_random_no_outliers_4[sample_size] = []
        mse_dict_ga[sample_size] = []
        mse_dict_reverse_ga[sample_size] = []
        
        for i in range(redundancy):
            print('Iteration {}/{}'.format(i+1, redundancy))
            np.random.seed(i)
            
            x_train_random, y_train_random = sample_data(x_train, y_train, sample_size)

            x_train_random_no_outliers_2, y_train_random_no_outliers_2 = sample_data(x_train_no_outliers_2, y_train_no_outliers_2, sample_size)
            x_train_random_no_outliers_25, y_train_random_no_outliers_25 = sample_data(x_train_no_outliers_25, y_train_no_outliers_25, sample_size)
            x_train_random_no_outliers_3, y_train_random_no_outliers_3 = sample_data(x_train_no_outliers_3, y_train_no_outliers_3, sample_size)
            x_train_random_no_outliers_35, y_train_random_no_outliers_35 = sample_data(x_train_no_outliers_35, y_train_no_outliers_35, sample_size)
            x_train_random_no_outliers_4, y_train_random_no_outliers_4 = sample_data(x_train_no_outliers_4, y_train_no_outliers_4, sample_size)


            
            genetic_sampler = GeneticAlgorithmSampler(
                fitness_function='wasserstein_distance',
                sample_size=sample_size,
                x_train=x_train,
                y_train=y_train,
                population_size=10, 
                max_generations=30,
                crossover_rate=0.8, 
                mutation_rate=0.6,
                mutation_cap=0.2,
                elite_size=1,
                verbose=False
            )

            x_train_ga, y_train_ga, history = genetic_sampler.sample()

            reverse_genetic_sampler = GeneticAlgorithmSampler(
                fitness_function='reverse_wasserstein_distance',
                sample_size=sample_size,
                x_train=x_train,
                y_train=y_train,
                population_size=10, 
                max_generations=30,
                crossover_rate=0.8, 
                mutation_rate=0.6,
                mutation_cap=0.2,
                elite_size=1,
                verbose=False
            )

            x_train_reverse_ga, y_train_reverse_ga, history = reverse_genetic_sampler.sample()
            
            results_random = xgb_results_regression(x_train_random, x_test, y_train_random, y_test)
            results_random_no_outliers_2 = xgb_results_regression(x_train_random_no_outliers_2, x_test, y_train_random_no_outliers_2, y_test)
            results_random_no_outliers_25 = xgb_results_regression(x_train_random_no_outliers_25, x_test, y_train_random_no_outliers_25, y_test)
            results_random_no_outliers_3 = xgb_results_regression(x_train_random_no_outliers_3, x_test, y_train_random_no_outliers_3, y_test)
            results_random_no_outliers_35 = xgb_results_regression(x_train_random_no_outliers_35, x_test, y_train_random_no_outliers_35, y_test)
            results_random_no_outliers_4 = xgb_results_regression(x_train_random_no_outliers_4, x_test, y_train_random_no_outliers_4, y_test)
            results_ga = xgb_results_regression(x_train_ga, x_test, y_train_ga, y_test)
            results_reverse_ga = xgb_results_regression(x_train_reverse_ga, x_test, y_train_reverse_ga, y_test)
            
            mse_dict_random[sample_size].append(results_random['Testing Metrics']['MSE'])
            mse_dict_random_no_outliers_2[sample_size].append(results_random_no_outliers_2['Testing Metrics']['MSE'])
            mse_dict_random_no_outliers_25[sample_size].append(results_random_no_outliers_25['Testing Metrics']['MSE'])
            mse_dict_random_no_outliers_3[sample_size].append(results_random_no_outliers_3['Testing Metrics']['MSE'])
            mse_dict_random_no_outliers_35[sample_size].append(results_random_no_outliers_35['Testing Metrics']['MSE'])
            mse_dict_random_no_outliers_4[sample_size].append(results_random_no_outliers_4['Testing Metrics']['MSE'])
            mse_dict_ga[sample_size].append(results_ga['Testing Metrics']['MSE'])
            mse_dict_reverse_ga[sample_size].append(results_reverse_ga['Testing Metrics']['MSE'])


    avg_dict ={}
    std_dict = {}

    avg_dict['Random'] = []
    std_dict['Random'] = []
    std_dict['Random No Outliers 2'] = []
    avg_dict['Random No Outliers 2'] = []
    std_dict['Random No Outliers 2.5'] = []
    avg_dict['Random No Outliers 2.5'] = []
    std_dict['Random No Outliers 3'] = []
    avg_dict['Random No Outliers 3'] = []
    std_dict['Random No Outliers 3.5'] = []
    avg_dict['Random No Outliers 3.5'] = []
    std_dict['Random No Outliers 4'] = []
    avg_dict['Random No Outliers 4'] = []
    avg_dict['GA'] = []
    std_dict['GA'] = []
    avg_dict['Reverse GA'] = []
    std_dict['Reverse GA'] = []

    for sample_size in sample_sizes:
        avg_dict['Random'].append(np.mean(mse_dict_random[sample_size]))
        std_dict['Random'].append(np.std(mse_dict_random[sample_size]))
        avg_dict['Random No Outliers 2'].append(np.mean(mse_dict_random_no_outliers_2[sample_size]))
        std_dict['Random No Outliers 2'].append(np.std(mse_dict_random_no_outliers_2[sample_size]))
        avg_dict['Random No Outliers 2.5'].append(np.mean(mse_dict_random_no_outliers_25[sample_size]))
        std_dict['Random No Outliers 2.5'].append(np.std(mse_dict_random_no_outliers_25[sample_size]))
        avg_dict['Random No Outliers 3'].append(np.mean(mse_dict_random_no_outliers_3[sample_size]))
        std_dict['Random No Outliers 3'].append(np.std(mse_dict_random_no_outliers_3[sample_size]))
        avg_dict['Random No Outliers 3.5'].append(np.mean(mse_dict_random_no_outliers_35[sample_size]))
        std_dict['Random No Outliers 3.5'].append(np.std(mse_dict_random_no_outliers_35[sample_size]))
        avg_dict['Random No Outliers 4'].append(np.mean(mse_dict_random_no_outliers_4[sample_size]))
        std_dict['Random No Outliers 4'].append(np.std(mse_dict_random_no_outliers_4[sample_size]))
        avg_dict['GA'].append(np.mean(mse_dict_ga[sample_size]))
        std_dict['GA'].append(np.std(mse_dict_ga[sample_size]))
        avg_dict['Reverse GA'].append(np.mean(mse_dict_reverse_ga[sample_size]))
        std_dict['Reverse GA'].append(np.std(mse_dict_reverse_ga[sample_size]))

    return avg_dict, std_dict, mse_dict_random, mse_dict_random_no_outliers_2, mse_dict_random_no_outliers_25, mse_dict_random_no_outliers_3, mse_dict_random_no_outliers_35, mse_dict_random_no_outliers_4, mse_dict_ga, mse_dict_reverse_ga, all_data_results, baseline_results

def do_plots(   dataset_name,
                avg_dict,
                std_dict,
                mse_dict_random,
                mse_dict_random_no_outliers_2,
                mse_dict_random_no_outliers_25,
                mse_dict_random_no_outliers_3,
                mse_dict_random_no_outliers_35,
                mse_dict_random_no_outliers_4,
                mse_dict_ga,
                mse_dict_reverse_ga,
                all_data_results,
                baseline_results,
                sample_sizes):
    
    plot_comparison_bar(metric_name='MSE',
                        sample_sizes=sample_sizes,
                        avg_dict=avg_dict,
                        std_dict=std_dict,
                        methods=['Random','Random No Outliers 2','Random No Outliers 2.5','Random No Outliers 3','Random No Outliers 3.5','Random No Outliers 4', 'GA', 'Reverse GA'], 
                        save = True,
                        dataset_name=dataset_name)
    



    baseline_results_for_print = baseline_results.copy()
    baseline_results_for_print['All Data']= all_data_results['Testing Metrics']


    plot_comparison_line(metric_name = 'MSE',
                            dictionaries = [mse_dict_random,mse_dict_random_no_outliers_2, mse_dict_random_no_outliers_25, mse_dict_random_no_outliers_3, mse_dict_random_no_outliers_35, mse_dict_random_no_outliers_4, mse_dict_ga, mse_dict_reverse_ga],
                            labels = ['Random','Random No Outliers 2','Random No Outliers 2.5','Random No Outliers 3','Random No Outliers 3.5','Random No Outliers 4', 'GA', 'Reverse GA'],
                            baseline_results = baseline_results_for_print,
                            save = True,
                            dataset_name=dataset_name)


def save_dicts_to_csv(  dataset_name,
                        avg_dict,
                        std_dict,
                        mse_dict_random,
                        mse_dict_random_no_outliers_2,
                        mse_dict_random_no_outliers_25,
                        mse_dict_random_no_outliers_3,
                        mse_dict_random_no_outliers_35,
                        mse_dict_random_no_outliers_4,
                        mse_dict_ga,mse_dict_reverse_ga,
                        all_data_results,
                        baseline_results,sample_sizes):
    
    baseline_results_for_print = baseline_results.copy()
    baseline_results_for_print['All Data']= all_data_results['Testing Metrics']

    df = pd.DataFrame(avg_dict)
    df.to_csv(f'output/raw_numbers/{dataset_name}/avg_dict.csv')

    df = pd.DataFrame(std_dict)
    df.to_csv(f'output/raw_numbers/{dataset_name}/std_dict.csv')

    df = pd.DataFrame(mse_dict_random)
    df.to_csv(f'output/raw_numbers/{dataset_name}/mse_dict_random.csv')

    df = pd.DataFrame(mse_dict_random_no_outliers_2)
    df.to_csv(f'output/raw_numbers/{dataset_name}/mse_dict_random_no_outliers_2.csv')

    df = pd.DataFrame(mse_dict_random_no_outliers_25)
    df.to_csv(f'output/raw_numbers/{dataset_name}/mse_dict_random_no_outliers_2.5.csv')

    df = pd.DataFrame(mse_dict_random_no_outliers_3)
    df.to_csv(f'output/raw_numbers/{dataset_name}/mse_dict_random_no_outliers_3.csv')

    df = pd.DataFrame(mse_dict_random_no_outliers_35)
    df.to_csv(f'output/raw_numbers/{dataset_name}/mse_dict_random_no_outliers_3.5.csv')

    df = pd.DataFrame(mse_dict_random_no_outliers_4)
    df.to_csv(f'output/raw_numbers/{dataset_name}/mse_dict_random_no_outliers_4.csv')

    df = pd.DataFrame(mse_dict_ga)
    df.to_csv(f'output/raw_numbers/{dataset_name}/mse_dict_ga.csv')

    df = pd.DataFrame(mse_dict_reverse_ga)
    df.to_csv(f'output/raw_numbers/{dataset_name}/mse_dict_reverse_ga.csv')

    df = pd.DataFrame(baseline_results_for_print)
    df.to_csv(f'output/raw_numbers/{dataset_name}/baseline_results.csv')

    df = pd.DataFrame(all_data_results)
    df.to_csv(f'output/raw_numbers/{dataset_name}/all_data_results.csv')

    df = pd.DataFrame(sample_sizes)
    df.to_csv(f'output/raw_numbers/{dataset_name}/sample_sizes.csv')


def do_full_simulation(dataset_name, sample_sizes, redundancy):
    if dataset_name == 'ALL':
        dataset_names = ['Abalone',
                         'Insurance',
                         'Melbourne Housing',
                         'Seol Bike',
                         'Sleep Efficiency',
                         'Wine Quality']
        print('Running full simulation for all datasets')
    else:
        dataset_names = [dataset_name]

    for dataset_name in dataset_names:
        print('Running full simulation for', dataset_name)
        create_output_folder(dataset_name)

        train = pd.read_csv(f'data/split_datasets/{dataset_name}/train.csv')
        test = pd.read_csv(f'data/split_datasets/{dataset_name}/test.csv')

        # convert to numpy
        train = train.to_numpy()
        test = test.to_numpy()


        results =  sample_and_get_results(train, test, sample_sizes, redundancy)

        (   avg_dict,
            std_dict,
            mse_dict_random,
            mse_dict_random_no_outliers_2,
            mse_dict_random_no_outliers_25,
            mse_dict_random_no_outliers_3,
            mse_dict_random_no_outliers_35,
            mse_dict_random_no_outliers_4,
            mse_dict_ga,
            mse_dict_reverse_ga,
            all_data_results,
            baseline_results) = results


        save_dicts_to_csv(  dataset_name = dataset_name,
                            avg_dict=avg_dict,
                            std_dict=std_dict,
                            mse_dict_random=mse_dict_random,
                            mse_dict_random_no_outliers_2=mse_dict_random_no_outliers_2,
                            mse_dict_random_no_outliers_25=mse_dict_random_no_outliers_25,
                            mse_dict_random_no_outliers_3=mse_dict_random_no_outliers_3,
                            mse_dict_random_no_outliers_35=mse_dict_random_no_outliers_35,
                            mse_dict_random_no_outliers_4=mse_dict_random_no_outliers_4,
                            mse_dict_ga=mse_dict_ga,
                            mse_dict_reverse_ga=mse_dict_reverse_ga,
                            all_data_results=all_data_results,
                            baseline_results=baseline_results,
                            sample_sizes=sample_sizes)
        
        do_plots(dataset_name = dataset_name,
                            avg_dict=avg_dict,
                            std_dict=std_dict,
                            mse_dict_random=mse_dict_random,
                            mse_dict_random_no_outliers_2=mse_dict_random_no_outliers_2,
                            mse_dict_random_no_outliers_25=mse_dict_random_no_outliers_25,
                            mse_dict_random_no_outliers_3=mse_dict_random_no_outliers_3,
                            mse_dict_random_no_outliers_35=mse_dict_random_no_outliers_35,
                            mse_dict_random_no_outliers_4=mse_dict_random_no_outliers_4,
                            mse_dict_ga=mse_dict_ga,
                            mse_dict_reverse_ga=mse_dict_reverse_ga,
                            all_data_results=all_data_results,
                            baseline_results=baseline_results,
                            sample_sizes=sample_sizes)
