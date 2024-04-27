import pandas as pd

from scripts.utils import *
from scripts.save_csv import save_dicts_to_csv
from scripts.fitness_funcs import *
from scripts.plots import *
from scripts.sample_loop import sample_and_get_results


ALL_DATASET_NAMES = [   'Abalone',
                    'Insurance',
                    'Melbourne Housing',
                    'Seoul Bike',
                    'Sleep Efficiency',
                    'Wine Quality']



def do_full_simulation(dataset_name, sample_sizes, redundancy, columns_to_use=None, weights = None):
    print('Running full simulation for', dataset_name)
    create_output_folder(dataset_name)

    train = pd.read_csv(f'data/split_datasets/{dataset_name}/train.csv')
    test = pd.read_csv(f'data/split_datasets/{dataset_name}/test.csv')


    results =  sample_and_get_results(
        dataset_name=dataset_name,
        train=train,
        test=test,
        sample_sizes=sample_sizes,
        redundancy=redundancy,
        columns_to_use=columns_to_use,
        weights=weights)

    (   avg_dict,
        std_dict,
        mse_dicts,
        labels,
        all_data_results,
        baseline_results) = results


    save_dicts_to_csv(  dataset_name = dataset_name,
                        avg_dict=avg_dict,
                        std_dict=std_dict,
                        mse_dicts=mse_dicts,
                        labels=labels,
                        all_data_results=all_data_results,
                        baseline_results=baseline_results,
                        sample_sizes=sample_sizes)
    
    do_sim_plots(dataset_name = dataset_name,
                        avg_dict=avg_dict,
                        std_dict=std_dict,
                        mse_dicts = mse_dicts,
                        labels=labels,
                        all_data_results=all_data_results,
                        baseline_results=baseline_results,
                        sample_sizes=sample_sizes)
    
    print('Finished running full simulation')