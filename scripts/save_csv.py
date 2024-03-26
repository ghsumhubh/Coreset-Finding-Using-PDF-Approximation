import pandas as pd

def save_dicts_to_csv(  dataset_name,
                        avg_dict,
                        std_dict,
                        mse_dicts,
                        labels,
                        all_data_results,
                        baseline_results,sample_sizes):
    
    baseline_results_for_print = baseline_results.copy()
    baseline_results_for_print['All Data']= all_data_results['Testing Metrics']

    df = pd.DataFrame(avg_dict)
    df.to_csv(f'output/raw_numbers/{dataset_name}/avg_dict.csv')

    df = pd.DataFrame(std_dict)
    df.to_csv(f'output/raw_numbers/{dataset_name}/std_dict.csv')

    for mse_dict, label in zip(mse_dicts, labels):
        df = pd.DataFrame(mse_dict)
        df.to_csv(f'output/raw_numbers/{dataset_name}/mse_dict_{label}.csv')

    df = pd.DataFrame(baseline_results_for_print)
    df.to_csv(f'output/raw_numbers/{dataset_name}/baseline_results.csv')

    df = pd.DataFrame(all_data_results)
    df.to_csv(f'output/raw_numbers/{dataset_name}/all_data_results.csv')

    df = pd.DataFrame(sample_sizes)
    df.to_csv(f'output/raw_numbers/{dataset_name}/sample_sizes.csv')